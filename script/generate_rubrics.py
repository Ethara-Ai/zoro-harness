#!/usr/bin/env python3
"""
generate_rubrics.py — Author rubrics.json for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
trajectory (solution/tool_calls.jsonl), the task's TRUTH.md, and calls
Claude via the local claude_bridge using the system prompt at
`truth/rubrics-plan.md`. Writes the generated JSON array into
`<task_dir>/tests/rubrics.json` (creating the tests/ subdirectory if needed).

Prerequisites (see `claude_bridge/bridge-setup.md`):

    export ZORO_CC_BRIDGE_SECRET="your-random-string"
    ./script/start_claude_bridge.sh start

Usage:

    python truth/generate_rubrics.py <task_dir>
    python truth/generate_rubrics.py <task_dir> --model sonnet --overwrite
    python truth/generate_rubrics.py <task_dir> --dry-run
    python truth/generate_rubrics.py <task_dir> --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_truth import (  # noqa: E402
    call_llm,
    die,
    discover_task_files,
    extract_task_config,
    measure_oracle_trajectory,
    read_json,
    read_toml,
    strip_fenced_wrapper,
)


DEFAULT_MODEL = "sonnet"
DEFAULT_OUTPUT_NAME = "rubrics.json"
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 0.2
RUBRICS_PLAN_PATH = Path(__file__).resolve().parent.parent / "truth" / "rubrics-plan.md"

ALLOWED_TYPES = {
    "setup", "pricing", "supplier", "reorder", "stockout",
    "portfolio", "cash", "strategy", "endgame", "meta_behavior",
}
ALLOWED_IMPORTANCE = {"critically_important", "important", "minor"}
SCORE_MAGNITUDE = {"critically_important": 5, "important": 3, "minor": 1}


def _find_truth_md(task_dir: Path) -> Path:
    for name in ("TRUTH.md", "truth.md"):
        p = task_dir / name
        if p.is_file():
            return p
    die(f"no TRUTH.md or truth.md found in {task_dir}. Run generate_truth.py first.")
    return task_dir


def build_user_message(config: dict, golden: dict, stats: dict, truth_text: str) -> str:
    cats = sorted(config.get("selected_categories") or [])
    price_confirm_pct = stats["price_confirmation_rate"] * 100

    return f"""Author the `rubrics.json` file for the following task, following the system-prompt specification.

## Task configuration (from dataset.json + task.toml)

- task_id: {config["task_id"]}
- store_id: {config["store_id"]}
- days (horizon): {config["days"]}
- initial_funds: ${config["initial_funds"]}
- everyday_rent: ${config["everyday_rent"]}
- inventory_capacity: {config["inventory_capacity"]} units
- selected_categories ({len(cats)}, alphabetized): {", ".join(cats)}
- category_cannibalization_coefficient: {config["category_cannibalization_coefficient"]}
- enable_review: {config["enable_review"]}
- enable_new: {config["enable_new"]}
- global_random_seed: {config["global_random_seed"]}

Note: `category_effect = {config["category_cannibalization_coefficient"]}` is the CROSS-SKU CANNIBALIZATION coefficient (NOT price elasticity). Do not draft a rubric that treats it as sensitivity of one SKU's own demand to its own price.

## Oracle terminal state (from golden.json)

- final_net_worth: ${golden.get("final_net_worth"):.2f}
- days_completed: {golden.get("days_completed")} of {golden.get("days_requested")}

## Oracle aggregate measurements

- Total orders: {stats["total_orders"]} (~{stats["orders_per_day"]:.1f} per day)
- Price-touch confirmation rate: {price_confirm_pct:.1f}%
- Days with at least one stockout: {stats["stockout_day_count"]}

## TRUTH.md contents

Base positive rubrics on the Decision-quality and Mixed-reasoning requirements. Base negative rubrics on the named Failure modes. Do not invent rubrics that lack a corresponding line in TRUTH.md.

---
{truth_text}
---

Return only the JSON array. No preamble. No fenced code block wrapper. First character = `[`, last non-whitespace character = `]`.
"""


def _validate_rubrics(rubrics: list) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(rubrics, list):
        return False, [f"top-level must be a JSON array, got {type(rubrics).__name__}"]
    if not (8 <= len(rubrics) <= 12):
        errors.append(f"rubric count {len(rubrics)} outside 8..12 range")

    seen_numbers = []
    for i, r in enumerate(rubrics):
        prefix = f"rubric[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{prefix}: must be object, got {type(r).__name__}")
            continue

        for field in ("number", "criterion", "is_positive", "type", "importance", "score"):
            if field not in r:
                errors.append(f"{prefix}: missing field '{field}'")

        num = r.get("number", "")
        if not (isinstance(num, str) and num.startswith("R") and num[1:].isdigit()):
            errors.append(f"{prefix}: number '{num}' does not match R<digits>")
        else:
            seen_numbers.append(int(num[1:]))

        crit = r.get("criterion", "")
        if not (isinstance(crit, str) and crit.startswith("The agent") and crit.rstrip().endswith(".")):
            errors.append(f"{prefix}: criterion must start with 'The agent' and end with '.'")

        if not isinstance(r.get("is_positive"), bool):
            errors.append(f"{prefix}: is_positive must be bool")

        if r.get("type") not in ALLOWED_TYPES:
            errors.append(f"{prefix}: type '{r.get('type')}' not in allowed set")

        imp = r.get("importance")
        if imp not in ALLOWED_IMPORTANCE:
            errors.append(f"{prefix}: importance '{imp}' not in allowed set")

        score = r.get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            errors.append(f"{prefix}: score must be int")
        elif imp in SCORE_MAGNITUDE:
            expected_mag = SCORE_MAGNITUDE[imp]
            if abs(score) != expected_mag:
                errors.append(f"{prefix}: |score|={abs(score)} does not match importance '{imp}' magnitude {expected_mag}")
            if r.get("is_positive") is True and score < 0:
                errors.append(f"{prefix}: is_positive=true but score<0")
            if r.get("is_positive") is False and score > 0:
                errors.append(f"{prefix}: is_positive=false but score>0")

    if seen_numbers:
        expected = list(range(1, len(seen_numbers) + 1))
        if sorted(seen_numbers) != expected:
            errors.append(f"numbering not contiguous R1..R{len(seen_numbers)}: got {sorted(seen_numbers)}")

    return (not errors), errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Author rubrics.json for a harbor-format RetailBench task via claude_bridge."
    )
    ap.add_argument("task_dir", type=Path, help="Path to the task directory (harbor format).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    ap.add_argument(
        "--rubrics-plan",
        type=Path,
        default=RUBRICS_PLAN_PATH,
        help=f"Path to rubrics-plan.md system prompt (default: {RUBRICS_PLAN_PATH}).",
    )
    ap.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Filename to write inside <task_dir>/tests/ (default: {DEFAULT_OUTPUT_NAME}).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true", help="Print prompt + user message; do not call LLM or write.")
    ap.add_argument("--stats-only", action="store_true", help="Print measured aggregate stats and exit.")
    args = ap.parse_args()

    if not args.task_dir.is_dir():
        die(f"not a directory: {args.task_dir}")
    if not args.rubrics_plan.is_file():
        die(f"rubrics-plan.md not found at: {args.rubrics_plan}")

    tests_dir = args.task_dir / "tests"
    output_path = tests_dir / args.output_name

    files = discover_task_files(args.task_dir)
    task_toml = read_toml(files["task_toml"])
    dataset = read_json(files["dataset_json"])
    golden = read_json(files["golden_json"])

    config = extract_task_config(task_toml, dataset)
    if not config["task_id"]:
        die("task.toml missing [metadata].id")

    truth_path = _find_truth_md(args.task_dir)
    truth_text = truth_path.read_text()

    print(f"[generate_rubrics] measuring oracle trajectory: {files['tool_calls_jsonl']}", file=sys.stderr)
    stats = measure_oracle_trajectory(files["tool_calls_jsonl"])

    if args.stats_only:
        print(json.dumps({"config": config, "golden": golden, "stats": stats}, indent=2, default=str))
        return 0

    if output_path.exists() and not args.overwrite and not args.dry_run:
        die(f"output exists: {output_path}. Use --overwrite to replace, or --dry-run to preview.")

    system_prompt = args.rubrics_plan.read_text()
    user_message = build_user_message(config, golden, stats, truth_text)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.rubrics_plan.name}, first 40 lines):")
        print("=" * 72)
        print("\n".join(system_prompt.splitlines()[:40]))
        print("...")
        print("=" * 72)
        print("USER MESSAGE:")
        print("=" * 72)
        print(user_message)
        return 0

    print(f"[generate_rubrics] calling bridge: model={args.model}", file=sys.stderr)
    raw = call_llm(system_prompt, user_message, args.model, args.max_tokens, args.temperature)
    content = strip_fenced_wrapper(raw)

    try:
        rubrics = json.loads(content)
    except json.JSONDecodeError as e:
        preview = content[:500].replace("\n", " ")
        die(f"LLM output is not valid JSON: {e}\npreview: {preview}")
        return 1

    ok, errors = _validate_rubrics(rubrics)
    if not ok:
        for err in errors:
            print(f"[generate_rubrics] validation error: {err}", file=sys.stderr)
        die("rubrics failed validation. Re-run with --dry-run to inspect the prompt, or retry.")

    tests_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rubrics, indent=2, ensure_ascii=False) + "\n")
    print(f"[generate_rubrics] wrote {output_path} ({len(rubrics)} rubrics)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

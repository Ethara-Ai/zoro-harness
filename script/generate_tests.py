#!/usr/bin/env python3
"""
generate_tests.py — Author test_outputs.py for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
trajectory (solution/tool_calls.jsonl), the task's TRUTH.md, and calls
Claude via the local claude_bridge using the system prompt at
`truth/tests-plan.md`. Writes the generated Python module into
`<task_dir>/tests/test_outputs.py` (creating the tests/ subdirectory if
needed).

Prerequisites (see `claude_bridge/bridge-setup.md`):

    export ZORO_CC_BRIDGE_SECRET="your-random-string"
    ./script/start_claude_bridge.sh start

Usage:

    python truth/generate_tests.py <task_dir>
    python truth/generate_tests.py <task_dir> --model sonnet --overwrite
    python truth/generate_tests.py <task_dir> --dry-run
    python truth/generate_tests.py <task_dir> --stats-only
"""

from __future__ import annotations

import argparse
import ast
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
DEFAULT_OUTPUT_NAME = "test_outputs.py"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.2
TESTS_PLAN_PATH = Path(__file__).resolve().parent.parent / "truth" / "tests-plan.md"

REQUIRED_OUTCOME_IDS = {
    "completes_full_horizon",
    "terminal_solvency",
    "terminal_funds_healthy",
    "terminal_nw_strong",
    "terminal_nw_near_peak",
    "no_trailing_30d_nw_crash",
    "no_cash_paralysis",
    "no_portfolio_collapse",
    "return_burden_bounded",
}
REQUIRED_PROCESS_IDS = {
    "day1_price_correction_called",
    "day1_order_placed",
    "supplier_board_before_order",
    "view_orders_before_place",
    "reorder_after_stockout_bounded",
    "no_price_stagnation",
    "price_variance_nontrivial",
    "no_prolonged_order_drought",
    "low_signal_drift_bounded",
}
KNOCKOUT_IDS = {
    "completes_full_horizon",
    "terminal_solvency",
    "no_trailing_30d_nw_crash",
    "no_portfolio_collapse",
}


def _find_truth_md(task_dir: Path) -> Path:
    for name in ("TRUTH.md", "truth.md"):
        p = task_dir / name
        if p.is_file():
            return p
    die(f"no TRUTH.md or truth.md found in {task_dir}. Run generate_truth.py first.")
    return task_dir


def _estimate_oracle_active_skus(stats: dict, config: dict) -> int:
    n_cats = len(config.get("selected_categories") or [])
    default = max(3, 2 * n_cats)
    return int(stats.get("oracle_active_skus") or default)


def build_user_message(config: dict, golden: dict, stats: dict, truth_text: str) -> str:
    cats = sorted(config.get("selected_categories") or [])
    price_confirm_pct = stats["price_confirmation_rate"] * 100
    oracle_nw = float(golden.get("final_net_worth") or 0.0)
    oracle_active_skus = _estimate_oracle_active_skus(stats, config)

    return f"""Author the `test_outputs.py` file for the following task, following the system-prompt specification.

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

Note: `category_effect = {config["category_cannibalization_coefficient"]}` is the CROSS-SKU CANNIBALIZATION coefficient (NOT price elasticity).

## Oracle terminal state (from golden.json)

- final_net_worth (ORACLE_NW): ${oracle_nw:.2f}
- days_completed: {golden.get("days_completed")} of {golden.get("days_requested")}

## Oracle aggregate measurements

- Total orders: {stats["total_orders"]} (~{stats["orders_per_day"]:.1f} per day)
- Price-touch confirmation rate: {price_confirm_pct:.1f}%
- Days with at least one stockout: {stats["stockout_day_count"]}
- Cash trough: day {stats["cash_trough_day"]} at ${stats["cash_trough_value"]:.2f}
- Steady-state NW slope: ${stats["steady_state_nw_slope_per_day"]:.2f} / day
- Measured terminal NW: ${stats["measured_terminal_nw"]:.2f}
- Measured terminal cash: ${stats["measured_terminal_cash"]:.2f}
- ORACLE_ACTIVE_SKUS (estimate): {oracle_active_skus}

## Threshold constants you MUST inline verbatim in the file

Compute and hardcode these at the top of the module (immediately below imports), then reference them in the tests:

- `F = {config["initial_funds"]}` (initial_funds)
- `DAYS = {config["days"]}` (horizon days requested)
- `ORACLE_NW = {oracle_nw:.2f}`
- `ORACLE_ACTIVE_SKUS = {oracle_active_skus}`
- `SOLVENCY_NW = min(F * 4.0, ORACLE_NW * 0.6)`
- `K_STRONG = max(2.0, round((ORACLE_NW / F) * 0.5, 1))`
- `PORTFOLIO_FLOOR = max(3, int(0.5 * ORACLE_ACTIVE_SKUS))`
- `CRASH_FRAC = 0.20`
- `PEAK_FRAC = 0.80`
- `CASH_IDLE_FRAC = 2.0`  # multiplier on F
- `STALL_FRAC = 0.02`
- `RETURN_RATE_MAX = 0.15`
- `SUPPLIER_BOARD_FRAC = 0.85`
- `VIEW_ORDERS_FRAC = 0.80`
- `STOCKOUT_REORDER_WINDOW = 7`
- `PRICE_STAGNATION_GAP = 30`
- `PRICE_CV_MIN = 0.001`
- `DROUGHT_LEN = 20`
- `DROUGHT_TAIL_EXCLUSION = 15`
- `LOWSIGNAL_RATIO_MAX = 5.0`

## Low-signal tool set (task-config dependent — hardcode based on this task)

- Base always: `view_inventory`, `view_funds_and_date`, `add_note`, `view_notes`
- Add `view_today_news` IFF `enable_new = False` (this task: enable_new={config["enable_new"]} → {"include" if not config["enable_new"] else "OMIT"} `view_today_news`)
- NEVER include review tools (enable_review is True or irrelevant to the low-signal set)

## TRUTH.md contents

Use TRUTH.md's Requirements section as the ground truth of what the tests must cover. Do not invent tests outside the outcome/process rosters in the system prompt.

---
{truth_text}
---

Return only the Python source. No preamble. No fenced code block wrapper. First non-blank line = the module docstring or an import.
"""


def _extract_check_marker_ids(module: ast.Module) -> tuple[set[str], set[str], set[str]]:
    outcome_ids: set[str] = set()
    process_ids: set[str] = set()
    knockout_ids: set[str] = set()

    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            attr_chain = _dotted_name(dec.func)
            if attr_chain not in ("pytest.mark.check", "mark.check"):
                continue
            kwargs = {kw.arg: kw.value for kw in dec.keywords if kw.arg}
            id_val = _literal_str(kwargs.get("id"))
            kind_val = _literal_str(kwargs.get("kind"))
            ko_val = _literal_bool(kwargs.get("knockout"))
            if id_val is None:
                continue
            if kind_val == "outcome":
                outcome_ids.add(id_val)
            elif kind_val == "process":
                process_ids.add(id_val)
            if ko_val is True:
                knockout_ids.add(id_val)

    return outcome_ids, process_ids, knockout_ids


def _dotted_name(node: ast.expr) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _literal_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_bool(node: ast.expr | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _validate_test_module(source: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        module = ast.parse(source)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    outcome_ids, process_ids, knockout_ids = _extract_check_marker_ids(module)

    missing_outcome = REQUIRED_OUTCOME_IDS - outcome_ids
    missing_process = REQUIRED_PROCESS_IDS - process_ids
    missing_ko = KNOCKOUT_IDS - knockout_ids
    extra_ko = knockout_ids - KNOCKOUT_IDS

    for mid in sorted(missing_outcome):
        errors.append(f"missing outcome test id: {mid}")
    for mid in sorted(missing_process):
        errors.append(f"missing process test id: {mid}")
    for mid in sorted(missing_ko):
        errors.append(f"knockout=True missing for required id: {mid}")
    for xid in sorted(extra_ko):
        errors.append(f"knockout=True on non-approved id: {xid}")

    if outcome_ids & process_ids:
        errors.append(f"ids appear as both outcome and process: {sorted(outcome_ids & process_ids)}")

    return (not errors), errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Author test_outputs.py for a harbor-format RetailBench task via claude_bridge."
    )
    ap.add_argument("task_dir", type=Path, help="Path to the task directory (harbor format).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    ap.add_argument(
        "--tests-plan",
        type=Path,
        default=TESTS_PLAN_PATH,
        help=f"Path to tests-plan.md system prompt (default: {TESTS_PLAN_PATH}).",
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
    if not args.tests_plan.is_file():
        die(f"tests-plan.md not found at: {args.tests_plan}")

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

    print(f"[generate_tests] measuring oracle trajectory: {files['tool_calls_jsonl']}", file=sys.stderr)
    stats = measure_oracle_trajectory(files["tool_calls_jsonl"])

    if args.stats_only:
        print(json.dumps({"config": config, "golden": golden, "stats": stats}, indent=2, default=str))
        return 0

    if output_path.exists() and not args.overwrite and not args.dry_run:
        die(f"output exists: {output_path}. Use --overwrite to replace, or --dry-run to preview.")

    system_prompt = args.tests_plan.read_text()
    user_message = build_user_message(config, golden, stats, truth_text)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.tests_plan.name}, first 40 lines):")
        print("=" * 72)
        print("\n".join(system_prompt.splitlines()[:40]))
        print("...")
        print("=" * 72)
        print("USER MESSAGE:")
        print("=" * 72)
        print(user_message)
        return 0

    print(f"[generate_tests] calling bridge: model={args.model}", file=sys.stderr)
    raw = call_llm(system_prompt, user_message, args.model, args.max_tokens, args.temperature)
    content = strip_fenced_wrapper(raw)

    ok, errors = _validate_test_module(content)
    if not ok:
        for err in errors:
            print(f"[generate_tests] validation error: {err}", file=sys.stderr)
        die("test module failed validation. Re-run with --dry-run to inspect the prompt, or retry.")

    tests_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content if content.endswith("\n") else content + "\n")
    print(f"[generate_tests] wrote {output_path} ({len(content.splitlines())} lines)", file=sys.stderr)

    conftest_path = tests_dir / "conftest.py"
    if not conftest_path.exists():
        conftest_path.write_text("from llm_council.pytest_bridge import *  # noqa: F401,F403\n")
        print(f"[generate_tests] wrote {conftest_path}", file=sys.stderr)
    else:
        print(f"[generate_tests] conftest.py already exists at {conftest_path}, leaving in place", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

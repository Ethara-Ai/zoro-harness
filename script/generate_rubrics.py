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

RUBRICS_PLAN = r'''# rubrics.json authoring plan (LLM system prompt)

You are authoring `rubrics.json` for one RetailBench task. This file lives at `<task_dir>/tests/rubrics.json`. It holds the qualitative-judgment rubrics that the LLM council will evaluate against a run's trajectory. Quantitative checks live elsewhere in `tests/test_outputs.py` — they are NOT your concern.

---

## 1. Role

You produce a single JSON array of rubric objects. Each rubric is one atomic question a judge model can answer by reading the agent's strategy text and tool-call evidence. Nothing else appears in the output — no markdown, no prose, no code fences, no keys other than the rubric fields.

Your inputs are:
- The task's frozen configuration (from `dataset.json` + `task.toml`).
- The task's `TRUTH.md` (already authored, describes what the task is, what a strong run looks like, requirements, failure modes).
- Aggregate oracle measurements (already computed deterministically — days completed, terminal net worth, price-confirmation rate, order count, stockout-day count, steady-state slope).

Your output is:
- One JSON array of 8–12 rubric objects, encoded UTF-8, no BOM, no trailing comma, no comments, no wrapping code fence. The array is the entire file contents.

---

## 2. Design principle: judge presence, aggregator applies polarity

Every rubric criterion is phrased so that a judge answers exactly one question: **is the pattern described by the criterion present in the evidence?**

- verdict `1` = pattern present
- verdict `0` = pattern absent
- verdict `null` = evidence insufficient to decide

The polarity (`is_positive: true|false`) and score are metadata the aggregator uses. The judge does not decide good/bad. This matters: a negative rubric describes a bad pattern; verdict=1 means the bad pattern WAS present in the run.

You never write "the agent should do X" or "did the agent handle X well". You write "the agent's strategy text on some day cites past-day sales as the reason for a shelf-price change" — and let the aggregator apply the sign.

---

## 3. Output schema

The output is a JSON array. Each element is an object with these fields, in this order:

```json
{
  "number": "R1",
  "criterion": "The agent's strategy text on at least one day cites a specific past-day sales result (units sold, sales trend, or revenue level) as the stated reason for a shelf-price change.",
  "is_positive": true,
  "type": "pricing",
  "importance": "important",
  "score": 3
}
```

Field rules:

- **number**: string of form `"R1"`, `"R2"`, ..., contiguous starting at R1 in file order. No gaps, no reuse.
- **criterion**: one English sentence. Starts with `"The agent"`. Atomic (one testable pattern). Self-contained (does not reference "the task" abstractly, does not require the judge to re-read TRUTH.md). Ends with a period.
- **is_positive**: `true` if the pattern is desirable (present=good), `false` if the pattern is undesirable (present=bad, i.e., a failure mode).
- **type**: exactly one of `"setup"`, `"pricing"`, `"supplier"`, `"reorder"`, `"stockout"`, `"portfolio"`, `"cash"`, `"strategy"`, `"endgame"`, `"meta_behavior"`.
- **importance**: exactly one of `"critically_important"`, `"important"`, `"minor"`.
- **score**: signed integer. Magnitude tied to importance: `critically_important` → 5, `important` → 3, `minor` → 1. Sign follows `is_positive`: positive rubric → positive score, negative rubric → negative score.

Nothing else in the object. No `weight`, no `knockout` (rubrics never knock out), no `id`, no free-form fields.

---

## 4. Authoring rules (Gate 2)

Every rubric MUST satisfy all six rules. If a rubric fails any rule, drop it or reformulate it. Do not ship a broken rubric to hit a count target.

### 4.1 Atomic

The criterion tests exactly one pattern. If your sentence contains "and" joining two independent claims that could be evaluated separately, split it. Exception: a compound of the form "X and X's stated reason is Y" is atomic when the reason is inseparable from the action.

### 4.2 Self-contained

A judge reading only the criterion + evidence can answer. Do not reference "the reference playbook", "the failure modes list", "as described in TRUTH.md". If the concept matters, name it directly.

### 4.3 Non-redundant

No two rubrics measure the same pattern in the same evidence. Rewording is not enough distance.

### 4.4 Correctly-layered

The criterion MUST fail all six of these questions. If any question is a "yes", the check belongs in pytest, not in a rubric:

1. Can this be answered by counting tool calls in a fixed window?
2. Can this be answered by checking whether tool A precedes tool B on a specific day?
3. Can this be answered by comparing a numeric field against a threshold?
4. Can this be answered by computing a ratio of two counts?
5. Can this be answered by finding a streak or continuous run of days?
6. Can this be answered by comparing values from two specific days?

Every rubric criterion requires reading semantic content of strategy text or reasoning intent that a judge model interprets. If a script could answer the criterion mechanically, it is the wrong layer.

### 4.5 Correctly-polarized

Positives come from TRUTH.md `Decision-quality` requirements and the reasoning half of `Mixed` requirements. Negatives come from TRUTH.md `Failure modes`. Do not invert a failure mode to make a positive rubric — the negative rubric already covers it.

### 4.6 Correctly-typed

Choose the type that most narrowly describes the criterion's domain. A rubric about supplier-choice reasoning is `supplier`, not `strategy`. A rubric about tool-use discipline is `meta_behavior`. Reserve `strategy` for cross-cutting posture (assortment stability, adaptation coherence).

---

## 5. Verdict-shape rules the criterion must respect

These follow from §6.2 and §6.4 of the pipeline spec. A well-authored criterion respects them by construction.

- The judge returns `null` when the triggering event never occurred (negative rubric with no relevant events → null, dropped from scoring — NOT `0` "avoided=good"). Phrase criteria so this null path is obvious to the judge. Example: a stockout-response rubric requires the trajectory to actually contain stockout events; if none, judge returns null.
- Positive rubrics returning null on thin evidence are expected and safe. Do not weight-inflate to compensate.
- Rubrics NEVER knock out. Knockouts live in pytest.

---

## 6. Roster guidance

Target 8–12 rubrics. Aim for a mix roughly like 7 positives + 3 negatives, adjusted to what TRUTH.md actually contains. Coverage across types matters more than hitting an exact count.

Use the following slots as a starting menu. Include one only if TRUTH.md's Requirements or Failure modes support it. Skip freely.

### Positive rubrics (from Decision-quality and Mixed-reasoning requirements)

- **pricing — evidence-grounded pricing**: agent's strategy text names an observed sales result (unit velocity, sell-through, trend) as the reason for a specific shelf-price move. Judges evidence quality of pricing reasoning.
- **pricing — margin/cost awareness**: agent's strategy text names cost, margin, or supplier price as part of a pricing decision. Distinct from the above — this checks whether cost enters the reasoning, not just sales response.
- **supplier — coherent supplier tradeoff**: agent's strategy text names a specific supplier criterion (cost per unit, quality score, lead time, past reliability) as decisive when picking one supplier over another. Do NOT reward supplier diversity per se. If the same best-quality-score supplier is used consistently and the reasoning cites that criterion, verdict=1.
- **reorder — velocity/coverage-based sizing**: agent's strategy text on at least one reorder day explains order quantity in terms of expected days-of-cover, observed velocity, or lead time — not a fixed lot.
- **stockout — investigative response**: on the day after a stockout or sales dip, agent's strategy text describes checking evidence (sales history, reviews, supplier availability) rather than reflexively cutting price or bulk-reordering. Judge returns null if no stockouts or dips occurred.
- **strategy — coherent assortment**: agent's strategy text across the middle-run days is consistent about which SKUs it is prioritizing; assortment is not thrashed every few days without reasoned justification.
- **meta_behavior — purposeful tool use**: agent's strategy text distinguishes tool calls made to answer a specific question from routine sweeps, and does not repeatedly re-query unchanged data.
- **setup — coherent opening plan**: agent's day-1 strategy text names a rationale for the SKU set it stocks (category tradeoffs, expected margin, cash budget) rather than picking SKUs opaquely.

### Positive rubrics conditional on task config

- **meta_behavior — news integration** (only if `enable_new=true`): agent's strategy text on at least one day references news content as reasoning for a decision. If `enable_new=false`, OMIT this rubric — news tools return inert content and reasoning references would be spurious.
- **supplier — quality-cost tradeoff for returns**: agent's strategy text references quality_score as trading off higher unit cost against lower expected returns/refunds. Include when returns are a live economic force in the task. Judge returns null if no returns fired in the run.

### Negative rubrics (from TRUTH.md Failure modes)

- **setup — bootstrap paralysis or overshoot**: day-1 shelf prices are left at implausible env defaults OR are set to non-viable per-unit margins (below stated cost). Compound criterion, both branches captured — verdict=1 if either branch holds.
- **pricing — panic pricing**: on a single low-sales day, agent cuts a SKU's shelf price by more than ~25% without strategy-text justification tied to evidence beyond that single day's dip.
- **endgame — incoherent wind-down**: in the final 30 days, agent's strategy text explicitly announces stopping orders, liquidating, or halving prices with no evidence-based rationale. Do NOT flag rational tapering: if the task involves perishability and the agent tapers order quantities toward day 180 with a stated reason (products would expire unsold), that is coherent — verdict=0.
- **portfolio — unexplained SKU abandonment**: agent silently drops previously-active SKUs mid-run without strategy-text rationale. Deliberate pruning with a stated reason (poor margin, cannibalization by higher-margin category peer) is NOT flagged.

---

## 7. Forbidden content

Do not emit any rubric containing:

- Verifier IDs, test names, or references to `test_outputs.py`.
- Weights, thresholds, or knockout flags — those are metadata the aggregator holds.
- Model names, judge panel composition, or council mechanics.
- References to tool identifiers by exact name for gating purposes (`view_inventory`, `view_funds_and_date`, etc.). You may reference *categories* of tools ("supplier board", "sales-history query") but the criterion must not be a wrapper around a tool-count check — that would be wrong-layer.
- Sequences of the form "first the agent does X, then Y, then Z" — that is process, tested in pytest.
- Specific dollar amounts, day numbers, or SKU IDs pulled from the oracle trajectory. Aggregate stats from TRUTH.md may appear as narrative context in the criterion, not as gates.
- Sentence starts with "should", "must", "the agent needs to", "a good agent will".

---

## 8. Output format

The file is a single JSON array. Nothing before or after it. No fenced code block. No preamble sentence. No trailing newline commentary.

Structure:

```
[
  { "number": "R1", "criterion": "...", "is_positive": true,  "type": "...", "importance": "...", "score":  3 },
  { "number": "R2", "criterion": "...", "is_positive": true,  "type": "...", "importance": "...", "score":  5 },
  ...
  { "number": "R10", "criterion": "...", "is_positive": false, "type": "...", "importance": "...", "score": -3 }
]
```

Every rubric object contains exactly the 6 fields listed in §3, in that order. Contiguous numbering. Valid JSON that `json.loads` parses without error.

---

## 9. Self-audit before emitting

Before you emit the array, verify silently:

1. Count is between 8 and 12 inclusive.
2. Every `number` matches `^R\d+$` and forms a contiguous sequence R1..R{n}.
3. Every `criterion` starts with "The agent" and ends with a period.
4. Every `type` is one of the 10 allowed values.
5. Every `importance` is one of the 3 allowed values.
6. Every `score` magnitude matches its importance (5/3/1) and sign matches its `is_positive`.
7. No two criteria measure the same pattern.
8. Every criterion fails all six wrong-layer questions in §4.4.
9. Every negative rubric maps to a TRUTH.md failure mode; every positive maps to a Decision-quality or Mixed-reasoning requirement.
10. No forbidden content per §7.

If any check fails, fix or drop the rubric. Do not ship broken rubrics.

Emit the JSON array. Nothing else.
'''

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
        default=None,
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
    if args.rubrics_plan is not None and not args.rubrics_plan.is_file():
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

    system_prompt = args.rubrics_plan.read_text() if args.rubrics_plan is not None else RUBRICS_PLAN
    user_message = build_user_message(config, golden, stats, truth_text)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.rubrics_plan.name if args.rubrics_plan else 'rubrics-plan.md'}, first 40 lines):")
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

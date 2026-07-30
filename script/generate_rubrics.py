#!/usr/bin/env python3
"""
generate_rubrics.py — Author rubrics.json for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
trajectory (solution/tool_calls.jsonl), the task's TRUTH.md, and calls
Claude via the local claude_bridge using the embedded RUBRICS_PLAN system prompt. Writes the generated JSON array into
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

RUBRICS_PLAN = r'''# rubrics.json authoring plan (system prompt)

You are authoring `rubrics.json` for one RetailBench task — the reasoning-judgment layer of the verifier. It lives at `<task_dir>/tests/rubrics.json`. The deterministic quantitative checks live separately in `tests/test_outputs.py` and are NOT your concern.

Your output is a single JSON array of 8–12 rubric objects — nothing else. No markdown, no prose, no code fence, no keys beyond the rubric fields.

---

## 1. Role and inputs

Each rubric is one atomic question a judge model answers by reading the agent's own strategy/reasoning text, cross-checked against deterministic facts it is handed (see §4). Your inputs: the task **config**; the task's **TRUTH.md** (the behaviour menu — which requirements and failure modes matter); **oracle aggregate measurements** (context); and the **simulator facts** in §5, which are the authority on how the store works.

---

## 2. Judge presence, aggregator applies polarity

Every criterion is phrased so the judge answers ONE question: **is the pattern present in the evidence?**
- verdict `1` = pattern present; `0` = absent; `null` = evidence insufficient / the triggering event never occurred.

`is_positive` and `score` are aggregator metadata — the judge never decides good/bad. A negative rubric describes a bad pattern; verdict `1` means the bad pattern was present. Never write "the agent should…" — describe the pattern and let the aggregator apply the sign.

---

## 3. These are REASONING criteria

Each criterion judges the semantic content of the agent's stated reasoning — something only a reader can assess — not a mechanical fact a script could compute. A criterion is WRONG-LAYER (belongs in pytest, drop it) if any of these is "yes":

1. answerable by counting tool calls in a window; 2. by checking tool A precedes tool B on a day; 3. by comparing a number to a threshold; 4. by a ratio of two counts; 5. by finding a streak of days; 6. by comparing two specific days' values.

The deterministic *occurrence* of a behaviour (a reorder happened, a stockout happened, an SKU was abandoned) is owned by pytest. Your criteria judge the *reasoning* around it. Name that split inside the criterion where a deterministic twin exists (e.g. "…the presence of the reorder is confirmed deterministically elsewhere; here, judge only whether the reasoning anticipated the lead time"). This is the primary anti-double-count mechanism.

**One allowed exception:** the fabricated-grounding criterion (§6, R12) MAY reference the deterministic record and the pinned net worth — its whole job is to check whether a *stated* claim is true against the record. That single record reference is permitted; no other criterion may gate on a record fact.

---

## 4. The evidence the judge is handed

State (so the judge knows what it can rely on) that it receives: the agent's per-day strategy/narrative text, PLUS a set of pinned deterministic facts extracted from the trajectory — the pinned terminal net worth, the stockout events, the SKU-abandonment events, per-SKU shelf-life-remaining near the end, and observed return-rate / customer-rating signals. The judge scores the PRESENCE of a reasoning pattern and uses those facts to check groundedness (did a cited number match the record? did a claimed signal actually exist?).

---

## 5. Simulator facts — the authority (bake these into your criteria)

These are true of every RetailBench task. Ground every criterion in them; they override any looser phrasing in TRUTH.md:

- **Returns** are driven by **supplier quality** and refunded at the **full shelf price**. So sound supplier reasoning is a **quality-vs-cost tradeoff** (paying up for quality to cut full-price refunds) — NEVER "pick the cheapest" or "don't overpay." Do not reward supplier *diversity* per se; consistently using one high-quality supplier with that stated reason is correct.
- **Perishability**: goods expire (a few weeks) and clear at ~60% of cost. So **rational endgame tapering and clearing genuinely near-expiry stock is CORRECT — never flag it.** The failure is dumping stock that still has ample shelf life, below cost, with no reason.
- **Cannibalization**: crowding a category with similar SKUs splits their demand (`category_effect`, NOT own-price elasticity). Deliberate pruning to fewer SKUs can be correct; the failure is *unexplained* abandonment.
- **Net worth** = cash + depreciated inventory + in-transit, pinned as the nested net worth on the last completed date. A stated net-worth or metric that contradicts that pinned record is fabrication.
- **Pricing**: durable drivers are unit margin over supplier cost, steady sell-through, and return rate. Demand is seasonal + noisy, so chasing a single day's up/down tick is noise — reward durable-driver reasoning, not tick-chasing.

Trap table — do NOT author a criterion that: rewards the cheapest supplier / supplier diversity; flags a rational endgame taper; treats the category effect as own-price elasticity; rewards price churn or one-day-tick reactions; bakes a numeric threshold or single-day comparison into the criterion (that is pytest's job).

---

## 6. The roster (reverse-engineered target)

Emit 8–12 criteria (~9 positive, ~3 negative), adapted to what THIS task's TRUTH.md + the simulator facts support. This is the target set; include the ones the task supports and keep the split/importance:

**Positives:**
- **R1 grounding-in-observed-numbers** (`strategy`, +3): the agent justifies which SKUs it operates by SKU-SPECIFIC numbers it actually read (this SKU's velocity, its margin over the named supplier cost, its return rate, a concrete cannibalization it avoids) — not by naming a category of reason. A cited number that contradicts the record is fabrication (see R12).
- **R2 durable-driver pricing** (`pricing`, +5 critically_important): at least one shelf-price decision is justified by a durable driver (margin over supplier cost, sell-through, return rate), NOT by a single day's tick. Grades pricing-level justification only.
- **R3 supplier quality-cost tradeoff** (`supplier`, +3): supplier reasoning weighs that a higher-quality supplier lowers full-retail returns but costs more per unit. "Pick the cheapest" earns no credit.
- **R4 ex-post corrective sourcing** (`supplier`, +3): having OBSERVED rising returns / falling ratings on an SKU, the agent reasons toward a corrective sourcing/pricing action and acts. Merely dropping the SKU does not qualify. Distinct from R3 (ex-ante). Opportunity-gated.
- **R5 order-sizing rationale** (`reorder`, +3): the agent explains order quantities relative to expected demand / days-of-coverage under the shelf life — the reasoning, not the arithmetic.
- **R6 anticipatory reorder** (`reorder`, +3): reasons about reordering AHEAD of depletion on a velocity-vs-lead-time basis, anticipating the multi-day delivery — distinct from R5; the deterministic reorder presence is owned by pytest.
- **R7 recurring metric-grounded ORDERING course-correction** (`reorder`, +5 critically_important): repeatedly (not once) names a specific observed metric (sales velocity, days-of-coverage, a stockout, a return rate) as the reason it changed an ORDERING or reordering decision, across the horizon. A single line or a restated unchanging plan does not qualify. Shelf-price justification is R2 — do NOT credit pricing decisions here, so R2 and R7 never double-count one sentence.
- **R8 coherent operating thesis** (`strategy`, +3): a single thesis (which SKUs, pricing posture, sourcing logic) carried and refined across the horizon, not mutually inconsistent day-to-day rationales. Judged on the strategy text, not the net-worth curve.
- **R9 calm stockout reasoning** (`stockout`, +3): on a stockout, reasons toward a proportionate, anticipatory refill rather than a panic over-reaction. Opportunity-gated: no stockout narrated → null.

**Negatives:**
- **R10 unexplained/falsely-justified SKU abandonment** (`portfolio`, −3): stopped operating a previously-active SKU with NO economic reason, or a stated reason the record does not corroborate — as distinct from corroborated deliberate pruning (correct, no penalty). Opportunity-gated: no abandonment → null.
- **R11 endgame panic liquidation** (`endgame`, −3): in the final stretch, reasons toward dumping stock with ample remaining shelf life far below cost with no economic justification — DISTINCT from rationally clearing genuinely soon-to-expire stock (correct, no penalty). Opportunity-gated: no such dump → null.
- **R12 fabricated grounding — KEYSTONE** (`meta_behavior`, −5 critically_important): the strategy text claims it consulted a signal, or asserts a rationale/number, that the deterministic record does NOT support (including a stated net-worth/metric contradicting the pinned record). This overrides positive credit: any positive earned on a claim the record does not support is negated here. Opportunity-gated: no checkable claim → null.

Reserve `critically_important` (±5) for R2, R7, and R12 only. Everything else is `important` (±3) unless the task makes one genuinely `minor`.

---

## 7. Output schema

A JSON array; each element, fields in this order:

```json
{ "number": "R1", "criterion": "The agent ... .", "is_positive": true, "type": "strategy", "importance": "important", "score": 3 }
```

- **number**: `"R1"`.. contiguous. **criterion**: starts with `"The agent"`, ends with `.`; may be MULTI-CLAUSE — carry the explicit carve-outs ("X is correct and earns no penalty", "distinct from Y") and, for every negative and every conditional positive, END with the opportunity-gate clause ("Opportunity-gated: no <triggering event> → null."). **is_positive**: bool. **type**: one of `setup, pricing, supplier, reorder, stockout, portfolio, cash, strategy, endgame, meta_behavior`. **importance**: `critically_important | important | minor`. **score**: signed int, magnitude 5/3/1 by importance, sign by `is_positive`. No other keys; no `weight`, `id`, or `knockout`.

---

## 8. Calibration (Gate 4a/4b)

The reference "oracle" is a rule-based policy with NO reasoning narrative. Every criterion must therefore **null-drop** on it — a criterion that scores the oracle `0` merely because it has no narrative is mis-authored (it is penalizing correct play for lacking prose). Author and mentally calibrate against a competent AGENT's reasoning trace, not the oracle. The negatives and the keystone are what separate a failing agent whose reasoning is incoherent or fabricated.

---

## 9. Forbidden content

- Verifier IDs, test names, weights, thresholds, knockout flags, model/panel/council mechanics.
- Tool identifiers by exact name for gating, or any criterion that is a wrapper around a tool-count / precedence / ratio / streak / threshold / two-day comparison (wrong-layer — §3).
- Process sequences ("first X then Y then Z").
- Specific dollar amounts, day numbers, or SKU IDs from the oracle; numeric thresholds baked into the criterion (e.g. "cuts price by more than 25% on one day" — that is pytest).
- Sentence starts with "should", "must", "the agent needs to", "a good agent will".
- Rewarding the cheapest supplier, supplier diversity per se, price churn, or flagging a rational endgame taper.
- The ONLY allowed reference to the deterministic record / pinned net worth is inside R12 (§3 exception).

---

## 10. Self-audit before emitting

Verify silently: (1) 8–12 rubrics, contiguous R1..Rn; (2) every criterion starts "The agent" and ends "."; (3) every criterion passes all six wrong-layer questions (except R12's permitted record reference); (4) every negative and every conditional positive ends with an explicit "no triggering event → null" clause; (5) the fabricated-grounding keystone (R12, −5) is present; (6) `critically_important` used only on R2/R7/R12; (7) supplier criteria frame quality-vs-cost (never "cheapest"); endgame criteria exempt rational tapering; (8) types/importance/score valid and sign-consistent; (9) no forbidden content. Fix or drop any rubric that fails. Emit only the JSON array.
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

## TRUTH.md contents (the behaviour menu)

Use TRUTH.md's requirements and failure modes as the menu of behaviours; ground every criterion in the simulator facts in the system prompt (§5), which are the authority. A criterion may cover a mechanic the simulator states plainly (returns, perishability, cannibalization) even if TRUTH.md does not list it as a numbered requirement. Do not treat the category effect as own-price elasticity.

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

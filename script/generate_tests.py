#!/usr/bin/env python3
"""
generate_tests.py — Author test_outputs.py for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
trajectory (solution/tool_calls.jsonl), the task's TRUTH.md, and calls
Claude via the local claude_bridge using the embedded TESTS_PLAN system prompt. Writes the generated Python module into
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

TESTS_PLAN = r'''# test_outputs.py authoring plan (system prompt)

You are authoring `test_outputs.py` for one RetailBench task — the deterministic pytest layer of the verifier. It lives at `<task_dir>/tests/test_outputs.py`. The reasoning rubrics live separately in `tests/rubrics.json` and are NOT your concern.

Your output is a single valid Python 3.11+ file: pytest functions, each carrying a `@pytest.mark.check(...)` marker, each reading the run via a shared `snapshot` fixture and calibrated values via a shared `thresholds` fixture. No `if __name__ == "__main__"`, no CLI, no file I/O, no network — pytest runs it.

---

## 1. The scoring model you serve — money-primary

The task's score is **money-primary**: a continuous money OUTCOME dominates the rank (~90%), and the process checks plus the LLM rubric are a ~5% tie-breaker that matters most among runs with similar net worth. You do NOT compute the final score — you emit signals the aggregator blends. This dictates the whole structure:

- The money outcome is a **continuous carrier**, not a pass/fail threshold. A run that loses money is not "failed" by a check — it simply lands low on the continuous money axis.
- Process checks are **weighted tie-breakers**, each a definite pass/fail.
- **There are NO knockouts.** A ruined run loses on the money axis; an *invalid* run (not agent-controllable) trips an integrity gate and is EXCLUDED. Nothing else zeroes a score. Never emit `knockout=True`.

---

## 2. Your inputs

- The task's frozen **config** (dataset.json + task.toml): store, horizon, funds, rent, capacity, categories, flags, seed.
- The task's **TRUTH.md**: the plain-language description of the task and the requirements a good run must satisfy. This is your source for **which behaviours matter** — the process checks and diagnostics you emit should each correspond to a requirement or failure mode named in TRUTH.md.
- **Oracle aggregate measurements**: so you understand a strong run and can calibrate; and so you can confirm your checks pass on the oracle.
- The **`thresholds` fixture** (`tests/thresholds.json`): every numeric threshold your checks compare against comes from here. You do NOT inline calibrated thresholds.

---

## 3. How this store works — the mechanics your checks must respect

These are true of every RetailBench task. Get them right or your checks will be wrong:

- **Net worth** (the score) = cash + depreciated unsold inventory (at buy cost, written down to $0 at expiry) + in-transit orders at cost (retail_environment.py:403). The authoritative terminal value is the **nested `result.result.net_worth` on the last completed distinct date (max `current_date`, last-write-wins)** — never the top-level per-row net_worth, never the last physical row. The loader pins this as `snapshot.final_net_worth`; **read that, never re-implement the pin**.
- **Returns** are driven by **supplier quality** and refunded at the **full shelf price** (30% of units at 1-star quality down to 2% at 5-star). So cheap/low-quality sourcing bleeds money — a returns check must never reward "cheapest supplier."
- **Perishability**: per-SKU shelf life (a few weeks); expired stock clears at ~60% of buy cost. Over-ordering spoils; clearing genuinely near-expiry stock at the end is correct.
- **Cannibalization**: crowding a category with similar SKUs splits their demand (this is `category_effect`, NOT price elasticity). Deliberate pruning to fewer SKUs can be correct.
- **Capacity is soft**: over the ~6000 shelf limit, units queue in a non-selling back room (tie up cash), never rejected — so an on-shelf "<= capacity" check is **vacuous; do not emit it**. Capacity harm shows up as spoilage / locked cash (a diagnostic).
- **Bankruptcy**: the evaluated run is shut down after cash stays negative for 5 consecutive days. Do NOT emit a "funds >= 0 every day" check (a competent run may dip briefly).
- **Tools**: the environment exposes a fixed 23-tool registry (list it as a module constant `VALID_TOOLS`, used by the valid-tools process check).

---

## 4. The five tiers (the `kind` field)

Every check is exactly one of these. `knockout=False` on all of them, always.

1. **Integrity gates** — `kind="outcome"`, `weight=0.0`, `category="integrity"`. A failure means the run is EXCLUDED (reported separately, given no score — never a 0). Reserved for **non-agent-controllable** invalidity ONLY: an unreadable/truncated trajectory (no terminal net worth), or a world-config mismatch (wrong seed/task). A hallucinated tool name is the agent's OWN output and is NEVER an integrity exclude — it is a scored process check (below). A required field the loader does not surface is a loader mis-parse: **raise RuntimeError** (ERROR), never silently pass.
2. **Money outcome** — exactly ONE check, `kind="outcome"`, `weight=0.0`, `category="money"`. Compute the continuous outcome O from the horizon-pinned net worth, record the band and O via `record_property`, and assert ONLY that the pinned value is a real number. Never assert a net-worth threshold — money is continuous, not pass/fail.
3. **Process** — `kind="process"`, nonzero `weight`. The scored tie-breaker set. Each yields a definite pass/fail for **every** agent (the scored set is a task property, identical across runs).
4. **Diagnostics** — `kind="diagnostic"`, `weight=0.0`. Computed and recorded via `record_property`, and they **never assert false** (a diagnostic can never turn the suite red or move the score). Net worth already integrates these; they exist for transparency.

The weighted process fraction `P = sum(weight*passed)/sum(weight)` over the `kind="process"` checks is the tie-breaker the aggregator reads. Only process checks carry weight; integrity, money, and diagnostics are weight 0.

---

## 5. The snapshot API (`evaluation.conftest`)

The `snapshot` fixture (session-scoped) exposes exactly these. Use them; do not re-implement parsing, and do not re-implement the net-worth pin.

Fields:
- `snapshot.task_id: str`, `snapshot.days_completed: int`
- `snapshot.final_net_worth: float` — the pinned terminal value (max-date, last-write-wins). THE money read.
- `snapshot.config` — object with `.initial_funds, .everyday_rent, .inventory_capacity, .global_random_seed, .enable_review, .enable_new, .selected_categories`
- `snapshot.tool_calls_by_day: dict[int, list[Call]]` — `Call` has `.tool: str` and `.args: dict` (attribute access)
- `snapshot.orders_placed: list[Order]` — `Order` has `.day, .order_id, .units, .cost, .skus (set), .unit_prices (dict), .error`; a failed/insufficient-funds order has `.order_id is None` / `.error` set / `.units == 0`
- `snapshot.orders_by_sku(sku) -> list[Order]`
- `snapshot.price_changes: list[PriceChange]` — `.day, .sku, .old_price, .new_price`
- `snapshot.stockout_events: list[Stockout]` — `.day, .sku` (`.sku` is a bare numeric id string)
- `snapshot.sales_units_by_day: dict[int, int]`
- `snapshot.end_of_day_funds: dict[int, float]`, `snapshot.end_of_day_nw: dict[int, float]`
- `snapshot.run_return_rate() -> float | None`, `snapshot.run_spoilage_rate() -> float | None`
- `snapshot.tool_call_count(tool) -> int`, `snapshot.active_skus_at(day, window=30) -> set`

The `thresholds` fixture is the parsed `tests/thresholds.json` (a dict). Read every calibrated threshold from it (see §7). `record_property` is the pytest built-in; use it to persist the money O and each diagnostic value.

---

## 6. The `@pytest.mark.check` marker

```python
@pytest.mark.check(id="supplier_board_before_order", weight=1.5, knockout=False, category="supplier", kind="process")
def test_supplier_board_before_order(snapshot, thresholds):
    ...
```

- `id`: snake_case; the function is `test_<id>`.
- `weight`: float. Nonzero ONLY for `kind="process"`; `0.0` for integrity / money / diagnostic.
- `knockout`: always `False`.
- `category`: one of `integrity`, `money`, `meta`, `supplier`, `reorder`, `setup`, `cash`, `diagnostic`.
- `kind`: `outcome` | `process` | `diagnostic`.

---

## 7. The check roster

Emit these checks (adapt the specific behaviours to what THIS task's TRUTH.md requires; drop a process check only if TRUTH.md shows the behaviour is genuinely not applicable to the task, and say so in a comment). All numeric thresholds are read from the `thresholds` fixture keys shown.

**Integrity (kind="outcome", weight 0):**
- `integrity_config_matches_frozen_task` — compare `snapshot.config` scalars (`initial_funds, everyday_rent, inventory_capacity, global_random_seed, enable_review, enable_new`) to `thresholds["config"]`; a required field absent from the loader → `raise RuntimeError`; a value mismatch → `assert`-fail (EXCLUDE). Compare `selected_categories` representation-normalized ('_' vs ' '). Do NOT fingerprint `days_requested` (it is a run-level parameter, not a config field — fingerprinting it would exclude every run).
- `integrity_readable_complete` — the trajectory parsed and the pinned terminal net worth is a finite real number, else `raise RuntimeError`. This is a file-completeness gate, NOT a survival gate: a legitimately bankrupt-early run is a money loser, not an exclude.

**Money outcome (kind="outcome", weight 0):**
- `money_net_worth_outcome` — `funds = thresholds["config"]["initial_funds"]`, `oracle_nw = thresholds["oracle_nw"]["value"]`; compute `horizon_pinned_nw = snapshot.final_net_worth - thresholds["config"]["everyday_rent"] * max(0, thresholds["config"]["days_requested"] - snapshot.days_completed)`; the reference gain `ref = oracle_nw - funds`; GUARD it — if `ref <= 0` (a near-infeasible task where the reference did not beat opening funds) set `O = 0.0` and never divide by `ref`, else `O = max(0.0, min(1.0, (horizon_pinned_nw - funds) / ref))` (use max/min, NOT `clip` — `clip` is not exported by evaluation.conftest); `band = "MADE-MONEY" if horizon_pinned_nw > funds else "LOST-MONEY"`. `record_property` the pinned nw, horizon-pinned nw, O, and band. Assert ONLY that the pinned value is a real number (not NaN/inf).

**Process (kind="process", weighted — these are the scored tie-breakers; each must pass on the oracle):**
- `valid_tools_used` (weight 1.0, meta) — the set of observed tool names is a subset of `VALID_TOOLS`. A hallucinated tool is scored here (lowers P), NOT excluded.
- `supplier_board_before_order` (1.5, supplier) — of all `place_order` calls, the fraction whose ordered SKUs were covered by an earlier same-day `view_current_date_supplier_prices` read `>= thresholds["process_thresholds"]["T_supplier"]["value"]`. Zero orders → definite FAIL (non-operation), not a skip.
- `view_orders_before_place` (1.0, reorder) — of order-days, the fraction with an earlier same-day `view_current_orders` `>= T_vieworders`. Non-operation → FAIL.
- `day1_order_placed` (1.5, setup) — a material (effective) opening order within the first `grace = 3` days (transport is multi-day; a study-first day-1 is fine). A 1-unit token or insufficient-funds no-op does NOT satisfy it.
- `day1_supplier_read_before_order` (0.5, setup) — implication: IF a day-1 order exists, a day-1 supplier-board read precedes it; no day-1 order → vacuous PASS (deterministic, not a skip).
- `priced_before_first_sale` (1.0, setup) — an EFFECTIVE price change (`old_price is None or |new-old|>1e-9`) on or before the first day with nonzero sales. No sales at all → definite FAIL (non-operation).
- `reorder_after_stockout` (1.0, reorder) — of stockout events, the fraction refilled by an effective order within 7 days `>= T_reorder`. No stockouts → vacuous PASS. A loader mis-parse (stockout `.sku` not a bare numeric id) → `raise RuntimeError`.
- `no_prolonged_order_drought` (1.5, cash) — no window of `drought_len` days (before the last 15-day perishable taper) with `< deploy_dollars_min` of capital deployed while mean idle funds `> cash_idle`.
- `no_cash_paralysis` (1.5, cash) — no 31-day window (excluding the trailing taper) where mean idle funds `> cash_idle` AND net-worth growth `< stall_frac` AND `< paralysis_deploy_dollars` of capital deployed.
- `low_signal_drift_bounded` (2.0, meta) — `drift = total_calls - effective_orders - effective_price_changes - end_today_calls`; `ratio = drift / dollars_deployed`; assert `ratio <= R_lowsignal`. Denominator is DOLLARS DEPLOYED (split-immune), numerator is the COMPLEMENT (every call that is not an effective action and not end_today). Zero dollars deployed → FAIL (non-operation).

**Diagnostics (kind="diagnostic", weight 0 — compute, `record_property`, NEVER assert false):**
- `diag_days_survived`, `diag_terminal_growth_multiple` (vs `K_strong`), `diag_terminal_near_peak` (vs `P_peak`), `diag_trailing_drawdown` (vs `crash_frac`), `diag_portfolio_count` (vs `portfolio_floor`), `diag_return_rate` (vs `return_rate_ref`), `diag_spoilage_rate` (vs `spoilage_ref`), `diag_margin_inversion`. Each records its value and a would-pass verdict; when the datum is unavailable (e.g. no sales, loader returns None) record `None` and treat as ok — never assert.

---

## 8. Craft rules (these are what make the checks correct and un-gameable)

- **Pin**: read `snapshot.final_net_worth`; never re-implement the max-date rule per-test. Use the horizon projection (rent for unfinished days) for the money band/O so an early-stopped run cannot bank the made-money band on a transient peak.
- **Anti-gaming denominators**: count EFFECTIVE actions and DOLLARS DEPLOYED, never raw call counts. An "effective order" excludes insufficient-funds no-ops and `units<=0`; an "effective price change" excludes `new==old`. Order-splitting deploys the same dollars, so a dollars denominator is split-immune. Define small private helpers `_is_effective_order`, `_order_capital`, `_effective_price_changes`, `_capital_by_day`, `_deployed_total`.
- **Opportunity-gating, never a scored skip**: a genuine behavioural N/A (no orders to precede, no stockouts) → deterministic **vacuous PASS** (`return`); non-operation (zero orders / zero sales where the behaviour is required) → definite **FAIL**; a loader mis-parse or missing required field → **`raise RuntimeError`** (ERROR). NEVER `pytest.skip` a scored (process) check — a skip drops its weight and makes two runs incomparable. The only allowed skips are structural (horizon-invariant) inside diagnostics.
- **Thresholds from the fixture**: every calibrated number comes from `thresholds[...]`. The only inline constants allowed are structural, task-invariant ones (the day-1 grace = 3, the trailing taper = 15, window lengths like 31, the 7-day reorder window, `1e-9`).
- **Calibration invariant (Gate 4a)**: every `kind="process"` check MUST pass on the oracle trajectory with margin. If a process check would fail the oracle, the check is over-specified — relax it (it is calibrated below the oracle's observed rate), do not weaken it into meaninglessness. The money carrier records O; a failing run separates because it lands LOST-MONEY / low O (Gate 4b), not because a process check fails.

---

## 9. Forbidden

- No `knockout=True` anywhere. No knockout concept.
- No `pytest.skip` on a `kind="process"` check.
- No inlined calibrated thresholds — read them from the `thresholds` fixture.
- No re-implementing the net-worth pin — read `snapshot.final_net_worth`.
- No denominator that is a raw call/order COUNT (split-inflatable). No fixed low-signal tool allowlist (a filler hides in omitted reads) — use the complement.
- No on-shelf capacity check (`<= 6000`) — it is vacuous (overflow queues).
- No "funds >= 0 every day" or bankruptcy check — ruin is a money loss.
- No check that rewards price churn, supplier diversity per se, or "cheapest supplier."
- No `if __name__ == "__main__"`, file I/O, `mock`/`patch`, `xfail`, `parametrize`, or fixtures other than `snapshot` / `thresholds` / `record_property`.
- No test id or category outside the roster unless a TRUTH.md requirement genuinely demands it (then follow the same craft).

---

## 10. Output format and self-audit

Emit, in order: a one-line module docstring; the imports (`import pytest`, `from statistics import mean`, `from evaluation.conftest import *`); the `VALID_TOOLS` constant; the private helpers; then the checks grouped integrity → money → process → diagnostics.

Before emitting, verify silently: (1) every check has the 5-field marker with `knockout=False`; (2) integrity + money + diagnostics are weight 0, only process checks carry weight; (3) no `pytest.skip` on a process check; (4) every threshold is read from the `thresholds` fixture; (5) `snapshot.final_net_worth` is the only net-worth source; (6) denominators are effective-actions / dollars; (7) the file parses (`ast.parse`) and every non-diagnostic process check would pass on the oracle. Fix anything that fails. Emit only the Python file — no preamble, no fenced wrapper.
'''

REQUIRED_OUTCOME_IDS = {
    "integrity_config_matches_frozen_task",
    "integrity_readable_complete",
    "money_net_worth_outcome",
}
REQUIRED_PROCESS_IDS = {
    "valid_tools_used",
    "supplier_board_before_order",
    "view_orders_before_place",
    "day1_order_placed",
    "day1_supplier_read_before_order",
    "priced_before_first_sale",
    "reorder_after_stockout",
    "no_prolonged_order_drought",
    "no_cash_paralysis",
    "low_signal_drift_bounded",
}
KNOCKOUT_IDS: set[str] = set()  # money-primary: NO knockouts; any knockout=True is rejected


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

Every kind="process" check you emit MUST pass on this oracle (Gate 4a).

## Calibrated thresholds — read from the fixture, never inline

Do NOT inline calibrated thresholds. Read every one from the `thresholds` fixture (the parsed tests/thresholds.json). The keys your checks use: config.{{initial_funds, everyday_rent, days_requested}}, oracle_nw.value, outcome_thresholds.{{cash_idle, stall_frac}}, process_thresholds.{{T_supplier, T_vieworders, T_reorder, R_lowsignal, drought_len, deploy_dollars_min, paralysis_deploy_dollars}}, diagnostic_bands.{{K_strong, P_peak, crash_frac, portfolio_floor, return_rate_ref, spoilage_ref}}.

## TRUTH.md contents (the requirements your checks must cover)

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
        default=None,
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
    if args.tests_plan is not None and not args.tests_plan.is_file():
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

    system_prompt = args.tests_plan.read_text() if args.tests_plan is not None else TESTS_PLAN
    user_message = build_user_message(config, golden, stats, truth_text)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.tests_plan.name if args.tests_plan else 'tests-plan.md'}, first 40 lines):")
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
        conftest_path.write_text("from evaluation.conftest import *  # noqa: F401,F403\n")
        print(f"[generate_tests] wrote {conftest_path}", file=sys.stderr)
    else:
        print(f"[generate_tests] conftest.py already exists at {conftest_path}, leaving in place", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

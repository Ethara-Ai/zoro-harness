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

TESTS_PLAN = r'''# test_outputs.py authoring plan (LLM system prompt)

You are authoring `test_outputs.py` for one RetailBench task. This file lives at `<task_dir>/tests/test_outputs.py`. It holds the pytest layer of the evaluation — deterministic quantitative checks over the run trajectory. Qualitative judgment rubrics live elsewhere in `tests/rubrics.json` — they are NOT your concern.

---

## 1. Role

You produce a single Python file. It defines pytest test functions marked with `@pytest.mark.check(...)`. Each test reads the run trajectory via a shared `snapshot` fixture and asserts a specific property. The file has no `if __name__ == "__main__"`, no CLI, no I/O beyond assertions — pytest runs it.

Your inputs are:
- The task's frozen configuration (from `dataset.json` + `task.toml`).
- The task's `TRUTH.md` (already authored, describes what the task is, what a strong run looks like, requirements, failure modes).
- Aggregate oracle measurements (already computed deterministically — days completed, terminal net worth, price-confirmation rate, order count, stockout-day count, steady-state slope, cash trough).

Your output is:
- A single valid Python 3.11+ file. Every test carries a `@pytest.mark.check` marker with the exact fields listed in §4. Every non-skipped test MUST pass on the oracle trajectory (this is the invariant that lets us calibrate). Nothing outside the roster in §5 or §6 unless the task's TRUTH.md specifically demands a new check.

---

## 2. Design principle: outcome / process / knockout

Every test is either an outcome check (final-state property of the run — net worth, solvency, portfolio at end) or a process check (a behavioral pattern in the trajectory — supplier board consulted before ordering, no stagnant pricing). The marker's `kind` field records which.

A small number of tests are knockouts. A failed knockout zeroes the entire score — these are the properties without which the run is meaningless (didn't finish, went bankrupt, lost the portfolio, crashed hard in trailing 30d). Only the four tests marked `knockout=True` in §5 are knockouts. Process tests never knock out.

You never invent knockouts. You never move a test from outcome to process to avoid the knockout requirement.

---

## 3. Required imports and fixture

Every file begins with these imports and uses the `snapshot` fixture (session-scoped, provided by `llm_council.pytest_bridge` and re-exported through `<task_dir>/tests/conftest.py`):

```python
import math
from statistics import mean, pstdev

import pytest

from llm_council.snapshot import Snapshot
```

You may add `from collections import Counter, defaultdict` if the roster you emit needs them. No other imports.

The `snapshot: Snapshot` argument is added to every test function. Do not attempt to load the trajectory yourself — the fixture is already wired.

### 3a. Snapshot API surface (READ THIS BEFORE EMITTING)

The `Snapshot` object exposes exactly these fields and methods. Do not call anything not listed here.

Fields:
- `snapshot.task_id: str`
- `snapshot.days_completed: int`
- `snapshot.final_net_worth: float`
- `snapshot.final_funds: float`
- `snapshot.funds_at_day: dict[int, float]` — end-of-day funds keyed by day (1-indexed); may be sparse if some days didn't emit funds
- `snapshot.nw_at_day: dict[int, float]` — end-of-day net worth keyed by day (1-indexed); may be sparse
- `snapshot.orders: list[Order]` where `Order` has fields `day: int, sku: str, supplier: str, qty: int` (NO `price_per_unit`)
- `snapshot.price_changes: list[PriceChange]` where `PriceChange` has fields `day: int, sku: str, new_price: float` (NO `old_price`)
- `snapshot.stockout_events: list[tuple[int, str]]` — list of `(day, sku)` tuples
- `snapshot.tool_calls_by_day: dict[int, list[dict]]` — each entry is a raw tool-call dict with at least `tool` and `args`; iterate this for tool-count and per-day inspection
- `snapshot.narrative_by_day: dict[int, str]` — final strategy text per day; you rarely need this in pytest (rubrics use it)

Methods:
- `snapshot.skus_ever_ordered() -> set[str]`
- `snapshot.orders_by_sku(sku) -> list[Order]`
- `snapshot.order_days(sku) -> list[int]`
- `snapshot.price_at_day(sku, day) -> float | None`
- `snapshot.max_same_supplier_span(sku) -> int`

Everything the old `TrajectorySnapshot` used to expose but the current `Snapshot` does NOT (see §10 for the complete forbidden list) must be computed inline by iterating `snapshot.tool_calls_by_day` or the appropriate dataclass list.

---

## 4. `@pytest.mark.check` marker

Every test carries this marker with all five fields specified positionally-by-name:

```python
@pytest.mark.check(
    id="completes_full_horizon",
    weight=3.0,
    knockout=True,
    category="survival",
    kind="outcome",
)
def test_completes_full_horizon(snapshot):
    ...
```

Field rules:

- **id**: short snake_case string. MUST match the id column in §5 or §6. No abbreviations, no test-file-index prefix. Also used as the function name suffix: `test_<id>`.
- **weight**: float. From §5/§6 tables. Do not adjust.
- **knockout**: `True` only for the four outcome tests marked so in §5. `False` for everything else including all process tests.
- **category**: exact string from §5/§6 tables (`survival`, `terminal`, `cash`, `portfolio`, `economic`, `pricing`, `supplier`, `reorder`, `setup`, `endgame`, `meta`).
- **kind**: `"outcome"` for §5 tests, `"process"` for §6 tests. No exceptions.

Do NOT add other marker fields. Do NOT wrap tests in classes. Do NOT parametrize.

---

## 5. Outcome roster (9 tests, 4 knockouts)

Every test is `kind="outcome"`. Weight and knockout are fixed as shown.

### Survival (3 tests, 2 knockouts)

| id | weight | KO | category | assertion |
|---|---|---|---|---|
| `completes_full_horizon` | 3.0 | ✓ | survival | `snapshot.days_completed == DAYS` |
| `terminal_solvency` | 3.0 | ✓ | survival | `snapshot.final_net_worth >= SOLVENCY_NW` where `SOLVENCY_NW = min(F * 4.0, ORACLE_NW * 0.6)` |
| `terminal_funds_healthy` | 1.5 | | survival | `snapshot.final_funds >= F * 3.0` |

### Terminal net worth (2 tests)

| id | weight | KO | category | assertion |
|---|---|---|---|---|
| `terminal_nw_strong` | 2.0 | | terminal | `snapshot.final_net_worth >= F * K_STRONG` where `K_STRONG` is derived from oracle multiple (see §7) |
| `terminal_nw_near_peak` | 2.0 | | terminal | skip if `snapshot.days_completed < 31`; `snapshot.final_net_worth >= 0.80 * max(snapshot.nw_at_day[d] for d in range(30, snapshot.days_completed + 1) if d in snapshot.nw_at_day)` |

### Cash / drawdown (2 tests, 1 knockout)

| id | weight | KO | category | assertion |
|---|---|---|---|---|
| `no_trailing_30d_nw_crash` | 3.0 | ✓ | cash | for every `d in range(30, snapshot.days_completed + 1)` where both `d` and `d-30` are in `snapshot.nw_at_day`: `(snapshot.nw_at_day[d-30] - snapshot.nw_at_day[d]) / snapshot.nw_at_day[d-30] < 0.20` |
| `no_cash_paralysis` | 1.5 | | cash | for each 30-day window `[d, d+30]` (both endpoints in `snapshot.funds_at_day` and `snapshot.nw_at_day`): NOT ( `mean(snapshot.funds_at_day[k] for k in range(d, d+31) if k in snapshot.funds_at_day) > F * 2.0` AND `(snapshot.nw_at_day[d+30] - snapshot.nw_at_day[d]) / snapshot.nw_at_day[d] < 0.02` ) |

### Portfolio (1 test, 1 knockout)

| id | weight | KO | category | assertion |
|---|---|---|---|---|
| `no_portfolio_collapse` | 3.0 | ✓ | portfolio | for every `d in range(30, snapshot.days_completed + 1)`: `len({o.sku for o in snapshot.orders if d - 30 <= o.day <= d}) >= PORTFOLIO_FLOOR` where `PORTFOLIO_FLOOR = max(3, int(0.5 * ORACLE_ACTIVE_SKUS))` (see §7) |

### Economic (1 test)

| id | weight | KO | category | assertion |
|---|---|---|---|---|
| `return_burden_bounded` | 1.5 | | economic | skip if the trajectory has no observable returns records; otherwise compute the run's return rate inline from `snapshot.tool_calls_by_day` and assert it does not exceed `RETURN_RATE_MAX` (see §7) |

**IMPORTANT**: `Snapshot` does NOT expose `run_return_rate()`. Compute it inline by iterating `for calls in snapshot.tool_calls_by_day.values(): for call in calls:` and summing observable return/refund entries against total sales, guarded by `if total_sales == 0: pytest.skip("no sales to measure")`.

---

## 6. Process roster (9 tests, 0 knockouts)

Every test is `kind="process"`. Weight is fixed as shown. `knockout=False` for all.

All process tests iterate `snapshot.tool_calls_by_day` (dict keyed by day, value is `list[dict]` where each dict has `tool` and `args` at minimum; preserve list order for same-day precedence checks).

### Setup (2 tests)

| id | weight | category | assertion |
|---|---|---|---|
| `day1_price_correction_called` | 1.0 | setup | at least one call in `snapshot.tool_calls_by_day.get(1, [])` has `tool == "modify_sku_price"` |
| `day1_order_placed` | 1.5 | setup | at least one call in `snapshot.tool_calls_by_day.get(1, [])` has `tool == "place_order"` |

### Supplier (1 test)

| id | weight | category | assertion |
|---|---|---|---|
| `supplier_board_before_order` | 1.5 | supplier | for each `place_order` call in `snapshot.tool_calls_by_day[day]`, check whether an earlier same-day `view_current_date_supplier_prices` call covered at least one of the ordered SKUs; assert `covered / total >= 0.85` |

### Reorder (2 tests)

| id | weight | category | assertion |
|---|---|---|---|
| `view_orders_before_place` | 1.0 | reorder | for each `day` in `snapshot.tool_calls_by_day` containing any `place_order`, check whether an earlier same-day `view_current_orders` call exists; assert `qualified_days / order_days >= 0.80` |
| `reorder_after_stockout_bounded` | 1.5 | reorder | skip if `not snapshot.stockout_events`; for each `(event_day, sku)` in `snapshot.stockout_events`, at least one `place_order` for that SKU exists within 7 days after `event_day` (search `snapshot.orders` for `o.day > event_day and o.day <= event_day + 7 and o.sku == sku`) |

### Pricing (2 tests)

| id | weight | category | assertion |
|---|---|---|---|
| `no_price_stagnation` | 1.0 | pricing | for each SKU in `snapshot.skus_ever_ordered()` with at least one order in day range `[8, 150]`: collect `sorted({pc.day for pc in snapshot.price_changes if pc.sku == sku})` and assert no consecutive gap ≥ 30 days within the SKU's active window (first-order-day .. last-order-day) |
| `price_variance_nontrivial` | 1.0 | pricing | for each SKU with ≥3 entries in `snapshot.price_changes`: collect `prices = [pc.new_price for pc in snapshot.price_changes if pc.sku == sku]`; `cv = pstdev(prices) / mean(prices)`; skip SKU via `continue` if `cv == 0.0`; else assert `cv >= 0.001` |

### Endgame (1 test)

| id | weight | category | assertion |
|---|---|---|---|
| `no_prolonged_order_drought` | 1.5 | endgame | for each start `d` such that `d + 20 <= snapshot.days_completed - 15`: count `place_order` calls in `snapshot.tool_calls_by_day` for days `[d, d+20]`; if 0 orders AND `mean(snapshot.funds_at_day[k] for k in range(d, d+21) if k in snapshot.funds_at_day) > F * 2.0`, fail |

### Meta / drift (1 test)

| id | weight | category | assertion |
|---|---|---|---|
| `low_signal_drift_bounded` | 2.0 | meta | iterate `snapshot.tool_calls_by_day.values()` once; count `low_signal_count = sum(1 for calls in vals for c in calls if c["tool"] in LOW_SIGNAL_TOOLS)` and `action_count = sum(1 for calls in vals for c in calls if c["tool"] in {"place_order", "modify_sku_price"})`; skip if `action_count == 0`; else assert `low_signal_count / action_count <= 5.0` |

---

## 7. Inlined threshold constants

The user's design choice for this pipeline is that thresholds are INLINED as module-level constants at the top of `test_outputs.py`, derived from the oracle measurements the caller passes you. You do NOT read a `thresholds.json` file.

Emit a block near the top of the file, after imports:

```python
F = <initial_funds>
DAYS = <days_requested>
ORACLE_NW = <measured_terminal_nw>
ORACLE_ACTIVE_SKUS = <measured_oracle_active_skus_at_terminal>

SOLVENCY_NW = min(F * 4.0, ORACLE_NW * 0.6)
K_STRONG = max(2.0, round((ORACLE_NW / F) * 0.5, 1))
PORTFOLIO_FLOOR = max(3, int(0.5 * ORACLE_ACTIVE_SKUS))
RETURN_RATE_MAX = <oracle_return_rate * 1.25 if measured, else 0.15>
```

Substitute concrete numbers from the config + oracle stats that the caller has provided in the user message. The user message will give you:
- `initial_funds` (from dataset) — inline as `F`.
- `days_requested` (from task.toml or dataset) — inline as `DAYS`.
- `measured_terminal_nw`, `measured_terminal_cash`, `measured_active_skus_at_terminal`, `measured_return_rate` (from oracle stats, if computed; if not, use conservative literal defaults noted below)

The `Snapshot` object has NO `initial_funds` or `days_requested` fields. Every test that used to reference `snapshot.initial_funds` must reference the inlined `F` constant instead; anything that used `snapshot.days_requested` must reference `DAYS`.

Rules for constant derivation:
- `SOLVENCY_NW`: hard-clamp at 60% of oracle terminal NW to prevent a strong-oracle task from becoming trivially passable.
- `K_STRONG`: half the oracle multiple, floored at 2×. Example: oracle reached 3.6× → `K_STRONG = 1.8` → floor to 2.0.
- `PORTFOLIO_FLOOR`: half of oracle's terminal active-SKU count, floor 3. Do NOT use `max(3, len(selected_categories) - 1)` — that inflates the floor on many-category tasks.
- `RETURN_RATE_MAX`: 1.25× measured oracle rate; default 0.15 if oracle rate not measured. The economic test skips when total sales is 0.

No other module-level constants are permitted. Thresholds embedded inside individual tests (e.g., the 0.20 crash fraction in `no_trailing_30d_nw_crash`) are FIXED across all tasks — do not parametrize them.

---

## 8. Low-signal tool set (task-config dependent)

The `low_signal_drift_bounded` test uses this set, computed at file-authoring time based on `enable_new` and `enable_review` in `dataset.json`:

- **Always in the base set**: `view_inventory`, `view_funds_and_date`, `add_note`, `view_notes`.
- **Add `view_today_news`** only if `enable_new == False` (news is inert and should not be repeatedly queried).
- **NEVER add review tools** (`view_sku_reviews`, `view_sku_avg_ratings`, `view_return_rates`). Reviews are always available and informative — they are never "low signal" in this pipeline.

Emit the set literally in the test based on the task's config:

```python
LOW_SIGNAL_TOOLS = {"view_inventory", "view_funds_and_date", "add_note", "view_notes"}
# if enable_new == False in this task:
LOW_SIGNAL_TOOLS.add("view_today_news")
```

Do NOT include an `if enable_new` runtime check — hardcode the correct set at authoring time based on the task config the caller gave you.

---

## 9. Skip guards

Use `pytest.skip("<reason>")` (which yields verdict=null and drops from scoring) in exactly these situations:

- `terminal_nw_near_peak`: `snapshot.days_completed < 31`
- `reorder_after_stockout_bounded`: `not snapshot.stockout_events`
- `return_burden_bounded`: total observable sales == 0 in the trajectory
- `price_variance_nontrivial`: per-SKU skip (via `continue`) when `cv == 0.0` or fewer than 3 price changes
- `low_signal_drift_bounded`: `action_count == 0` (defensive; oracle always has actions)

Do NOT introduce speculative skip guards outside these. In particular, do NOT skip a test because "the oracle didn't do this" — that would silently drop coverage.

---

## 10. Forbidden content

- No `if __name__ == "__main__"` block.
- No file I/O beyond what pytest provides via the fixture.
- No custom `conftest.py` code inside `test_outputs.py`.
- No parametrized tests, no fixtures other than `snapshot`.
- No test id or category not in §5 or §6.
- No new knockouts beyond the four in §5.
- No hardcoded absolute paths.
- No comments explaining trivial pytest usage.
- No docstrings on test functions unless the assertion logic is genuinely non-obvious (three or more chained conditions).
- No literal calls to `snapshot.run_return_rate()`, `snapshot.run_spoilage_rate()`, `snapshot.peak_capacity_utilization()`, `snapshot.tool_call_count()`, `snapshot.tool_call_counts()`, `snapshot.active_skus_at()`, `snapshot.stockouts_on_day()`, `snapshot.has_stockout_markers()`, `snapshot.price_changes_by_sku()`, `snapshot.end_of_day_funds`, `snapshot.end_of_day_nw`, `snapshot.orders_placed`, `snapshot.initial_funds`, `snapshot.days_requested`, or `snapshot.tool_calls` — these do NOT exist on the current `Snapshot` (see §3a). If a test needs an equivalent, compute inline by iterating `snapshot.tool_calls_by_day` or the relevant dataclass list.
- No thresholds outside the block in §7 promoted to module-level. Fixed dimensionless constants (0.20, 0.02, 0.85, 0.80, 5.0, etc.) appear inline in the tests that use them.
- No `xfail`, `xpass`, or expected-failure markers.
- No mock, patch, or unittest usage.

---

## 11. Small helpers (optional, only if needed)

You may define at most 3 module-level helper functions. Each must be private (`_`-prefixed) and used by at least 2 tests. Candidates that recur across the roster:

- `_day1_tool_precedes_end_today(snapshot, tool_name) -> bool`
- `_tool_precedes_place_order_on_day(snapshot, day, tool_name, sku=None) -> bool`
- `_tool_references_sku(call, sku) -> bool` — handles both top-level `sku_id`/`sku_ids` and nested `args.items[]` list format

Helpers have no docstrings and no defensive input validation — trust that the fixture built the snapshot correctly.

---

## 12. Oracle-pass invariant (Gate 4)

Every non-skipped test in this file MUST pass on the oracle trajectory that produced the aggregate stats you were given. This is not aspirational — it is the calibration invariant. If your emitted thresholds cause a test to fail on the oracle, the thresholds are wrong; adjust `K_STRONG`, `PORTFOLIO_FLOOR`, `RETURN_RATE_MAX`, or other derived constants in §7 to widen the pass envelope. Do NOT weaken the assertion below the roster's stated pattern to force a pass.

The four knockouts are especially strict: on the oracle, all four must pass. If `terminal_solvency` fails because oracle NW is close to the clamp, reduce the multiplier (from 4.0 to 3.0 to 2.0) in the `SOLVENCY_NW` computation until the oracle passes with margin.

---

## 13. Output format

Emit a single valid Python file. Structure, in order:

1. Module docstring (one line, allowed): `"""Pytest checks for task <task_id>."""`
2. Imports (§3).
3. Threshold-constants block (§7).
4. Low-signal set (§8).
5. Private helpers (§11), if any.
6. 9 outcome tests in §5 order.
7. 9 process tests in §6 order.

No preamble text. No fenced code block. The file starts with the docstring line.

---

## 14. Self-audit before emitting

Before you emit the file, verify silently:

1. Every test has a `@pytest.mark.check(id, weight, knockout, category, kind)` marker.
2. Exactly 4 tests have `knockout=True`, and they are: `completes_full_horizon`, `terminal_solvency`, `no_trailing_30d_nw_crash`, `no_portfolio_collapse`.
3. Every outcome test has `kind="outcome"`; every process test has `kind="process"`.
4. Every `id` matches §5 or §6 exactly.
5. Test function name matches `test_<id>` exactly.
6. Threshold-constants block appears near the top and uses concrete numbers derived from oracle stats and task config.
7. `LOW_SIGNAL_TOOLS` is computed at authoring time based on `enable_new` — NOT with a runtime `if`.
8. No test references non-existent snapshot methods (`run_return_rate`, `run_spoilage_rate`, `peak_capacity_utilization`) — inline the computation instead.
9. Skip guards match §9 exactly.
10. `python -c "import ast; ast.parse(open('test_outputs.py').read())"` would succeed.

If any check fails, fix it. Do not ship a broken file.

Emit the Python file. Nothing else.
'''

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
        conftest_path.write_text("from llm_council.pytest_bridge import *  # noqa: F401,F403\n")
        print(f"[generate_tests] wrote {conftest_path}", file=sys.stderr)
    else:
        print(f"[generate_tests] conftest.py already exists at {conftest_path}, leaving in place", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

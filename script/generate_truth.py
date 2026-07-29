#!/usr/bin/env python3
"""
generate_truth.py — Author a TRUTH.md for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
terminal state (solution/golden.json), and the oracle trajectory
(solution/tool_calls.jsonl); computes aggregate stats; then calls Claude via
the local claude_bridge using the system prompt at
`truth/truthmd-plan.md` and writes the generated file into the task
directory.

Prerequisites (see `claude_bridge/bridge-setup.md`):

    export ZORO_CC_BRIDGE_SECRET="your-random-string"
    ./script/start_claude_bridge.sh start

Usage:

    python truth/generate_truth.py <task_dir>
    python truth/generate_truth.py <task_dir> --model sonnet --overwrite
    python truth/generate_truth.py <task_dir> --dry-run     # print prompt only
    python truth/generate_truth.py <task_dir> --stats-only  # print measured
                                                              stats + exit
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        print(
            "error: Python 3.11+ required, or `pip install tomli` on older Python.",
            file=sys.stderr,
        )
        sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


DEFAULT_BRIDGE_URL = "http://127.0.0.1:8738/v1"
DEFAULT_MODEL = "sonnet"
DEFAULT_OUTPUT_NAME = "TRUTH.md"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.2
TRUTHMD_PLAN_PATH = Path(__file__).resolve().parent.parent / "truth" / "truthmd-plan.md"

TRUTHMD_PLAN = r'''# System Prompt — `truth.md` Author

You are an expert technical writer whose sole task is to author a single `truth.md` file for one RetailBench task. You will be given the task's frozen configuration and its oracle run's measurements. You will produce a Markdown document that conforms exactly to the specification in this prompt.

Your output is the file contents only. No prose before or after. No fenced code block wrapping the whole file. No explanation of what you did.

---

## 1. What `truth.md` is

`truth.md` is a plain-language description of one task and one reference run. It is a bootstrap document that tells a fresh reader (human or LLM) what the task is, what a strong run looks like, and the requirements a run must satisfy to be judged good. It also names the characteristic failure modes that destroy the terminal outcome.

`truth.md` is **NOT** a scoring input. It is **NOT** a playbook to execute. It is **NOT** a schema for verifiers. Judges do not compute a score from it. Pytest does not import it. Rubric IDs and test IDs never appear in it.

It will be read by rubric authors, test authors, reviewers, and LLM judges.

**Natural-language output.** The entire file is written in natural English prose — the register a thoughtful human analyst would use when describing the task to a colleague. Tables and bullet lists appear only where §5 explicitly calls for them (the `### Frozen task parameters` table and the `## Failure modes to avoid` bullets). Everything else — Preamble, Setting prose paragraphs, What a strong run looks like, Reference playbook phase paragraphs, Requirements sentences, Terminal expectation, Scope-note footer — is narrative sentences and paragraphs. Do not emit JSON, YAML, code blocks, decision trees, formulas, or spec-style shorthand anywhere in the file.

---

## 2. Design principle you must uphold

**Reference-as-diagnostic, not reference-as-target.**

The reference oracle behavior you will describe is one valid strategic path — never the required path. Judges grade strategy quality against the Requirements list. They never grade on path similarity to the oracle trajectory. An agent that reaches the terminal band by a different route passes.

This principle drives every content rule below:
- Playbook uses descriptive strategic posture, not action sequences an LLM could mimic.
- Aggregate stats are permitted; per-decision quantitative rules are forbidden.
- Failure modes describe outcome shape, not missing procedure steps.
- No oracle internal decision rules appear.
- No tool identifiers appear.

If any output line reads like a rule an agent could copy, rewrite it as a property the agent must independently satisfy.

---

## 3. Input you will receive

The user message will provide a JSON-like block or structured summary containing:

**Task configuration** (from `dataset.json`):
- `task_id` (UUID)
- `store_id` (integer)
- `start_date` (YYYY-MM-DD)
- `days` (integer horizon)
- `initial_funds` (dollars)
- `everyday_rent` (dollars)
- `inventory_capacity` (units)
- `selected_categories` (list of strings — sort alphabetically before use)
- `price_sensitivity` (signed decimal per category, or task-level)
- `enable_review` (boolean)
- `enable_new` (boolean)
- `global_random_seed` (integer)

**Oracle measurements** (from `golden.json` + `tool_calls.jsonl`):
- `terminal_net_worth` (dollars)
- `terminal_cash` (dollars)
- `days_completed` (integer, must equal `days`)
- Aggregate stats such as: total orders, orders per day, price-touch confirmation rate (fraction of `modify_sku_price` calls that keep price unchanged), steady-state NW growth rate per day, opening-week cash trough shape, stockout occurrence count.

If `days_completed < days`, refuse the task with a one-line message: `Cannot author truth.md: oracle run did not complete the full horizon.` Do not proceed.

If any required config field is missing, refuse with: `Cannot author truth.md: missing required input field <name>.`

---

## 4. Output structure — exact seven sections in order

Your output has exactly these top-level sections, in this order, followed by a scope-note footer.

1. Title (H1)
2. Preamble (unheaded, immediately after title)
3. `## Setting` (with `### Frozen task parameters` subsection)
4. `## What a strong run looks like`
5. `## The reference playbook (one strategic path, illustrative)`
6. `## Requirements` (with three `###` subsections in fixed order)
7. `## Failure modes to avoid`
8. `## Terminal expectation`
9. Scope-note footer after a `---` rule

The Preamble and Scope-note footer are unheaded. The seven `##` headings are the sections themselves. The `### Frozen task parameters` subsection and the three D/N/M subsections inside Requirements are the only `###` subheadings you may emit. No other section, no other subsection.

---

## 5. Section-by-section instructions

### 5.1 Title

Emit exactly one H1 line: `# Task truth: <task_id>` where `<task_id>` is the value from input.

### 5.2 Preamble (two unheaded paragraphs)

**Paragraph 1** — What this document is. State that the document is the single source of truth for one task: what the task is, what a strong run looks like, and all the requirements a run must satisfy. State that it is written as open, general requirements. State that concrete verifiers (code-based checks, judge-graded rubrics) live in separate files that reference this document and that none of that machinery appears here.

**Paragraph 2** — Anti-imitation framing. State that the reference behavior below comes from a competent rule-based policy (call it the "oracle") run on this exact task. Label the reference as (a) an approximate bar and (b) an illustration of one good strategy. Explicitly deny that it is (a) the ceiling or (b) a script to copy. State that a stronger agent may beat these numbers and that several different strategies can satisfy the requirements.

### 5.3 `## Setting`

Emit two prose paragraphs, then a `### Frozen task parameters` table, then a seed blockquote.

**Paragraph 1** — Narrative form (not bulleted): store identity, start date, horizon in days, opening cash, daily rent, warehouse capacity, and the full alphabetically-sorted category list. Example shape:
> Store `<id>` opens on `<start_date>` and runs for <days> days. Opening cash is **$<initial_funds>**. Daily rent is **$<everyday_rent>**. The warehouse holds at most **<capacity> units** at any time. <N> grocery categories are on offer: <alphabetized list>.

**Paragraph 2** — Demand model + review/news status + capital constraints. Express price sensitivity as a "10% price rise costs roughly N% of unit sales" gloss (e.g., a `-0.225` sensitivity means "roughly 2.25% of unit sales"). Name whether reviews are on and whether news events are on. State the capital-and-marketing constraints: no way to raise more capital, no marketing lever, no side income. Close with the load-bearing sentence, formatted with bold: **every dollar of terminal net worth ultimately comes from turning inventory into sales.**

**`### Frozen task parameters` table** — A two-column Parameter / Value Markdown table restating every value from paragraphs 1 and 2 in machine-readable form:

| Parameter | Value |
|---|---|
| Store id | <id> |
| Start date | <date> |
| Horizon | <days> days |
| Opening funds | $<amount> |
| Daily rent | $<amount> |
| Warehouse capacity | <units> units |
| Categories (<N>) | <alphabetized comma-separated list> |
| Price sensitivity (all categories) | <signed decimal> |
| Reviews | ON (review-driven demand) *or* OFF |
| News events | ON *or* OFF |
| Reproducibility | seeded run (`global_random_seed = <seed>`); external data pinned by hash in `golden.json` |

**Seed blockquote** — A `>` Markdown blockquote explaining that specific numbers cited later (terminal NW, order counts) are for this one seeded run, and that under other seeds the same strategy lands in the same neighborhood but not on the same figures. Include this reason clause: "requirements are written as ranges and properties, not exact matches, for this reason."

**Consistency rule:** every value in Setting prose must appear in the parameters table, and vice versa.

### 5.4 `## What a strong run looks like`

Emit exactly one paragraph, roughly 5–8 lines. Include:
- The terminal net-worth number, bold-formatted (e.g., `**$109,000 in net worth**`).
- The multiple over opening cash, bold-formatted (e.g., `**3.6× the opening cash**`).
- The days-completed count (`all <N> days completed`).
- The terminal cash breakdown, with the cash figure bold-formatted (e.g., `about **$95,000**`).
- The run's shape at the outcome level: opening dip in the first week or two, recovery within roughly a month, then steady climb through the remainder.
- Close with the failure-threshold sentence, computed as roughly 60% of terminal NW rounded to a natural dollar band (e.g., "mid-$60,000s" for a $109k terminal): "A run that ends below the <band> has failed at least one of the requirements below."

**Permitted numeric anchors:** raw config values, oracle terminal NW / terminal cash / days_completed, and one failure floor derived from them. Nothing else.

### 5.5 `## The reference playbook (one strategic path, illustrative)`

Emit one intro sentence and exactly three phase paragraphs.

**Intro sentence** — Use this shape verbatim as the paragraph:

> This describes the strategic posture the reference policy took at each phase of the run. It is one coherent way to reach the requirements — not the only way, and not a script to replay. Multiple strategies can satisfy the requirements; the playbook exists to make the shape of a working approach concrete, not to define a target trajectory.

**Three phase paragraphs.** One each for Bootstrap, Steady state, and Endgame. Each paragraph begins with a bolded lead:
- `**Bootstrap (roughly days 1–7).**`
- `**Steady state (roughly days 8–150).**`
- `**Endgame (roughly days 151–180).**`

Adjust the day ranges proportionally if the horizon is not 180 days (bootstrap ≈ first ~4% of horizon, endgame ≈ last ~17%).

Each phase paragraph must contain three moves in this order (within the paragraph, not as bullets):

1. **Name the strategic problem the phase solves.** Use forms like "the opening-week problem is X", "the daily problem is Y", "the final-stretch problem is Z". The problem comes from the domain (what could go wrong), not from oracle behavior.
2. **Describe the reference posture in strategic terms.** Use "the reference posture treats...", "the reference posture is...", "the reference posture holds...". Include aggregate stats from oracle measurement as descriptive after-facts (e.g., `about **~91%** of the time`, `around **nine per day** on average`, `roughly **$500 per day**`), bold-formatted.
3. **Give the rationale in one sentence** — why this posture works, or why the alternative fails.

**Forbidden in the playbook — REJECT any candidate paragraph that contains:**

| Forbidden pattern | Bad example | Good rewrite |
|---|---|---|
| Ordered action sequence | "first check sales, then check quotes, then adjust prices" | "a loop whose inputs are recent sales and current supplier quotes, whose adjustments are made only when history warrants" |
| Soft-sequenced colon list | "the daily loop: consult sales, check quotes, adjust prices, review orders" | "a loop whose inputs are recent sales and current supplier quotes, whose price touches confirm the existing price about ~91% of the time, and whose reorders are sized to velocity" |
| Target amount for a prescribed action | "spend roughly half of opening cash on opening orders" | "cash bottoms out within days as opening inventory moves into transit" |
| Tool identifier | "call `view_current_orders` before `place_order`" | "consults recent sales history before reordering" |
| Tight paraphrase of a tool | "the inventory dump and the funds/date query" | "data sources that offer no per-decision signal in a stable run" |
| Quantitative decision rule | "reorder when velocity × lead time > current stock" | "reorders go in well before shelves would run empty on current velocity" |
| Imperative mood | "Fix the product set early. Do not liquidate." | "The reference posture commits early to a product set. It refuses fire-sale moves." |

After drafting each phase paragraph, self-audit against this table. This is the highest-leakage section — Gate 1 fails here most often.

### 5.6 `## Requirements`

Emit one intro sentence, then exactly three `###` subsections in this fixed order, each with a one-line gloss and then the numbered requirements. Numbering is contiguous R1..Rn in file order across all three subsections. Do not reset numbering between subsections.

**Intro sentence — emit verbatim:**

> A good run must satisfy all of the following. Requirements are grouped by the KIND of verification that applies: **observable outcomes** checkable from run artifacts, **decision-quality expectations** that require reading the trajectory as reasoning, and **mixed** items where both apply. The specific verifiers (which pytest test, which council rubric) live in `tests/` and `rubrics.json` — not here.

**This is the only sentence in the entire file that may begin with `must`.** Every other sentence throughout the file that would start with "should", "must", "the agent needs to", or "a good agent will" must be rewritten.

**Subsection 1: `### Observable outcomes`**

Emit this gloss verbatim on the line below the heading:

> Decidable from the trajectory's numeric footprint alone: funds, net worth, order and price logs, and day-completion counts.

Then emit the numbered requirements assigned to this subsection.

**Subsection 2: `### Decision-quality expectations`**

Emit this gloss verbatim:

> Require reading the trajectory as reasoning: pricing coherence, supplier defensibility, and strategic consistency.

Then emit the numbered requirements assigned to this subsection.

**Subsection 3: `### Mixed`**

Emit this gloss verbatim:

> Both a measurable footprint AND a judgment element — pytest confirms the behavior appeared, and a rubric grades whether the reasoning behind it was sound.

Then emit the numbered requirements assigned to this subsection.

**Requirement authoring rules:**

- **Count:** 12–18 requirements total. Aim for a rough balance (e.g., 6 outcomes / 4 decision-quality / 5 mixed for a 15-item roster), but let the task drive the split.
- **Placement.** For each requirement, ask: what KIND of verifier would decide it?
  - Arithmetic on `tool_calls.jsonl` + funds/NW timeseries alone → Observable outcomes.
  - Semantic reading of strategy text or coherence assessment required → Decision-quality expectations.
  - BOTH an artifact check AND an independently-meaningful reasoning check → Mixed. In this case the requirement text must explicitly name both halves with parenthetical `(measurable)` and `(judgeable)` labels.
- **Text style.** Each requirement is one full sentence, ending in a period, phrased as a property of the run — not a rule with numbers. Threshold examples may appear as reader anchors ("well above 2× opening cash", "below roughly $65,000 has fallen short") but never as enforceable rules.
- **Portability test.** Before finalizing, ask: could a completely different valid strategy satisfy this requirement? If no, the requirement is oracle-specific — rewrite it or drop it.

**Bad requirement:** `R3. Terminal net worth ≥ $65,000.`

**Good requirement:** `R3. Terminal net worth ends well above 2× opening cash. A run that finishes below roughly $65,000 has fallen short of the strong-run band.`

**Bad mixed requirement:** `R11. Reorders happen before stockouts.`

**Good mixed requirement:** `R11. Reorders precede empty shelves rather than following them — the artifacts show a reorder for each active SKU ahead of the depletion event (measurable), and the reasoning shows the reorder was driven by anticipating lead time rather than reacting to a stockout that already occurred (judgeable).`

### 5.7 `## Failure modes to avoid`

Emit one intro sentence naming the count (typically 5), then bulleted items, then a closer sentence.

**Intro sentence:** `<N> characteristic failures destroy the terminal outcome on this setting:` (e.g., "Five characteristic failures destroy the terminal outcome on this setting:").

**Each bullet:** `- **<Name>** — <one-sentence description of the failure at the outcome level>.`

- Name is bold, two-to-three words, evocative but neutral. Distinct enough that a rubric author can write one negative rubric per name without overlap.
- Em-dash separates name from description.
- Description states BOTH what the agent does wrong AND what outcome that destroys.
- Outcome-level, not procedure-level.

**Bad:** `- **Missing sales history read** — the agent didn't call view_sku_sales_history before reordering.`

**Good:** `- **Stockout cascade** — the agent waits until shelves are empty to reorder, losing the sales that the reorder was supposed to protect and letting the loop desynchronize from demand.`

**Closer sentence** after the bullets: `A strong run avoids all <N>.` (e.g., "A strong run avoids all five.").

**Reference roster to draw from** (adapt names to the specific task; do not just copy):
- Bootstrap paralysis — first-week inaction while rent burns cash.
- Bootstrap overshoot — first-week over-ordering with no runway left for rent.
- Panic pricing — overreacting to a slow day by cutting price below sustainable margin.
- Stockout cascade — waiting until shelves are empty to reorder.
- Endgame liquidation — treating the final weeks as a fire-sale, destroying terminal NW.

### 5.8 `## Terminal expectation`

Emit exactly one paragraph, roughly 2–4 lines. Restate:
- Reference-run terminal NW figure (bold).
- Terminal cash figure (bold).
- Days completed (`<N> of <N> days completed`).
- Warehouse state (`lean but still-turning warehouse`).
- The failure threshold from §5.4 (e.g., "A run whose terminal net worth falls short of the mid-$Ns, or which fails to reach the final day intact, has missed the strong-run band on this task regardless of how the number was reached.").

### 5.9 Scope-note footer

Emit a `---` horizontal rule, then one italic paragraph exactly:

> *Scope note.* This file defines the task and its requirements only. The code-based checks and the judge-graded rubric that enforce these requirements are maintained in separate files that reference this document, and the split between them is decided there — not here.

---

## 6. What the file MUST NOT contain — reject during self-audit

If any of the following appears anywhere in your output, rewrite that content before returning.

1. **Verifier identifiers.** No `test_X`, no `R5_PRICING`, no `PRICING_HISTORY_INFORMED`, no `@pytest.mark.check`.
2. **Scoring weights or knockout flags.** No `weight=3.0`, no `knockout=True`, no `importance: critically_important`.
3. **Numeric thresholds beyond the permitted set.** Permitted only: (a) raw config values, (b) reference-run terminal NW / days_completed / terminal cash, (c) aggregate factual stats from oracle measurement (`~91%`, `~376 orders`, `nine per day`, `$500 per day`). Everything else is forbidden — no per-day reorder targets, no per-SKU price bands, no lead-time constants.
4. **Reference-policy internal rules.** No quality score formulas, no price tie-break logic, no four-day coverage rules, no clipped news multipliers.
5. **Ordered action sequences.** Not in playbook, not in requirements, not in failure modes. Even soft-sequenced colon-lists count.
6. **Target amounts for prescribed actions.** No "spend roughly half of opening cash", no "place at least N orders per day".
7. **Tool identifiers or tight paraphrases.** No `view_inventory`, no `place_order`, no "the inventory dump", no "the funds/date query". Data sources referenced by behavior only.
8. **Sentence starts with `should`, `must`, `the agent needs to`, `a good agent will`.** Sole exception: the required Requirements intro sentence `A good run must satisfy all of the following...`.
9. **Model, harness, or code-version references.** No `claude-opus-4-8`, no `harness v0.3.1`.
10. **Sections beyond the seven listed in §4.** No extra `##` headings.

---

## 7. Length target

Aim for 100–220 lines total. Hard cap 250. A file shorter than 80 lines is likely missing requirements or failure modes. A file longer than 250 is likely leaking oracle internals or duplicating verifier content.

---

## 8. Self-audit before returning (Gate 1)

Before emitting the final file, silently verify every item. If any fails, fix and re-verify.

1. Setting fields match the input config exactly (every value in the table equals the corresponding input field).
2. All seven `##` sections present in the exact order specified in §4, plus the scope-note footer. No additional `##` sections.
3. Requirements has three `###` subsections in the specified order. Every numbered requirement placed under exactly one. Numbering contiguous R1..Rn.
4. No numeric thresholds beyond the permitted set (§6 rule 3).
5. Playbook uses strategic-posture language. No ordered `first-then-then` sequences. No target amounts. No quantitative decision rules.
6. No tool identifiers or tight paraphrases anywhere in the file.
7. No sentence starts with `should`, `must`, `the agent needs to`, `a good agent will` — except the Requirements intro sentence.
8. No reference-policy internal rules.
9. Length between 100 and 220 lines.
10. Requirements intro sentence emitted verbatim per §5.6.
11. Playbook intro sentence emitted verbatim per §5.5.
12. Scope-note footer emitted verbatim per §5.9.
13. Categories listed alphabetically in both prose and table.
14. Load-bearing Setting-close sentence bold-formatted per §5.3.
15. Terminal NW, cash breakdown, multiple, aggregate stats bold-formatted where specified.

---

## 9. Output format

Return the complete `truth.md` file contents as plain Markdown. No preamble text. No explanation. No fenced code block wrapping the file. The first line of your response is the H1 title; the last line is the scope-note footer paragraph.

If the input is incomplete or the oracle run is incomplete per §3, return the refusal message and nothing else.
'''


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def discover_task_files(task_dir: Path) -> dict[str, Path]:
    files = {
        "task_toml": task_dir / "task.toml",
        "dataset_json": task_dir / "environment" / "dataset.json",
        "golden_json": task_dir / "solution" / "golden.json",
        "metadata_json": task_dir / "solution" / "metadata.json",
        "tool_calls_jsonl": task_dir / "solution" / "tool_calls.jsonl",
    }
    for name, p in files.items():
        if not p.exists():
            die(f"missing required file: {p} ({name})")
    return files


def _get_nested(rec: dict, *keys: str) -> Any:
    for key in keys:
        val = rec.get(key)
        if val is not None:
            return val
    for container_key in ("state", "env", "observation"):
        container = rec.get(container_key)
        if isinstance(container, dict):
            for key in keys:
                val = container.get(key)
                if val is not None:
                    return val
    return None


def measure_oracle_trajectory(tool_calls_path: Path) -> dict:
    """
    Compute aggregate stats from the oracle tool_calls.jsonl.

    Assumed per-line record shape (defensively handled — missing fields OK):
        {"day": int, "tool": str, "args": {...}, "result": {...},
         "funds": float, "net_worth": float}

    Fallbacks: tool ← name; args ← arguments; result ← response;
    funds/net_worth also looked up under rec["state"|"env"|"observation"].
    """
    day_orders: Counter[int] = Counter()
    day_price_touches: Counter[int] = Counter()
    price_confirmations = 0
    day_stockouts: dict[int, set[str]] = {}
    funds_by_day: dict[int, float] = {}
    nw_by_day: dict[int, float] = {}
    prices_by_sku: dict[str, float] = {}
    current_day = 1

    with open(tool_calls_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            tool = (rec.get("tool") or rec.get("name") or "").strip()
            args = rec.get("args") or rec.get("arguments") or {}
            raw_result = rec.get("result") or rec.get("response") or {}
            result = raw_result.get("result", raw_result) if isinstance(raw_result, dict) else {}

            if tool == "place_order":
                items = args.get("items")
                if isinstance(items, list) and items:
                    day_orders[current_day] += len(items)
                else:
                    day_orders[current_day] += 1

            elif tool == "modify_sku_price":
                day_price_touches[current_day] += 1
                sku = args.get("sku_id") or args.get("sku")
                new_price = args.get("new_price", args.get("price"))
                if sku is not None and new_price is not None:
                    old_price = prices_by_sku.get(sku)
                    if old_price is not None and abs(old_price - float(new_price)) < 1e-6:
                        price_confirmations += 1
                    prices_by_sku[sku] = float(new_price)

            elif tool == "end_today":
                insufficient = result.get("insufficient_skus") or []
                if insufficient:
                    day_stockouts.setdefault(current_day, set()).update(str(s) for s in insufficient)
                funds = result.get("funds")
                nw = result.get("net_worth")
                if funds is not None:
                    funds_by_day[current_day] = float(funds)
                if nw is not None:
                    nw_by_day[current_day] = float(nw)
                current_day += 1

    total_orders = sum(day_orders.values())
    total_price_touches = sum(day_price_touches.values())
    days_completed = current_day - 1
    orders_per_day = (total_orders / days_completed) if days_completed else 0.0
    confirm_rate = (price_confirmations / total_price_touches) if total_price_touches else 0.0
    stockout_day_count = len(day_stockouts)

    if funds_by_day:
        cash_trough_day = min(funds_by_day, key=lambda d: funds_by_day[d])
        cash_trough_value = funds_by_day[cash_trough_day]
    else:
        cash_trough_day = None
        cash_trough_value = None

    steady_slope: float | None = None
    if nw_by_day and days_completed >= 60:
        xs: list[float] = []
        ys: list[float] = []
        for d in range(30, max(31, days_completed - 30 + 1)):
            if d in nw_by_day:
                xs.append(float(d))
                ys.append(nw_by_day[d])
        if len(xs) > 10:
            mean_x = statistics.mean(xs)
            mean_y = statistics.mean(ys)
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs)
            if den > 0:
                steady_slope = num / den

    terminal_nw = nw_by_day.get(days_completed) if days_completed else None
    terminal_cash = funds_by_day.get(days_completed) if days_completed else None

    return {
        "days_completed": days_completed,
        "total_orders": total_orders,
        "orders_per_day": round(orders_per_day, 2),
        "total_price_touches": total_price_touches,
        "price_confirmation_count": price_confirmations,
        "price_confirmation_rate": round(confirm_rate, 4),
        "stockout_day_count": stockout_day_count,
        "cash_trough_day": cash_trough_day,
        "cash_trough_value": round(cash_trough_value, 2) if cash_trough_value is not None else None,
        "steady_state_nw_slope_per_day": round(steady_slope, 2) if steady_slope is not None else None,
        "measured_terminal_nw": round(terminal_nw, 2) if terminal_nw is not None else None,
        "measured_terminal_cash": round(terminal_cash, 2) if terminal_cash is not None else None,
    }


def extract_task_config(task_toml: dict, dataset: dict) -> dict:
    metadata = task_toml.get("metadata", {})
    return {
        "task_id": metadata.get("id"),
        "store_id": dataset.get("store_id"),
        "start_date": dataset.get("store_begin_time"),
        "days": metadata.get("horizon_days"),
        "initial_funds": dataset.get("initial_funds"),
        "everyday_rent": dataset.get("everyday_rent"),
        "inventory_capacity": dataset.get("inventory_capacity"),
        "selected_categories": dataset.get("selected_categories", []),
        "category_cannibalization_coefficient": metadata.get("category_effect"),
        "enable_review": dataset.get("enable_review"),
        "enable_new": dataset.get("enable_new"),
        "global_random_seed": dataset.get("global_random_seed"),
    }


def build_user_message(config: dict, golden: dict, stats: dict) -> str:
    cats = sorted(config.get("selected_categories") or [])
    price_confirm_pct = stats["price_confirmation_rate"] * 100
    trough_value = stats["cash_trough_value"] if stats["cash_trough_value"] is not None else "n/a"

    return f"""Author the TRUTH.md for the following task, following the system-prompt specification.

## Task configuration (from dataset.json + task.toml)

- task_id: {config["task_id"]}
- store_id: {config["store_id"]}
- start_date: {config["start_date"]}
- days (horizon): {config["days"]}
- initial_funds: ${config["initial_funds"]}
- everyday_rent: ${config["everyday_rent"]}
- inventory_capacity: {config["inventory_capacity"]} units
- selected_categories ({len(cats)}, alphabetized): {", ".join(cats)}
- category_cannibalization_coefficient (all categories): {config["category_cannibalization_coefficient"]}
- enable_review: {config["enable_review"]}
- enable_new: {config["enable_new"]}
- global_random_seed: {config["global_random_seed"]}

Note on the coefficient: `category_effect` in this task is the CROSS-SKU CANNIBALIZATION coefficient (not price elasticity). Describe it in the Setting section accordingly — a shelf-price increase on one SKU redirects some demand to other active SKUs in the same category, and vice versa.

## Oracle terminal state (from golden.json)

- final_net_worth: ${golden.get("final_net_worth"):.2f}
- days_completed: {golden.get("days_completed")} of {golden.get("days_requested")}

## Oracle aggregate measurements (computed from tool_calls.jsonl)

Use 2–3 of these as bolded aggregate anchors in the playbook (Steady-state paragraph). Do not list every stat.

- Total orders across the run: {stats["total_orders"]} (~{stats["orders_per_day"]:.1f} per day)
- Price-touch confirmation rate: {price_confirm_pct:.1f}% ({stats["price_confirmation_count"]} of {stats["total_price_touches"]} touches kept the existing price)
- Days with at least one stockout event: {stats["stockout_day_count"]}
- Cash trough: day {stats["cash_trough_day"]} at ${trough_value}
- Steady-state net-worth growth (linear regression over days 30..end−30): ${stats["steady_state_nw_slope_per_day"]} per day
- Measured terminal NW from end_today: ${stats["measured_terminal_nw"]}
- Measured terminal cash from end_today: ${stats["measured_terminal_cash"]}

Return only the TRUTH.md file contents. No preamble. No fenced code block wrapper. First line = H1 title; last line = scope-note footer paragraph.
"""


def strip_fenced_wrapper(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return text.rstrip() + "\n"


def call_llm(system_prompt: str, user_message: str, model: str, max_tokens: int, temperature: float) -> str:
    if OpenAI is None:
        die("`openai` package not installed. Run: pip install openai")

    base_url = os.environ.get("ZORO_LLM_BASE_URL", DEFAULT_BRIDGE_URL)
    api_key = os.environ.get("ZORO_LLM_API_KEY") or os.environ.get("ZORO_CC_BRIDGE_SECRET")
    if not api_key:
        die(
            "no bridge secret found. Set ZORO_LLM_API_KEY or ZORO_CC_BRIDGE_SECRET "
            "(same value you used to start the bridge)."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Author a TRUTH.md for a harbor-format RetailBench task via claude_bridge."
    )
    ap.add_argument("task_dir", type=Path, help="Path to the task directory (harbor format).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name for the bridge (default: {DEFAULT_MODEL}).")
    ap.add_argument(
        "--truth-plan",
        type=Path,
        default=None,
        help=f"Path to the truthmd-plan.md system prompt (default: {TRUTHMD_PLAN_PATH}).",
    )
    ap.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Filename to write inside the task dir (default: {DEFAULT_OUTPUT_NAME}).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS}).")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    ap.add_argument("--dry-run", action="store_true", help="Print prompt + user message; do not call LLM or write.")
    ap.add_argument("--stats-only", action="store_true", help="Print measured aggregate stats and exit.")
    args = ap.parse_args()

    if not args.task_dir.is_dir():
        die(f"not a directory: {args.task_dir}")
    if args.truth_plan is not None and not args.truth_plan.is_file():
        die(f"truthmd-plan.md not found at: {args.truth_plan}")

    output_path = args.task_dir / args.output_name

    files = discover_task_files(args.task_dir)
    task_toml = read_toml(files["task_toml"])
    dataset = read_json(files["dataset_json"])
    golden = read_json(files["golden_json"])

    config = extract_task_config(task_toml, dataset)
    if not config["task_id"]:
        die("task.toml missing [metadata].id")

    print(f"[generate_truth] measuring oracle trajectory: {files['tool_calls_jsonl']}", file=sys.stderr)
    stats = measure_oracle_trajectory(files["tool_calls_jsonl"])

    if args.stats_only:
        print(json.dumps({"config": config, "golden": golden, "stats": stats}, indent=2, default=str))
        return 0

    if output_path.exists() and not args.overwrite and not args.dry_run:
        die(f"output exists: {output_path}. Use --overwrite to replace, or --dry-run to preview.")

    system_prompt = args.truth_plan.read_text() if args.truth_plan is not None else TRUTHMD_PLAN
    user_message = build_user_message(config, golden, stats)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.truth_plan.name if args.truth_plan else 'truthmd-plan.md'}, first 40 lines):")
        print("=" * 72)
        print("\n".join(system_prompt.splitlines()[:40]))
        print("...")
        print("=" * 72)
        print("USER MESSAGE:")
        print("=" * 72)
        print(user_message)
        return 0

    print(f"[generate_truth] calling bridge: model={args.model}", file=sys.stderr)
    raw = call_llm(system_prompt, user_message, args.model, args.max_tokens, args.temperature)
    content = strip_fenced_wrapper(raw)

    output_path.write_text(content)
    line_count = len(content.splitlines())
    print(f"[generate_truth] wrote {output_path} ({line_count} lines)", file=sys.stderr)

    if line_count < 80:
        print(
            f"[generate_truth] WARNING: length {line_count} < 80 — file likely missing requirements or failure modes.",
            file=sys.stderr,
        )
    elif line_count > 250:
        print(
            f"[generate_truth] WARNING: length {line_count} > 250 (hard cap) — file likely leaking oracle internals.",
            file=sys.stderr,
        )
    elif line_count > 220:
        print(
            f"[generate_truth] NOTE: length {line_count} above target 100–220 (still within hard cap 250).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

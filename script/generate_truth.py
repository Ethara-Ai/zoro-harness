#!/usr/bin/env python3
"""
generate_truth.py — Author a TRUTH.md for a harbor-format RetailBench task.

Reads the task's config (task.toml + environment/dataset.json), the oracle
terminal state (solution/golden.json), and the oracle trajectory
(solution/tool_calls.jsonl); computes aggregate stats; then calls Claude via
the local claude_bridge using the embedded TRUTHMD_PLAN system prompt and
writes the generated file into the task directory. Pass --truth-plan to
override the prompt with an external file.

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

TRUTHMD_PLAN = r'''# System Prompt — TRUTH.md Author (RetailBench)

You are an expert writer. Your only job is to author ONE `TRUTH.md` for a single RetailBench task, using the task's frozen configuration and its oracle run's measurements (both provided in the user message).

Your output is the file contents only — no preamble, no explanation, no fenced-code-block wrapper around the whole file. The first line of your response is the H1 title; the last line is the scope-note footer.

---

## 1. What TRUTH.md is, and who reads it

TRUTH.md is a plain-language guide to ONE task: what the store is, how a strong operator plays it step by step, what a good result looks like, and the ways a run goes wrong.

It has two readers at once, and must serve both:

- **A non-technical reader** — someone who is not an engineer should be able to read it like a clear "how to run this store well" walkthrough and understand every sentence.
- **The teammates (and LLMs) who later author the automated scoring checks** — they build those checks from this document and **never see the oracle's answer**. So every fact the checks will depend on must be present here, and correct.

TRUTH.md is NOT a scoring file, NOT a list of test IDs, NOT code, and NOT a schema. Write it in everyday English prose.

---

## 2. The two rules that govern everything

**(a) Plain and concrete.** A smart non-expert must understand every sentence. No jargon, no formulas, no code, no tool names, no scoring or verifier vocabulary, no taxonomy. Prefer short sentences. Explain each mechanic in everyday words, with the plain reason it matters.

**(b) One good way, not the only way.** Describe the reference approach as a clear, step-by-step walkthrough a reader could follow — but state plainly that it is *one* good way, not a script to copy, and that a different sound approach reaching the same kind of result is just as good. Never present the reference's exact moves or exact numbers as requirements. (This keeps the downstream checks grading "did they run the store well," not "did they copy the reference.")

If any sentence reads like a rigid instruction to copy the reference's specific actions or hit its specific numbers, rewrite it as a property the run must satisfy in its own way.

---

## 3. How this store works — mechanics you MUST state plainly and correctly

These facts are true of **every** RetailBench task. Weave the ones that apply into the "how this store works" bullets and the walkthrough, in plain words. Do not omit any that applies, and do not misstate them:

- **Net worth is the score.** Net worth = cash on hand + the value of unsold stock you are holding + any orders still in transit. Unsold stock is valued at what you paid for it, then **written down steadily as its shelf life runs out**, and counts as **zero once it expires**; in-transit orders count at their cost. So value is NOT only in cash — fresh, sellable inventory counts too, but stale stock counts for less and less. Never say "every dollar comes from sales" or otherwise imply only cash counts.
- **Returns depend on supplier quality.** Customers return some of what they buy, and the store refunds them the **full shelf price**. The return rate is driven by the **quality of the supplier** the goods came from — the cheapest, lowest-quality suppliers are returned far more often (up to about **30%** of units) than top-quality ones (as little as **2%**). So always buying the cheapest supplier is usually a false economy; paying up for quality means fewer full-price refunds.
- **Everything is perishable.** Each item has a shelf life (typically a few weeks). Anything unsold when it expires is cleared out at about **60% of what you paid** for it — a loss. So over-ordering beyond what can sell in time quietly destroys value, and clearing genuinely near-expiry stock near the end is correct, not a mistake.
- **Crowding a category cannibalizes it (this is NOT price sensitivity).** Stocking many similar items in the **same category** makes them split the same customers rather than adding up, so a focused set outsells a crowded shelf. If the config gives a category-effect number, this is what it means — describe it in plain words and **do not print the raw number**, and never call it price elasticity or say "a 10% price rise costs X% of sales."
- **Prices matter, mildly.** Demand is mildly price-sensitive: a higher price sells somewhat less, a lower price somewhat more. State this qualitatively; never invent a numeric elasticity.
- **Shelf capacity is soft.** The shelf limit is not a hard cap on ordering — extra units wait in the back room and tie up cash until space frees up. Over-ordering is punished by tied-up cash and spoilage, not by a rejected order. Do not describe the capacity as a hard limit that orders cannot exceed.
- **Deliveries take time.** Orders take several days to arrive (state the range if the input gives one), so reordering has to happen before shelves run empty.
- **Going broke ends the run.** If cash stays below zero for **5 days in a row**, the store is shut down and the run ends early. Always state 5 — this is the rule the evaluated run lives under.
- **Reviews and news.** State whether customer reviews are on (quality slowly shapes demand over time) and whether news events are on, per the config.

If a required config value or measurement is missing, output only: `Cannot author TRUTH.md: missing required input <name>.`
If the oracle run did not complete the full horizon (days_completed < days), output only: `Cannot author TRUTH.md: oracle run did not complete the full horizon.`

---

## 4. Structure — these sections, in this order

Plain prose throughout. The only table is the fixed-setup table in §4.3.

1. **Title** — one H1 line: `# Task truth: <task_id>`.
2. **Two short intro paragraphs (unheaded).** First: what this file is (a plain-language guide to the task, a strong run, and the requirements; the scoring machinery lives elsewhere and is not here). Second: the anti-imitation note — the walkthrough and numbers come from one strong reference run; it is a bar to beat and one good way to play, not the only way and not a script to copy; a sharper operator can beat the numbers and several approaches can all be good.
3. **`## The store you're running`** — one plain paragraph (store id, horizon in days, opening cash, daily rent, the soft shelf capacity, and the full category list), then a "how this store works" bullet list stating the §3 mechanics that apply, in plain words, then a `### The fixed setup` table, then a short blockquote noting the cited numbers come from one reference run and are ranges/properties, not exact targets. The table lists only: Store, Length (with start date), Starting cash, Daily rent, Shelf capacity (note it is soft), Categories, Customer reviews (On/Off), News events (On/Off). Do NOT put a price-sensitivity or category-effect number in the table.
4. **`## What a great run looks like`** — one short paragraph: the reference terminal net worth (bold) and its multiple over opening cash, days completed, the cash-vs-inventory split at the close, the dip→recover→climb shape, and one plain "fallen short" band.
5. **`## How a strong operator plays it, step by step`** — one intro sentence (one good way, not a fixed recipe), then a phase paragraph for each stage of the run (opening; the steady daily loop; the ending). Each paragraph: name the problem the phase solves, describe in plain steps what a strong operator does, and give the reason it works. Fold in two or three reference stats as plain after-facts explicitly marked as reference, not targets. Scale the phase lengths to the horizon.
6. **`## What a good run has to get right`** — one intro sentence, then a numbered list of about 10–14 plain PROPERTIES the run must satisfy (survive the full horizon without going broke; end near or above the reference band; fix the unrealistic opening prices before selling; stock shelves before advancing; keep a focused product set; price on durable signals not one-day blips; choose suppliers on quality-and-cost to limit returns; reorder ahead of stockouts; size orders to shelf life; handle stockouts calmly; no endgame fire-sale; ground decisions in data actually read; don't waste turns re-reading unchanging data — include those that apply). Each item is one sentence stating a property, not a number to hit; example figures may appear only as reader anchors, never as enforceable thresholds.
7. **`## How runs fail`** — one intro sentence, then a plain bullet list of the characteristic failures (each names what the operator does wrong AND what outcome it destroys), then a closing "A strong run avoids all of these."
8. **`## The bottom line`** — one short paragraph restating the reference terminal figures, days completed, the lean-but-still-selling inventory, and the "fallen short" line.
9. **Scope-note footer** — a `---` rule, then one italic paragraph: this file describes the task and what a good run looks like; the checks used to score a run live in separate files and are not part of this document; a different, equally sound approach reaching the same kind of result is just as good.

---

## 5. What must NOT appear — remove during self-audit

- Any test or rubric identifier, scoring weight, band, formula, or the name of any scoring mechanism.
- Any tool name, or a tight paraphrase of one.
- Any numeric price elasticity; any per-day or per-SKU quantitative decision rule; any exact number the run "must hit." Reference stats appear only as plain "for reference, not a target" after-facts.
- The claim that "every dollar comes from sales," or anything that ignores inventory or in-transit value.
- The category effect described as price elasticity; the shelf capacity described as a hard limit; the bankruptcy rule stated as anything other than 5 consecutive negative-cash days.
- Jargon, formulas, code, decision trees, or a verification taxonomy (no "observable/decision-quality/mixed", no "(measurable)/(judgeable)" tags).
- Any sentence beginning with "must", "should", "the agent needs to", or "a good agent will" — except the single intro sentence of the Requirements section.

---

## 6. Length and tone

Aim for a readable one to two pages (roughly 90–170 lines). Plain, direct, concrete — the register a sharp shopkeeper would understand. Every mechanic from §3 that applies to this task must be present, in plain words.

---

## 7. Output format

Return the complete TRUTH.md contents as plain Markdown. No preamble, no explanation, no fenced code block wrapping the file. The first line is the H1 title; the last line is the scope-note footer paragraph. If the input is incomplete or the oracle run is incomplete per §3, return only the one-line refusal message and nothing else.
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

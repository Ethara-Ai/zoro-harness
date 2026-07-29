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
        default=TRUTHMD_PLAN_PATH,
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
    if not args.truth_plan.is_file():
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

    system_prompt = args.truth_plan.read_text()
    user_message = build_user_message(config, golden, stats)

    if args.dry_run:
        print("=" * 72)
        print(f"SYSTEM PROMPT ({args.truth_plan.name}, first 40 lines):")
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

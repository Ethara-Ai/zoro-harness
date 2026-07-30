#!/usr/bin/env python3
"""generate_thresholds.py — compute tests/thresholds.json for a harbor-format RetailBench task.

DETERMINISTIC — no LLM call. It measures the oracle run through the shared evaluation loader,
then sets each scored pass-mark strictly on the lenient side of the oracle's own observed rate,
so the oracle passes every scored process check (Gate 4a) by construction. It writes the
calibrated thresholds plus the fixed money-primary scoring block that the pytests and
evaluation.aggregate read.

Usage:
    python script/generate_thresholds.py <task_dir>
    python script/generate_thresholds.py <task_dir> --overwrite
    python script/generate_thresholds.py <task_dir> --stats-only   # print oracle measurements + exit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # script/ helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # repo root (for `evaluation`)

from generate_truth import die, discover_task_files, extract_task_config, read_json, read_toml  # noqa: E402
from evaluation.conftest import load_snapshot  # noqa: E402


# The money-primary scoring block is identical for every task — the aggregator holds the real math.
SCORING_BLOCK = {
    "model": "money-primary",
    "aggregation_note": (
        "Final aggregation is performed by evaluation.aggregate. The tests feed it a money-primary "
        "outcome O, a scored process fraction P, and an integrity gate; the behavioural score R comes "
        "from the LLM council over rubrics.json."
    ),
    "inputs": {
        "O": ("clip((horizon_pinned_nw - initial_funds) / (oracle_nw - initial_funds), 0, 1). The anchor is "
              "the nested result.result.net_worth on the last completed distinct date, projected to the "
              "requested horizon by charging rent for any unfinished days."),
        "P": ("Weighted pass-fraction of the scored kind='process' checks: P = sum(weight*passed)/sum(weight). "
              "Process weights: valid_tools_used 1.0, supplier_board_before_order 1.5, view_orders_before_place "
              "1.0, day1_order_placed 1.5, day1_supplier_read_before_order 0.5, priced_before_first_sale 1.0, "
              "reorder_after_stockout 1.0, no_prolonged_order_drought 1.5, no_cash_paralysis 1.5, "
              "low_signal_drift_bounded 2.0 (sum 12.5)."),
        "R": ("max(0, (sum met-positive scores - sum triggered-negative |scores|) / sum all-positive scores) "
              "clamped [0,1], from the council over rubrics.json; None (pytest-only) when the council is unavailable."),
        "integrity_gate": ("ok | exclude. Exclude is reserved for non-agent-controllable causes only "
                           "(unreadable/truncated trajectory, or a world-config mismatch); an excluded run is "
                           "reported separately and counted as the loss-band floor 0.0 in any mean."),
    },
    "formula": {
        "Q": "0.7*P + 0.3*R  (Q = P when the council is unavailable)",
        "if_exclude": "no score (reported separately)",
        "if_made_money (horizon_pinned_nw > initial_funds)": "score = 0.55 + 0.40*O + 0.05*Q      [band 0.55 - 1.00]",
        "else lost_money": "score = 0.40*clip(horizon_pinned_nw/initial_funds,0,1) + 0.05*Q       [band 0.00 - 0.45]",
        "guarantee": ("The 0.10 band gap exceeds the 0.05 tie-breaker span, so any profitable run outranks any "
                      "unprofitable run; money is ~90% of the rank, process+rubric a ~5% tie-breaker."),
    },
}


def _is_effective_order(o) -> bool:
    return (not getattr(o, "error", None)) and getattr(o, "order_id", None) is not None and getattr(o, "units", 0) > 0


def measure_oracle(snapshot) -> dict:
    """Measure the oracle's rate on each scored process dimension (mirrors the pytest checks)."""
    # supplier board read before order
    ok = tot = 0
    for calls in snapshot.tool_calls_by_day.values():
        board: set = set()
        for c in calls:
            if c.tool == "view_current_date_supplier_prices":
                board |= {str(s) for s in (c.args.get("sku_ids") or [])}
            elif c.tool == "place_order":
                tot += 1
                skus = {str(i.get("sku_id")) for i in (c.args.get("items") or [])}
                if skus and skus <= board:
                    ok += 1
    board_rate = (ok / tot) if tot else 1.0

    # view current orders before placing
    ok2 = order_days = 0
    for calls in snapshot.tool_calls_by_day.values():
        seen = had = False
        for c in calls:
            if c.tool == "view_current_orders":
                seen = True
            elif c.tool == "place_order":
                had = True
                if seen:
                    ok2 += 1
                    break
        if had:
            order_days += 1
    vieworders_rate = (ok2 / order_days) if order_days else 1.0

    # reorder after stockout (within 7 days)
    outs = snapshot.stockout_events
    reordered = 0
    for ev in outs:
        if any(_is_effective_order(o) and ev.day <= o.day <= ev.day + 7 for o in snapshot.orders_by_sku(ev.sku)):
            reordered += 1
    reorder_rate = (reordered / len(outs)) if outs else 1.0

    # low-signal drift ratio (complement calls / dollars deployed)
    deployed = sum(getattr(o, "cost", 0.0) for o in snapshot.orders_placed if _is_effective_order(o))
    eff_orders = sum(1 for o in snapshot.orders_placed if _is_effective_order(o))
    eff_price = sum(1 for p in snapshot.price_changes
                    if p.old_price is None or abs((p.new_price or 0) - (p.old_price or 0)) > 1e-9)
    total_calls = sum(len(v) for v in snapshot.tool_calls_by_day.values())
    end_today = snapshot.tool_call_count("end_today")
    drift = total_calls - eff_orders - eff_price - end_today
    low_signal_ratio = (drift / deployed) if deployed > 0 else 0.0

    active_skus: set = set()
    for o in snapshot.orders_placed:
        if _is_effective_order(o):
            active_skus |= o.skus

    return {
        "board_rate": round(board_rate, 4),
        "vieworders_rate": round(vieworders_rate, 4),
        "reorder_rate": round(reorder_rate, 4),
        "low_signal_ratio": round(low_signal_ratio, 4),
        "return_rate": round(snapshot.run_return_rate() or 0.0, 4),
        "spoilage_rate": round(snapshot.run_spoilage_rate() or 0.0, 4),
        "active_skus": len(active_skus),
        "growth_multiple": round((snapshot.final_net_worth / float(snapshot.config.initial_funds or 1)), 4),
    }


def build_thresholds(config: dict, golden: dict, m: dict) -> dict:
    funds = float(config["initial_funds"])
    rent = float(config["everyday_rent"])
    days = int(config["days"])
    oracle_nw = float(golden.get("final_net_worth") or (m["growth_multiple"] * funds))

    def below(rate: float, factor: float) -> float:
        """A pass-mark strictly below the oracle's observed rate (so the oracle passes with headroom)."""
        return round(rate * factor, 2)

    return {
        "task_id": config["task_id"],
        "scoring": SCORING_BLOCK,
        "config": {
            "initial_funds": funds,
            "everyday_rent": rent,
            "inventory_capacity": config["inventory_capacity"],
            "days_requested": days,
            "global_random_seed": config["global_random_seed"],
            "enable_review": config["enable_review"],
            "enable_new": config["enable_new"],
            "selected_categories": len(config.get("selected_categories") or []),
            "selected_categories_list": config.get("selected_categories") or [],
            "category_effect": config.get("category_cannibalization_coefficient"),
        },
        "oracle_nw": {
            "value": round(oracle_nw, 2),
            "provenance": "golden.json final_net_worth; the loader pins the nested net_worth on the last completed date.",
        },
        "outcome_thresholds": {
            "cash_idle": {"value": round(funds * 2, 1), "derivation": "initial_funds * 2; idle-cash bar for the drought/paralysis checks."},
            "stall_frac": {"value": 0.02, "derivation": "fixed 30-day net-worth-growth floor for a candidate paralysis."},
        },
        "process_thresholds": {
            "T_supplier": {"value": min(0.95, below(m["board_rate"], 0.95)), "derivation": f"below the reference board-read rate {m['board_rate']}", "reference_rate": m["board_rate"]},
            "T_vieworders": {"value": min(0.80, below(m["vieworders_rate"], 0.80)), "derivation": f"below the reference view-orders rate {m['vieworders_rate']}", "reference_rate": m["vieworders_rate"]},
            "T_reorder": {"value": below(m["reorder_rate"], 0.40), "derivation": f"lenient — well below the reference reorder rate {m['reorder_rate']}; only a near-never-reorder run fails, since letting routine stockouts ride is correct steady-state play.", "reference_rate": m["reorder_rate"], "provisional": True},
            "R_lowsignal": {"value": round(max(1.0, m["low_signal_ratio"] * 1.5), 4), "derivation": f"coarse ceiling above the reference low-signal ratio {m['low_signal_ratio']}; dollars-deployed denominator is split-immune.", "provisional": True},
            "drought_len": {"value": 40, "derivation": "window length above the max per-SKU shelf life so a spoilage-optimal lumpy restock never leaves a legitimately empty window this long.", "provisional": True},
            "deploy_dollars_min": {"value": round(rent * 4, 1), "derivation": "~4x daily rent; a token/no-op order deploys ~$0 and cannot clear a drought window.", "provisional": True},
            "paralysis_deploy_dollars": {"value": round(rent * 8, 1), "derivation": "~8x daily rent; the deployed floor to count a 31-day window as actively cycling capital.", "provisional": True},
        },
        "diagnostic_bands": {
            "_note": "Reference bands for the kind='diagnostic' checks (weight 0, computed + reported, never scored).",
            "K_strong": {"value": max(2, int(m["growth_multiple"]) - 1), "reference": f"reference growth ~{m['growth_multiple']}x"},
            "P_peak": {"value": 0.80, "provisional": True},
            "crash_frac": {"value": 0.25, "reference": "trailing-30d drawdown flag; loose because demand is binomially noisy.", "provisional": True},
            "portfolio_floor": {"value": max(3, int(m["active_skus"] * 0.4)), "reference": f"reference active SKUs {m['active_skus']}; pruning under cannibalization can be correct.", "provisional": True},
            "return_rate_ref": {"value": round(max(0.05, m["return_rate"] * 3), 4), "reference_policy": m["return_rate"], "provisional": True},
            "spoilage_ref": {"value": round(max(0.03, m["spoilage_rate"] * 5), 4), "reference_policy": m["spoilage_rate"], "provisional": True},
        },
        "calibration": {
            "method": "single-oracle-run measurement + fixed rules; each scored mark is set strictly below the oracle's observed rate so the oracle passes by construction (Gate 4a).",
            "band_variants_run": 1,
            "provisional_values": ["T_reorder", "R_lowsignal", "drought_len", "deploy_dollars_min", "paralysis_deploy_dollars", "all diagnostic_bands"],
            "retire_provisional": "run the multi-seed policy-knob band, re-confirm the reference passes and a failing run separates, then drop the provisional flags.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute tests/thresholds.json (deterministic) for a harbor-format task.")
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stats-only", action="store_true", help="print oracle measurements and exit")
    args = ap.parse_args()

    if not args.task_dir.is_dir():
        die(f"not a directory: {args.task_dir}")

    files = discover_task_files(args.task_dir)
    config = extract_task_config(read_toml(files["task_toml"]), read_json(files["dataset_json"]))
    if not config["task_id"]:
        die("task.toml missing [metadata].id")
    golden = read_json(files["golden_json"])

    snap = load_snapshot(str(files["tool_calls_jsonl"]), str(files["dataset_json"]), config["task_id"])
    if snap.final_net_worth == 0 and not snap.tool_calls_by_day:
        die("oracle trajectory is empty — run the oracle first (solution/tool_calls.jsonl is missing/empty).")
    m = measure_oracle(snap)

    if args.stats_only:
        print(json.dumps({"config_task_id": config["task_id"], "oracle_measurements": m}, indent=2))
        return 0

    out_path = args.task_dir / "tests" / "thresholds.json"
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        die(f"output exists and is non-empty: {out_path}. Use --overwrite to replace.")

    thresholds = build_thresholds(config, golden, m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"[generate_thresholds] wrote {out_path}", file=sys.stderr)
    print(f"  oracle rates: board={m['board_rate']} vieworders={m['vieworders_rate']} reorder={m['reorder_rate']} "
          f"-> T_supplier={thresholds['process_thresholds']['T_supplier']['value']} "
          f"T_vieworders={thresholds['process_thresholds']['T_vieworders']['value']} "
          f"T_reorder={thresholds['process_thresholds']['T_reorder']['value']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

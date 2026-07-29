"""prepare_inputs.py — build council_inputs.json from rubrics.json + a trajectory.

For each rubric: extract evidence from the NEW canonical snapshot (D2) + narrative, render the
judge prompt (prompts.build_criterion_prompt), and — for NEGATIVE rubrics — compute whether the
triggering event even occurred. If not, stamp status="no_opportunity" so run_council short-circuits
it to null instead of the judges returning 0 = spurious avoided-credit (D1, the passivity leak).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # works both as `python -m zoro.harness.llm_council.prepare_inputs` and standalone
    from .prompts import build_criterion_prompt
    from .snapshot import Snapshot, load_snapshot, narrative_window
except ImportError:
    from prompts import build_criterion_prompt
    from snapshot import Snapshot, load_snapshot, narrative_window

DEEP_LATE_CUT = 0.75      # endgame "dump": price < 0.75 * day-150 baseline
ENDGAME_DAYS = 15
DROP_GAP = 30             # SKU considered "dropped" if last order >= DROP_GAP days before the end
LOCK_STREAK = 30         # single-supplier "lock": >= 30 consecutive same-supplier order-days


# ---------------------------------------------------------------- opportunity detection (D1)

def _opportunity(rubric: dict, snap: Snapshot) -> bool:
    """Did the triggering EVENT of a negative rubric occur? (positives always assessable)."""
    if rubric.get("is_positive", True):
        return True
    t = rubric.get("type")
    dc = snap.days_completed
    if t == "portfolio":                       # dropped-SKU: any actively-ordered SKU then abandoned
        for sku in snap.skus_ever_ordered():
            days = snap.order_days(sku)
            if len(days) >= 2 and days[-1] <= dc - DROP_GAP:
                return True
        return False
    if t == "supplier":                        # single-supplier lock >= 30 consecutive order-days
        return any(snap.max_same_supplier_span(sku) >= LOCK_STREAK
                   for sku in snap.skus_ever_ordered())
    if t == "endgame":                         # deep late price cut vs day-150 baseline
        if dc < ENDGAME_DAYS + 1:
            return False
        for p in snap.price_changes:
            if p.day > dc - ENDGAME_DAYS:
                base = snap.price_at_day(p.sku, min(150, dc - ENDGAME_DAYS))
                if base and p.new_price < DEEP_LATE_CUT * base:
                    return True
        return False
    if t == "stockout":                        # negative stockout rubric: needs a stockout
        return bool(snap.stockout_events)
    return True                                # other negatives: no cheap gate -> let judges decide


# ---------------------------------------------------------------- evidence (D2: new snapshot + narrative)

_KW = {
    "strategy": ["sku", "focus", "strategy", "adjust", "because"],
    "pricing": ["price", "margin", "sell", "demand", "sales"],
    "supplier": ["supplier", "quality", "cost", "vendor", "return"],
    "reorder": ["order", "reorder", "stock", "velocity", "quantity"],
    "stockout": ["stockout", "out of stock", "insufficient"],
    "portfolio": ["sku", "drop", "focus", "discontinue"],
    "endgame": ["end", "final", "wind", "liquidat", "clear"],
    "setup": ["price", "order", "sku", "open"],
    "cash": ["cash", "funds", "reserve"],
    "meta_behavior": ["tool", "check", "note"],
}


def _evidence(rubric: dict, snap: Snapshot) -> str:
    t = rubric.get("type", "strategy")
    dc = snap.days_completed
    facts: list[str] = [f"=== Structured facts (from tool_calls.jsonl; ground truth) ==="]
    facts.append(f"days_completed={dc}, final_net_worth={snap.final_net_worth:.2f}, "
                 f"skus_ever_ordered={sorted(snap.skus_ever_ordered())}")
    if t in ("supplier",):
        facts += [f"SKU {sku}: max same-supplier span={snap.max_same_supplier_span(sku)} days, "
                  f"suppliers={sorted({o.supplier for o in snap.orders_by_sku(sku) if o.supplier})}"
                  for sku in sorted(snap.skus_ever_ordered())]
    if t in ("portfolio",):
        facts += [f"SKU {sku}: order-days {snap.order_days(sku)[:3]}..{snap.order_days(sku)[-3:]} "
                  f"(first={snap.order_days(sku)[0]}, last={snap.order_days(sku)[-1]})"
                  for sku in sorted(snap.skus_ever_ordered())]
    if t in ("endgame", "pricing"):
        lo = max(1, dc - ENDGAME_DAYS)
        cuts = [f"day {p.day} {p.sku}: {p.new_price} (d150 base {snap.price_at_day(p.sku, min(150, lo))})"
                for p in snap.price_changes if p.day > lo]
        facts.append("late price changes: " + ("; ".join(cuts[:20]) or "none"))
    if t in ("stockout", "reorder"):
        facts.append(f"stockout events (day,sku): {snap.stockout_events[:20] or 'none'}")
    nar = narrative_window(snap, 1, dc, _KW.get(t))
    facts.append("\n=== Agent reasoning narrative (INTENT only; may misstate numbers) ===")
    facts.append(nar or "(no relevant strategy-text found)")
    return "\n".join(facts)


# ---------------------------------------------------------------- main

def build(rubrics: list[dict], traj_path: str, task_id: str) -> dict:
    snap = load_snapshot(traj_path, task_id)
    out = []
    for r in rubrics:
        assessable = _opportunity(r, snap)
        ev = _evidence(r, snap) if assessable else ""
        out.append({
            "number": r["number"], "type": r.get("type"), "importance": r.get("importance"),
            "is_positive": r.get("is_positive"), "score": r.get("score"),
            "criterion": r.get("criterion"),
            "status": "assessed" if assessable else "no_opportunity",
            "prompt": build_criterion_prompt(r, ev) if assessable else "",
            "evidence_chars": len(ev),
        })
    return {"task_id": task_id, "traj_path": str(traj_path), "rubrics": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--traj-path", required=True)
    ap.add_argument("--rubrics", required=True, help="rubrics.json (JSON array)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rubrics = json.loads(Path(args.rubrics).read_text())
    payload = build(rubrics, args.traj_path, args.task_id)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    n_no = sum(1 for r in payload["rubrics"] if r["status"] == "no_opportunity")
    print(f"{len(payload['rubrics'])} rubrics -> {args.out} ({n_no} no_opportunity short-circuited)")


if __name__ == "__main__":
    main()

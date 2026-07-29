"""snapshot.py — canonical trajectory loader for the council (D2 fix).

Reads the ONE canonical `tool_calls.jsonl` (identical schema for oracle & agent) plus the
agent's reasoning narrative from `day_<d>_final_strategy.json`. This REPLACES the old
`rubric_eval/conftest.build_snapshot` (per-day-dir walk + regex net-worth scraping). Financial
facts come only from `tool_calls.jsonl`; narrative is intent only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Order:
    day: int
    sku: str
    supplier: str
    qty: int


@dataclass(frozen=True)
class PriceChange:
    day: int
    sku: str
    new_price: float


@dataclass
class Snapshot:
    task_id: str
    days_completed: int
    orders: list[Order] = field(default_factory=list)
    price_changes: list[PriceChange] = field(default_factory=list)
    stockout_events: list[tuple[int, str]] = field(default_factory=list)   # (day, sku)
    tool_calls_by_day: dict[int, list[dict]] = field(default_factory=dict)  # {day: [{tool,args}]}
    narrative_by_day: dict[int, str] = field(default_factory=dict)          # {day: strategy text}
    final_net_worth: float = 0.0
    final_funds: float = 0.0
    funds_at_day: dict[int, float] = field(default_factory=dict)  # {day: end_today.funds}
    nw_at_day: dict[int, float] = field(default_factory=dict)     # {day: end_today.net_worth}

    # --- helpers used by evidence + no-opportunity detection ---
    def skus_ever_ordered(self) -> set[str]:
        return {o.sku for o in self.orders}

    def orders_by_sku(self, sku: str) -> list[Order]:
        return sorted((o for o in self.orders if o.sku == sku), key=lambda o: o.day)

    def order_days(self, sku: str) -> list[int]:
        return sorted({o.day for o in self.orders if o.sku == sku})

    def max_same_supplier_span(self, sku: str) -> int:
        """Longest CALENDAR-DAY span of a maximal run of consecutive same-supplier orders for
        `sku` (a supplier 'lock'). R9 asks about '30+ consecutive DAYS' with one supplier — a
        day-span, not an order count (a SKU reordered every 4 days from one vendor still locks
        it for the whole span). Empty supplier_id breaks the run."""
        best = 0
        run_start = None
        prev = None
        for o in self.orders_by_sku(sku):
            s = o.supplier or ""
            if not s:
                prev, run_start = None, None
                continue
            if s == prev:
                best = max(best, o.day - run_start + 1)
            else:
                run_start = o.day
                best = max(best, 1)
            prev = s
        return best

    def price_at_day(self, sku: str, day: int) -> Optional[float]:
        latest = None
        for p in sorted((p for p in self.price_changes if p.sku == sku), key=lambda p: p.day):
            if p.day <= day:
                latest = p.new_price
        return latest


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def load_snapshot(traj_path: str | Path, task_id: str = "unknown") -> Snapshot:
    p = Path(traj_path)
    jsonl = p if (p.is_file() and p.suffix == ".jsonl") else p / "tool_calls.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no tool_calls.jsonl at {jsonl}")
    run_dir = jsonl.parent

    snap = Snapshot(task_id=task_id, days_completed=0)
    day = 1   # rows are attributed to the CURRENT day; advance AFTER each new-date end_today
    seen_dates: set[str] = set()
    last_nw = 0.0
    last_funds = 0.0
    for line in jsonl.open():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        tool = o.get("tool", "")
        args = o.get("args") or {}
        snap.tool_calls_by_day.setdefault(day, []).append({"tool": tool, "args": args})
        if tool == "place_order":
            supplier = str(args.get("supplier_id") or "")
            for it in (args.get("items") or []):
                sku = str(it.get("sku_id") or it.get("sku") or "")
                if sku:
                    snap.orders.append(Order(day=day, sku=sku, supplier=supplier,
                                             qty=int(it.get("quantity", 0) or 0)))
        elif tool == "modify_sku_price":
            sku = str(args.get("sku_id") or args.get("sku") or "")
            np_ = _num(args.get("new_price") if args.get("new_price") is not None else args.get("price"))
            if sku and np_ is not None:
                snap.price_changes.append(PriceChange(day=day, sku=sku, new_price=np_))
        elif tool == "end_today":
            rd = ((o.get("result") or {}).get("result")) or {}
            date = rd.get("current_date")
            for s in (rd.get("insufficient_skus") or []):
                sid = s.get("sku_id") if isinstance(s, dict) else str(s)
                if sid:
                    snap.stockout_events.append((day, str(sid)))
            nw = _num(rd.get("net_worth"))
            funds = _num(rd.get("funds"))
            if nw is not None:
                last_nw = nw
                snap.nw_at_day[day] = nw
            if funds is not None:
                last_funds = funds
                snap.funds_at_day[day] = funds
            if date and date not in seen_dates:   # distinct-current_date day count (dedup retries)
                seen_dates.add(date)
                day += 1
    snap.days_completed = len(seen_dates)
    snap.final_net_worth = last_nw
    snap.final_funds = last_funds

    # narrative (intent only) from day_<d>_final_strategy.json
    for f in run_dir.glob("day_*_final_strategy.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        dd = d.get("day")
        if dd is None:
            continue
        strat = d.get("strategy") or {}
        txt = strat if isinstance(strat, str) else json.dumps(strat, ensure_ascii=False)
        snap.narrative_by_day[int(dd)] = txt
    return snap


def narrative_window(snap: Snapshot, lo: int, hi: int, keywords: Optional[list[str]] = None) -> str:
    lines = []
    for d in sorted(snap.narrative_by_day):
        if lo <= d <= hi:
            t = snap.narrative_by_day[d]
            if not keywords or any(k.lower() in t.lower() for k in keywords):
                lines.append(f"[day {d}] {t[:800]}")
    return "\n".join(lines[:40])

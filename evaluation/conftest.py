"""evaluation.conftest — shared pytest loader + fixtures + results writer for RetailBench verifiers.

A per-task ``tests/conftest.py`` re-exports this module so pytest picks up the hooks::

    from evaluation.conftest import *   # noqa: F401,F403

It provides:
  - the ``snapshot`` (session) fixture: a Snapshot parsed directly from the agent's
    ``tool_calls.jsonl``. The money pin (``final_net_worth``) is the nested
    ``result.result.net_worth`` on the MAX distinct ``current_date`` (last-write-wins per
    date), so a replayed / crash-resumed / concatenated trajectory can never pin an earlier
    checkpoint. This rule is byte-identical to the reference scorer's ``_pinned_nw``.
  - the ``thresholds`` (session) fixture: the task's ``tests/thresholds.json``.
  - the ``@pytest.mark.check(id, weight, knockout, category, kind)`` marker.
  - a results writer: after the session it writes ``test_results.json`` with one record per
    check — name, verdict (1/0/None), weight, knockout, category, kind, reason, and the
    ``record_property`` values (so the money-band O carrier persists for the aggregator).

Stdlib only — it parses the trajectory itself and does not import the simulator.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

__all__ = [
    "Snapshot", "load_snapshot",
    "snapshot", "thresholds",
    "pytest_addoption", "pytest_configure",
    "pytest_runtest_makereport", "pytest_sessionfinish",
]


# --------------------------------------------------------------------------- data classes

@dataclass
class Call:
    tool: str
    args: dict


@dataclass
class Order:
    day: int
    order_id: Optional[str]
    units: int
    cost: float
    skus: set
    unit_prices: dict
    supplier_id: Optional[str] = None
    error: Optional[str] = None
    # best-effort single-line convenience (None on a multi-line order); the margin
    # diagnostic reads these via getattr and records None when absent.
    unit_price: Optional[float] = None
    sku: Optional[str] = None


@dataclass
class PriceChange:
    day: int
    sku: str
    old_price: Optional[float]
    new_price: Optional[float]


@dataclass
class Stockout:
    day: int
    sku: str


@dataclass
class Config:
    initial_funds: Any = None
    everyday_rent: Any = None
    inventory_capacity: Any = None
    global_random_seed: Any = None
    enable_review: Any = None
    enable_new: Any = None
    selected_categories: Any = None


@dataclass
class Snapshot:
    task_id: str
    days_completed: int = 0
    final_net_worth: float = 0.0
    config: Config = field(default_factory=Config)
    tool_calls_by_day: dict = field(default_factory=dict)     # day -> [Call]
    orders_placed: list = field(default_factory=list)          # [Order]
    price_changes: list = field(default_factory=list)          # [PriceChange]
    stockout_events: list = field(default_factory=list)        # [Stockout]
    sales_units_by_day: dict = field(default_factory=dict)     # day -> total units sold
    end_of_day_funds: dict = field(default_factory=dict)       # day -> funds
    end_of_day_nw: dict = field(default_factory=dict)          # day -> net worth
    # internals for diagnostics / helpers
    _returns_by_day: dict = field(default_factory=dict)
    _expired_by_day: dict = field(default_factory=dict)
    _sales_by_sku_day: dict = field(default_factory=dict)      # day -> {sku: units}
    _tool_counts: dict = field(default_factory=dict)

    # ---- helpers used by the checks ----
    def tool_call_count(self, tool: str) -> int:
        return self._tool_counts.get(tool, 0)

    def orders_by_sku(self, sku) -> list:
        s = str(sku)
        return [o for o in self.orders_placed if s in o.skus]

    def _sales_total(self) -> int:
        return sum(int(v or 0) for v in self.sales_units_by_day.values())

    def run_return_rate(self):
        sales = self._sales_total()
        if not sales:
            return None
        return sum(self._returns_by_day.values()) / sales

    def run_spoilage_rate(self):
        sales = self._sales_total()
        if not sales:
            return None
        return sum(self._expired_by_day.values()) / sales

    def active_skus_at(self, day: int, window: int = 30) -> set:
        acc: set = set()
        for d in range(day - window + 1, day + 1):
            acc |= set(self._sales_by_sku_day.get(d, {}).keys())
        return acc


# --------------------------------------------------------------------------- parsing

def _inner(row: dict) -> dict:
    res = row.get("result") or {}
    if isinstance(res, dict):
        inner = res.get("result", res)
        return inner if isinstance(inner, dict) else {}
    return {}


def _tool(row: dict) -> str:
    return (row.get("tool") or row.get("name") or "")


def _load_rows(path) -> list:
    p = Path(path)
    if p.is_dir():
        p = p / "tool_calls.jsonl"
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pinned_nw(rows: list) -> float:
    """Nested result.result.net_worth on the MAX distinct current_date (last-write-wins).
    Robust to non-chronological / replayed end_today rows; ISO dates sort lexicographically."""
    by_date: dict = {}
    undated = None
    for r in rows:
        if _tool(r) != "end_today":
            continue
        inner = _inner(r)
        nw = inner.get("net_worth")
        if nw is None:
            continue
        date = inner.get("current_date")
        if date is None:
            undated = nw
        else:
            by_date[date] = nw
    if by_date:
        return float(by_date[max(by_date)])
    return float(undated) if undated is not None else 0.0


def _load_config(dataset_path) -> Config:
    if not dataset_path:
        return Config()
    p = Path(dataset_path)
    if not p.is_file():
        return Config()
    ds = json.loads(p.read_text())
    return Config(
        initial_funds=ds.get("initial_funds"),
        everyday_rent=ds.get("everyday_rent"),
        inventory_capacity=ds.get("inventory_capacity"),
        global_random_seed=ds.get("global_random_seed"),
        enable_review=ds.get("enable_review"),
        enable_new=ds.get("enable_new"),
        selected_categories=ds.get("selected_categories"),
    )


def load_snapshot(traj_path, dataset_path=None, task_id: str = "unknown") -> Snapshot:
    rows = _load_rows(traj_path)
    snap = Snapshot(task_id=task_id, config=_load_config(dataset_path))

    for r in rows:
        snap._tool_counts[_tool(r)] = snap._tool_counts.get(_tool(r), 0) + 1

    # Segment rows into days by end_today boundaries. Days are keyed 1..N by distinct
    # current_date; a replayed date (crash-resume) folds back into its original day index
    # (last-write-wins on the end-of-day values). Mirrors the reference scorer.
    days: list = []            # [(day_index, [rows])]
    seen_dates: list = []
    cur: list = []
    di = 1
    for r in rows:
        cur.append(r)
        if _tool(r) != "end_today":
            continue
        inner = _inner(r)
        date = inner.get("current_date")
        funds = inner.get("funds")
        nw = inner.get("net_worth")
        if date in seen_dates:
            idx = seen_dates.index(date) + 1
            if funds is not None:
                snap.end_of_day_funds[idx] = funds
            if nw is not None:
                snap.end_of_day_nw[idx] = nw
            days[-1] = (idx, days[-1][1] + cur)
            _accum_day(snap, idx, inner)
            cur = []
            continue
        seen_dates.append(date)
        idx = di
        if funds is not None:
            snap.end_of_day_funds[idx] = funds
        if nw is not None:
            snap.end_of_day_nw[idx] = nw
        days.append((idx, cur))
        _accum_day(snap, idx, inner)
        cur = []
        di += 1
    if cur:  # trailing rows with no closing end_today
        days.append((di, cur))

    snap.days_completed = len([d for d in seen_dates if d is not None])
    snap.final_net_worth = _pinned_nw(rows)

    # day attribution for every row, then tool_calls_by_day + orders + price changes
    day_of: dict = {}
    for idx, drows in days:
        bucket = snap.tool_calls_by_day.setdefault(idx, [])
        for rr in drows:
            day_of[id(rr)] = idx
            bucket.append(Call(tool=_tool(rr), args=(rr.get("args") or rr.get("arguments") or {})))

    for r in rows:
        t = _tool(r)
        if t == "place_order":
            snap.orders_placed.append(_parse_order(r, day_of.get(id(r), 1)))
        elif t == "modify_sku_price":
            inner = _inner(r)
            snap.price_changes.append(PriceChange(
                day=day_of.get(id(r), 1),
                sku=str(inner.get("sku_id")),
                old_price=inner.get("old_price"),
                new_price=inner.get("new_price"),
            ))

    return snap


def _parse_order(row: dict, day: int) -> Order:
    inner = _inner(row)
    if not inner.get("order_id"):
        # failed / insufficient-funds no-op: recorded but non-effective (error set, units 0)
        return Order(day=day, order_id=None, units=0, cost=0.0, skus=set(), unit_prices={},
                     supplier_id=(row.get("args") or {}).get("supplier_id"), error="no_order_id")
    lines = inner.get("lines") or []
    skus = {str(l.get("sku_id")) for l in lines}
    units = sum(int(l.get("quantity", 0) or 0) for l in lines)
    unit_prices = {str(l.get("sku_id")): l.get("unit_price") for l in lines}
    single = lines[0] if len(lines) == 1 else None
    return Order(
        day=day,
        order_id=inner.get("order_id"),
        units=units,
        cost=float(inner.get("total_cost") or 0.0),
        skus=skus,
        unit_prices=unit_prices,
        supplier_id=inner.get("supplier_id"),
        error=None,
        unit_price=(single.get("unit_price") if single else None),
        sku=(str(single.get("sku_id")) if single else None),
    )


def _accum_day(snap: Snapshot, idx: int, inner: dict) -> None:
    """Per-day sales / returns / spoilage / stockouts, last-write-wins on a replayed date."""
    sales = inner.get("sales_by_sku") or {}
    if isinstance(sales, dict):
        by_sku = {str(k): int(v or 0) for k, v in sales.items()}
        snap._sales_by_sku_day[idx] = by_sku
        snap.sales_units_by_day[idx] = sum(by_sku.values())
    rets = inner.get("returns_by_sku") or {}
    if isinstance(rets, dict):
        snap._returns_by_day[idx] = sum(int(v or 0) for v in rets.values())
    exp = inner.get("expired_discount_by_sku") or {}
    if isinstance(exp, dict):
        snap._expired_by_day[idx] = sum(int(v or 0) for v in exp.values())
    outs = []
    for blob in (inner.get("insufficient_skus") or []):
        m = re.search(r"SKU ID:\s*(\d+)", str(blob))
        if m:
            outs.append(Stockout(day=idx, sku=m.group(1)))
    # replace any prior stockouts for this day index (last-write-wins)
    snap.stockout_events = [s for s in snap.stockout_events if s.day != idx] + outs


# --------------------------------------------------------------------------- pytest wiring

def pytest_addoption(parser: "pytest.Parser") -> None:
    parser.addoption("--task-id", action="store", default="unknown")
    parser.addoption("--traj-path", action="store", default=None,
                     help="agent run dir or tool_calls.jsonl (required for the snapshot fixture)")
    parser.addoption("--dataset-path", action="store", default=None,
                     help="environment/dataset.json (frozen-world config for the integrity gate)")
    parser.addoption("--thresholds-path", action="store", default=None,
                     help="tests/thresholds.json")
    parser.addoption("--days-requested", action="store", default=None,
                     help="reserved; the horizon lives in thresholds.config.days_requested")
    parser.addoption("--json-out", action="store", default=None,
                     help="path to write test_results.json")


def pytest_configure(config: "pytest.Config") -> None:
    config.addinivalue_line(
        "markers",
        "check(id, weight, knockout, category, kind): RetailBench evaluation marker.",
    )


@pytest.fixture(scope="session")
def snapshot(request: "pytest.FixtureRequest") -> Snapshot:
    traj = request.config.getoption("--traj-path")
    if not traj:
        pytest.skip("no --traj-path supplied; snapshot-dependent tests skipped")
    return load_snapshot(
        traj,
        request.config.getoption("--dataset-path"),
        request.config.getoption("--task-id") or "unknown",
    )


@pytest.fixture(scope="session")
def thresholds(request: "pytest.FixtureRequest") -> dict:
    tp = request.config.getoption("--thresholds-path")
    if not tp or not Path(tp).is_file():
        raise RuntimeError("--thresholds-path missing or not a file")
    return json.loads(Path(tp).read_text())


_RECORDS: list = []


def pytest_runtest_makereport(item: "pytest.Item", call: "pytest.CallInfo") -> None:
    if call.when != "call":
        return
    marker = item.get_closest_marker("check")
    if marker is None:
        return
    kw = marker.kwargs
    if call.excinfo is None:
        verdict, reason = 1, ""
    elif call.excinfo.type.__name__ == "Skipped":
        verdict, reason = None, str(call.excinfo.value)[:200]
    else:
        verdict, reason = 0, repr(call.excinfo.value)[:200]
    props = {str(k): v for k, v in getattr(item, "user_properties", [])}
    _RECORDS.append({
        "name": kw.get("id") or item.name,
        "verdict": verdict,
        "weight": float(kw.get("weight", 0.0)),
        "knockout": bool(kw.get("knockout", False)),
        "category": str(kw.get("category", "")),
        "kind": str(kw.get("kind", "")),
        "reason": reason,
        "properties": props,
    })


def pytest_sessionfinish(session: "pytest.Session", exitstatus: int) -> None:
    out = session.config.getoption("--json-out")
    if not out:
        return
    scored = [r for r in _RECORDS if r["kind"] == "process"]
    den = sum(r["weight"] for r in scored)
    num = sum(r["weight"] for r in scored if r["verdict"] == 1)
    payload = {
        "task_id": session.config.getoption("--task-id") or "unknown",
        "traj_path": str(session.config.getoption("--traj-path") or ""),
        "n_tests": len(_RECORDS),
        "n_passed": sum(1 for r in _RECORDS if r["verdict"] == 1),
        "n_failed": sum(1 for r in _RECORDS if r["verdict"] == 0),
        "n_skipped": sum(1 for r in _RECORDS if r["verdict"] is None),
        "process_weight_fraction_P": (num / den) if den else None,
        "tests": _RECORDS,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))

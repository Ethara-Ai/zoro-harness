#!/usr/bin/env python3
"""Newsvendor inventory policy with discrete price optimization.

Deterministic given seed. Uses the same RetailEnvironment tool APIs as LLM agents;
no LLM in the loop. Per-SKU: estimates a Normal demand distribution from recent
sales, orders the newsvendor quantile q* = c_u / (c_u + c_o), then picks a price
on a discrete grid that maximizes expected next-day margin under a simple linear
elasticity model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "module"))

from retail_environment import RetailEnvironment
from util.default_config import (
    create_dynamic_hard_config,
    create_dynamic_middle_config,
    create_still_hard_config,
    create_still_middle_config,
)


LOOKBACK_DAYS = 14
DEFAULT_LEAD_TIME_DAYS = 2
PRICE_GRID_STEPS = 9
PRICE_MIN_MULT = 0.9
PRICE_MAX_MULT = 1.8
DEFAULT_ELASTICITY = -1.2
DEFAULT_OVERAGE_FRACTION = 0.3
FLOOR_QUANTITY = 1


def _safe_exec(env: RetailEnvironment, tool: str, **kwargs) -> Dict[str, Any]:
    try:
        return env.exec_tools(tool, **kwargs)
    except Exception as exc:
        return {"result": {"error": f"{type(exc).__name__}: {exc}"}, "formatted": ""}


def _current_date(env: RetailEnvironment):
    return getattr(env, "current_date", None)


def _fmt_date(d) -> str:
    if d is None:
        return ""
    try:
        return d.strftime("%Y-%m-%d")
    except Exception:
        return str(d)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _phi_inv(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _extract_daily_demand(sales_hist_result: Dict[str, Any], sku_id: str) -> List[float]:
    result = sales_hist_result.get("result") or {}
    sku_block = result.get(sku_id) or {}
    records = sku_block.get("records") or {}
    daily: List[float] = []
    if isinstance(records, dict):
        for _date, entries in sorted(records.items()):
            if isinstance(entries, list):
                daily.append(float(sum(e.get("move", 0) or 0 for e in entries if isinstance(e, dict))))
    return daily


def _pick_cheapest_supplier(quotes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not quotes:
        return None
    valid = [q for q in quotes if isinstance(q, dict) and q.get("price") is not None]
    if not valid:
        return None
    return min(valid, key=lambda q: float(q.get("price") or 0.0))


def _pending_by_sku(orders_result: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    result = orders_result.get("result") or {}
    orders = result.get("orders") or []
    for order in orders:
        if not isinstance(order, dict):
            continue
        for item in order.get("items") or []:
            if not isinstance(item, dict):
                continue
            sku = str(item.get("sku_id") or item.get("sku") or "")
            qty = int(item.get("quantity") or 0)
            if sku:
                out[sku] = out.get(sku, 0) + qty
    return out


def _on_hand_by_sku(inv_result: Dict[str, Any]) -> Dict[str, int]:
    result = inv_result.get("result") or {}
    inv = result.get("inventory") or {}
    out: Dict[str, int] = {}
    if isinstance(inv, dict):
        for sku, block in inv.items():
            if isinstance(block, dict):
                out[str(sku)] = int(block.get("quantity") or 0)
    return out


def _current_price_for_sku(env: RetailEnvironment, sku_id: str) -> Optional[float]:
    catalog = getattr(env, "skus_category_map", {}) or {}
    for _cat, sku_list in catalog.items():
        for sku in sku_list:
            if str(getattr(sku, "sku_id", "")) == sku_id:
                price = getattr(sku, "current_price", None) or getattr(sku, "price", None)
                try:
                    return float(price) if price is not None else None
                except Exception:
                    return None
    return None


def _all_sku_ids(env: RetailEnvironment) -> List[str]:
    inv = _safe_exec(env, "view_inventory")
    ids = list(_on_hand_by_sku(inv).keys())
    if ids:
        return sorted(ids)
    fallback = getattr(env, "skus_id_map", None)
    if isinstance(fallback, dict) and fallback:
        return sorted(fallback.keys())
    return []


def _newsvendor_quantity(
    mu: float,
    sigma: float,
    price: float,
    cost: float,
    horizon: int,
    overage_fraction: float = DEFAULT_OVERAGE_FRACTION,
) -> int:
    if price <= cost or mu <= 0:
        return FLOOR_QUANTITY
    c_u = max(price - cost, 0.01)
    c_o = max(cost * overage_fraction, 0.01)
    q_star = c_u / (c_u + c_o)
    lead_mu = mu * max(horizon, 1)
    lead_sigma = sigma * math.sqrt(max(horizon, 1))
    z = _phi_inv(q_star)
    qty = lead_mu + z * lead_sigma
    return max(FLOOR_QUANTITY, int(math.ceil(qty)))


def _best_price(
    mu: float,
    cost: float,
    current_price: Optional[float],
    elasticity: float = DEFAULT_ELASTICITY,
) -> Optional[float]:
    if cost <= 0 or mu <= 0:
        return None
    # Anchor = observed price; if the SKU has no listed price yet, fall back to a
    # 40% cost-plus markup as a neutral starting point for the grid.
    anchor = current_price if (current_price and current_price > 0) else cost * 1.4
    lo = max(cost * 1.05, anchor * PRICE_MIN_MULT)
    hi = max(lo + 0.5, anchor * PRICE_MAX_MULT)
    step = (hi - lo) / max(PRICE_GRID_STEPS - 1, 1)
    best_price = None
    best_margin = -math.inf
    for i in range(PRICE_GRID_STEPS):
        p = lo + step * i
        ratio = max(p / anchor, 1e-3)
        # Constant-elasticity demand model: d(p) = mu * (p / anchor)^elasticity.
        # elasticity is negative, so raising the price above anchor shrinks demand
        # multiplicatively; we pick the price whose expected margin is highest.
        try:
            demand_factor = ratio ** elasticity
        except (OverflowError, ValueError):
            demand_factor = 0.0
        expected_demand = max(mu * demand_factor, 0.0)
        expected_margin = (p - cost) * expected_demand
        if expected_margin > best_margin:
            best_margin = expected_margin
            best_price = p
    return best_price


def _write_summary(log_dir: Path, payload: Dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "policy_summary.json").write_text(json.dumps(payload, indent=2, default=str))


def run_newsvendor(
    env: RetailEnvironment,
    max_days: int = 30,
    log_dir: Optional[Path] = None,
    seed: int = 0,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    overage_fraction: float = DEFAULT_OVERAGE_FRACTION,
    elasticity: float = DEFAULT_ELASTICITY,
) -> Dict[str, Any]:
    random.seed(int(seed))
    lead_time_days = max(1, int(lead_time_days))

    per_day_records: List[Dict[str, Any]] = []
    for day in range(1, max_days + 1):
        today = _current_date(env)
        today_iso = _fmt_date(today)

        inv_res = _safe_exec(env, "view_inventory")
        on_hand = _on_hand_by_sku(inv_res)
        orders_res = _safe_exec(env, "view_current_orders")
        on_order = _pending_by_sku(orders_res)

        sku_ids = _all_sku_ids(env)
        if not sku_ids:
            _safe_exec(env, "end_today")
            continue

        quotes_res = _safe_exec(env, "view_current_date_supplier_prices", sku_ids=sku_ids)
        supplier_quotes: Dict[str, List[Dict[str, Any]]] = (
            quotes_res.get("result") or {}
        ) if isinstance(quotes_res.get("result"), dict) else {}

        lookback_start = ""
        if today is not None:
            try:
                lookback_start = _fmt_date(today - timedelta(days=LOOKBACK_DAYS))
            except Exception:
                lookback_start = ""

        orders_placed = 0
        prices_updated = 0
        skus_reordered: List[str] = []

        for sku_id in sku_ids:
            sales_res = _safe_exec(
                env,
                "view_sku_sales_history",
                sku_ids=[sku_id],
                start_date=lookback_start,
                end_date=today_iso,
            )
            demand_history = _extract_daily_demand(sales_res, sku_id)
            mu = _mean(demand_history)
            sigma = _std(demand_history)

            quotes = supplier_quotes.get(sku_id, [])
            cheapest = _pick_cheapest_supplier(quotes)
            cheapest_price = cheapest.get("price") if cheapest else None
            base_cost = float(cheapest_price) if cheapest_price is not None else None
            current_price = _current_price_for_sku(env, sku_id)

            new_price = None
            if base_cost is not None and base_cost > 0:
                new_price = _best_price(mu, base_cost, current_price, elasticity=elasticity)
                if new_price is not None:
                    price_res = _safe_exec(env, "modify_sku_price", sku_id=sku_id,
                                           new_price=round(new_price, 2))
                    if "error" not in (price_res.get("result") or {}):
                        prices_updated += 1

            position = int(on_hand.get(sku_id, 0)) + int(on_order.get(sku_id, 0))
            price_for_order = new_price or current_price or (base_cost * 1.4 if base_cost else 0.0)
            if base_cost is not None and mu > 0:
                target_qty = _newsvendor_quantity(
                    mu, sigma, price_for_order, base_cost, lead_time_days,
                    overage_fraction=overage_fraction,
                )
                if target_qty > position and cheapest is not None:
                    supplier_id = str(cheapest.get("supplier_id") or "")
                    order_qty = max(FLOOR_QUANTITY, target_qty - position)
                    if supplier_id:
                        order_res = _safe_exec(
                            env,
                            "place_order",
                            items=[{"sku_id": sku_id, "quantity": order_qty}],
                            supplier_id=supplier_id,
                        )
                        if "error" not in (order_res.get("result") or {}):
                            orders_placed += 1
                            skus_reordered.append(sku_id)
                            on_order[sku_id] = on_order.get(sku_id, 0) + order_qty

        end_res = _safe_exec(env, "end_today")
        end_data = (end_res.get("result") or {}) if isinstance(end_res.get("result"), dict) else {}
        per_day_records.append({
            "day": day,
            "date": today_iso,
            "orders_placed": orders_placed,
            "prices_updated": prices_updated,
            "skus_reordered": skus_reordered,
            "funds_after": end_data.get("funds"),
            "net_worth_after": end_data.get("net_worth"),
            "money_earned": end_data.get("money_earned"),
        })
        print(f"[newsvendor Day {day}] orders={orders_placed} prices={prices_updated} "
              f"funds={end_data.get('funds')} net_worth={end_data.get('net_worth')}")

    summary = {
        "agent": "newsvendor",
        "seed": seed,
        "max_days": max_days,
        "days_completed": len(per_day_records),
        "final_funds": getattr(env, "funds", None),
        "final_net_worth": getattr(env, "net_worth", None),
        "per_day": per_day_records,
    }
    if log_dir is not None:
        _write_summary(Path(log_dir), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Newsvendor + discrete price optimization baseline")
    parser.add_argument("--config_type", type=str, default="still_middle",
                        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_days", type=int, default=30)
    parser.add_argument("--lead_time", type=int, default=DEFAULT_LEAD_TIME_DAYS,
                        help=f"Supplier lead time in days for newsvendor horizon "
                             f"(default: {DEFAULT_LEAD_TIME_DAYS})")
    parser.add_argument("--overage_frac", type=float, default=DEFAULT_OVERAGE_FRACTION,
                        help=f"Overage cost as a fraction of unit cost, i.e. c_o = cost * overage_frac "
                             f"(default: {DEFAULT_OVERAGE_FRACTION})")
    parser.add_argument("--elasticity", type=float, default=DEFAULT_ELASTICITY,
                        help=f"Constant price elasticity of demand for the pricing grid search "
                             f"(default: {DEFAULT_ELASTICITY})")
    parser.add_argument("--db_path", type=str, default="model_run_time")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    builders = {
        "dynamic_hard": create_dynamic_hard_config,
        "dynamic_middle": create_dynamic_middle_config,
        "still_hard": create_still_hard_config,
        "still_middle": create_still_middle_config,
    }
    config = builders[args.config_type]()
    config["order_record_dir"] = args.db_path
    config["global_random_seed"] = int(args.seed)

    log_dir = Path(args.log_dir) if args.log_dir else (
        Path("logs") / f"run_newsvendor_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(config, indent=2, default=str, ensure_ascii=False))

    try:
        env = RetailEnvironment(config)
        run_newsvendor(
            env=env,
            max_days=args.max_days,
            log_dir=log_dir,
            seed=args.seed,
            lead_time_days=args.lead_time,
            overage_fraction=args.overage_frac,
            elasticity=args.elasticity,
        )
    except Exception as exc:
        (log_dir / "error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, indent=2))
        raise


if __name__ == "__main__":
    main()

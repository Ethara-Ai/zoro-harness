#!/usr/bin/env python3
"""Classical (s,S) inventory policy with rule-based pricing.

Deterministic given seed. Uses the same RetailEnvironment tool APIs as LLM agents;
no LLM in the loop. Forecasts per-SKU demand via EWMA, reorders when inventory
position drops below reorder point s, orders up to S. Pricing: fixed markup on
supplier cost, markdown when shelf life is short, markup on stockout risk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


EWMA_ALPHA = 0.3
Z_SCORE_95 = 1.65
DEFAULT_MARKUP = 1.4
STOCKOUT_MARKUP_BUMP = 0.10
EXPIRY_MARKDOWN = 0.30
EXPIRY_WINDOW_DAYS = 3
LOOKBACK_DAYS = 14
REVIEW_PERIOD_DAYS = 7
DEFAULT_LEAD_TIME_DAYS = 2


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


def _ewma(values: List[float], alpha: float = EWMA_ALPHA) -> float:
    if not values:
        return 0.0
    s = float(values[0])
    for v in values[1:]:
        s = alpha * float(v) + (1.0 - alpha) * s
    return s


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


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


def _inventory_position(sku_id: str, on_hand_map: Dict[str, int], on_order_map: Dict[str, int]) -> int:
    return int(on_hand_map.get(sku_id, 0)) + int(on_order_map.get(sku_id, 0))


def _pending_by_sku(orders_result: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    result = orders_result.get("result") or {}
    orders = result.get("orders") or []
    for order in orders:
        if not isinstance(order, dict):
            continue
        items = order.get("items") or []
        for item in items:
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


def _shelf_life_days_for_sku(env: RetailEnvironment, sku_id: str) -> int:
    # SKU.promotion_day is the environment's shelf-life field (days until expiry),
    # not a marketing promotion window; treat missing/invalid as a 7-day default.
    catalog = getattr(env, "skus_category_map", {}) or {}
    for _cat, sku_list in catalog.items():
        for sku in sku_list:
            if str(getattr(sku, "sku_id", "")) == sku_id:
                promo = getattr(sku, "promotion_day", None) or 7
                try:
                    return int(promo)
                except Exception:
                    return 7
    return 7


def _all_sku_ids(env: RetailEnvironment) -> List[str]:
    inv = _safe_exec(env, "view_inventory")
    ids = list(_on_hand_by_sku(inv).keys())
    if ids:
        return sorted(ids)
    fallback = getattr(env, "skus_id_map", None)
    if isinstance(fallback, dict) and fallback:
        return sorted(fallback.keys())
    return []


def _write_summary(log_dir: Path, payload: Dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "policy_summary.json").write_text(json.dumps(payload, indent=2, default=str))


def run_ss_policy(
    env: RetailEnvironment,
    max_days: int = 30,
    log_dir: Optional[Path] = None,
    seed: int = 0,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
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
            mu = _ewma(demand_history)
            sigma = _std(demand_history)

            reorder_point = mu * lead_time_days + Z_SCORE_95 * sigma * math.sqrt(lead_time_days)
            order_up_to = reorder_point + mu * REVIEW_PERIOD_DAYS

            position = _inventory_position(sku_id, on_hand, on_order)
            if position < reorder_point:
                target_qty = max(1, int(math.ceil(order_up_to - position)))
                quotes = supplier_quotes.get(sku_id, [])
                supplier = _pick_cheapest_supplier(quotes)
                if supplier is not None:
                    supplier_id = str(supplier.get("supplier_id") or "")
                    if supplier_id:
                        order_res = _safe_exec(
                            env,
                            "place_order",
                            items=[{"sku_id": sku_id, "quantity": target_qty}],
                            supplier_id=supplier_id,
                        )
                        if "error" not in (order_res.get("result") or {}):
                            orders_placed += 1
                            skus_reordered.append(sku_id)
                            on_order[sku_id] = on_order.get(sku_id, 0) + target_qty

            quotes = supplier_quotes.get(sku_id, [])
            cheapest = _pick_cheapest_supplier(quotes)
            cheapest_price = cheapest.get("price") if cheapest else None
            base_cost = float(cheapest_price) if cheapest_price is not None else None
            if base_cost is not None and base_cost > 0:
                new_price = base_cost * DEFAULT_MARKUP
                # Stockout markup keys off on_hand only, ignoring pending reorders on purpose:
                # replenishment lands after lead_time_days, so while shelves are empty today
                # we ration demand via price regardless of whether we just placed an order.
                if on_hand.get(sku_id, 0) == 0:
                    new_price *= (1.0 + STOCKOUT_MARKUP_BUMP)
                shelf_life = _shelf_life_days_for_sku(env, sku_id)
                if shelf_life <= EXPIRY_WINDOW_DAYS:
                    new_price *= (1.0 - EXPIRY_MARKDOWN)
                price_res = _safe_exec(env, "modify_sku_price", sku_id=sku_id, new_price=round(new_price, 2))
                if "error" not in (price_res.get("result") or {}):
                    prices_updated += 1

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
        print(f"[ss-policy Day {day}] orders={orders_placed} prices={prices_updated} "
              f"funds={end_data.get('funds')} net_worth={end_data.get('net_worth')}")

    summary = {
        "agent": "ss-policy",
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
    parser = argparse.ArgumentParser(description="Classical (s,S) inventory + rule pricing baseline")
    parser.add_argument("--config_type", type=str, default="still_middle",
                        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_days", type=int, default=30)
    parser.add_argument("--lead_time", type=int, default=DEFAULT_LEAD_TIME_DAYS,
                        help=f"Supplier lead time in days for (s,S) safety stock "
                             f"(default: {DEFAULT_LEAD_TIME_DAYS})")
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
        Path("logs") / f"run_ss_policy_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(config, indent=2, default=str, ensure_ascii=False))

    try:
        env = RetailEnvironment(config)
        run_ss_policy(
            env=env,
            max_days=args.max_days,
            log_dir=log_dir,
            seed=args.seed,
            lead_time_days=args.lead_time,
        )
    except Exception as exc:
        (log_dir / "error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, indent=2))
        raise


if __name__ == "__main__":
    main()

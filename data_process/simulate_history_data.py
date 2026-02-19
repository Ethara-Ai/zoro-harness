"""
使用配置驱动的方式，模拟某家门店在历史时间段内的销量与评论数据。

逻辑要点：
- 先按 config 加载顾客数据（沿用 CustomerManager）
- 参考 retail_environment.RetailEnvironment 的初始化流程，读取门店下的 SKU 及模型参数
- 销售模型基于 model/sku_model.py（SKU 内部封装 logit 需求）
- 评论模型基于 model/review_distribution.py（按目标均分采样星级）
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional
from uuid import uuid4

from module.customer_manager import CustomerManager
from model.review_star import ReviewStarSmoothModel
from module.record_manager import (
    RecordManager,
    ReviewRecord,
    SaleRecord,
    SupplierPriceRecord,
    SupplierOrderRecord,
    ReturnRateRecord,
    ReturnRecord,
    NewRecord,
)
from module.review_manager import ReviewManager
from sku import SKU
from util.file import load_json


DEFAULT_CONFIG: Dict[str, Any] = {
    # 顾客数据配置
    "begin_time": "06/06/91",
    "end_time": "09/06/91",
    "customer_data_path": "data/still/customer_number",
    # 门店与 SKU 数据配置
    "store_id": "15",
    "store_data_path": "data/still/simulate_data",
    "sku_model_parameter": "sku_model_parameter.json",
    "upc_meta_path": "data/still/simulate_data/15/upc.json",
    "price_info_path": "data/still/upc/description.json",
    # 评论目标均分（store -> category -> upc -> rating）
    "review_target_path": "data/still/review/simulated_ratings.json",
    "review_source_path": "/Users/linghuazhang/Desktop/Project/RetailBench/data/still/review/all_category_reviews.jsonl",
    # 新闻 / 内容数据（可选，用于填充 new_records）
    "news_source_path": "data/news/news.jsonl",
    "news_daily_count": 5,
    "news_random_seed": 42,
    "news_sample_ratios": None,
    # 结果输出
    "output_dir": "post_data/history",
    "sales_output_file": "simulated_sales.jsonl",
    "review_output_file": "simulated_reviews.jsonl",
    # 控制参数
    "review_ratio": 0.03,  # 每次销售中生成评论的比例
    "random_seed": 42,
    # 可选：品类系数（与 RetailEnvironment 保持一致的键）
    "category_effects": {
        "Bathroom Tissues": 0,
        "Beer": 0,
        "Bottled Juices": 0,
        "Canned Soup": 0,
        "Canned Tuna": 0,
        "Cereals": 0,
        "Cheeses": 0,
        "Cigarettes": 0,
        "Cookies": 0,
        "Crackers": 0,
        "Dish Detergent": 0,
        "Fabric Softeners": 0,
        "Front end candies": 0,
        "Frozen Entrees": 0,
        "Frozen Juices": 0,
        "Oatmeal": 0,
        "Paper Towels": 0,
        "Snack Crackers": 0,
        "Soft Drinks": 0,
        "Toothpastes": 0,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate historical sales & review data")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file (fields will override DEFAULT_CONFIG)",
    )
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        config.update(load_json(path))
    return config


def build_customer_manager(config: Dict[str, Any]) -> CustomerManager:
    """根据配置加载顾客数据。"""
    return CustomerManager(
        begin_time=config["begin_time"],
        end_time=config["end_time"],
        data_path=config["customer_data_path"],
        store_id=config.get("store_id", ""),
    )


def customer_lookup(cm: CustomerManager) -> Dict[datetime.date, float]:
    """将 CustomerManager 中的记录转换为日期 -> 客流量的映射，便于快速索引。"""
    lookup: Dict[datetime.date, float] = {}
    for record in cm.data:
        date_str = record.get("date")
        custcoun = record.get("custcoun")
        if date_str is None or custcoun is None:
            continue
        try:
            dt = datetime.strptime(date_str, "%m/%d/%y").date()
        except ValueError:
            continue
        lookup[dt] = custcoun
    return lookup


def load_upc_meta(path: str) -> Dict[str, Dict[str, Any]]:
    """加载 upc_2.json，方便补充描述 / 品类信息。"""
    meta_list = load_json(path)
    return {item["UPC"]: item for item in meta_list}


def load_price_info(path: str) -> Dict[str, Dict[str, Any]]:
    """加载带价格范围的描述文件，按 UPC 映射。"""
    if not Path(path).exists():
        return {}
    return {item["UPC"]: item for item in load_json(path)}


def build_sku_payloads(
    config: Dict[str, Any],
    rating_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    参考 RetailEnvironment._load_sku_data，读取门店下的 SKU。
    返回 sku_id -> {"sku": SKU, "price_series": List[(date, price)], "category": str}
    """
    store_root = Path(config["store_data_path"]) / str(config["store_id"])
    model_param_path = store_root / config["sku_model_parameter"]
    sku_params = load_json(str(model_param_path)) if model_param_path.exists() else {}
    upc_meta = load_upc_meta(config["upc_meta_path"])
    price_meta = load_price_info(config.get("price_info_path", ""))

    payloads: Dict[str, Dict[str, Any]] = {}
    category_buckets: Dict[str, List[SKU]] = {}
    store_id = str(config.get("store_id", ""))

    begin_date = datetime.strptime(str(config["begin_time"]), "%m/%d/%y").date()
    end_date = datetime.strptime(str(config["end_time"]), "%m/%d/%y").date()

    for sku_id, params in sku_params.items():
        meta = price_meta.get(sku_id) or upc_meta.get(sku_id, {})
        if not meta:
            continue

        category = meta.get("CATEGORY") or meta.get("category") or ""
        price_min = meta.get("price_min")
        price_max = meta.get("price_max")
        price_30 = meta.get("price_p30")
        price_70 = meta.get("price_p70")
        if price_min is None or price_max is None:
            continue

        # 构造连续 3-7 天一段的价格序列
        price_series: List[Tuple[Any, float]] = []
        current_date = begin_date
        while current_date <= end_date:
            span = random.randint(3, 7)
            if random.random() < 0.5:
                low, high = price_min, price_30
            else:
                low, high = price_70, price_max
            
            price = random.uniform(low, high)

            for _ in range(span):
                if current_date > end_date:
                    break
                price_series.append((current_date, price))
                current_date += timedelta(days=1)

        if not price_series:
            continue

        init_price = price_series[0][1]
        category_effect = config.get("category_effects", {}).get(category, 0.0)

        rating_val = None

        sku_obj = SKU(
            sku_id=sku_id,
            init_price=init_price,
            model_parameters=params,
            category_effect=category_effect,
            category=category,
        )

        if rating_table:
            rating_val = (
                rating_table.get(store_id, {})
                .get(category, {})
                .get(sku_id)
            )

            if rating_val is not None:
                setattr(sku_obj, "initial_rating", rating_val)
                if isinstance(sku_obj.attributes, dict):
                    sku_obj.attributes["initial_rating"] = rating_val

        payloads[sku_id] = {
            "sku": sku_obj,
            "price_series": price_series,
            "category": category,
            "initial_rating": rating_val if rating_table else None,
        }
        category_buckets.setdefault(category, []).append(sku_obj)

    # 同品类互相登记，模拟品类竞争效应
    for skus in category_buckets.values():
        for sku_obj in skus:
            others = [item for item in skus if item is not sku_obj]
            sku_obj.set_same_category_skus(others)

    return payloads


def load_review_targets(path: str) -> Dict[str, Any]:
    """读取模拟评分配置，找不到文件时返回空 dict。"""
    file_path = Path(path)
    return load_json(str(file_path)) if file_path.exists() else {}


def ingest_supplier_prices(
    config: Dict[str, Any],
    record_manager: RecordManager,
) -> int:
    """
    将 store 下 *_suppliers.json 中、处于日期范围内的报价写入 supplier_prices 表。
    """
    begin_date = datetime.strptime(str(config["begin_time"]), "%m/%d/%y").date()
    end_date = datetime.strptime(str(config["end_time"]), "%m/%d/%y").date()

    store_root = Path(config["store_data_path"]) / str(config["store_id"])
    inserted = 0

    for path in store_root.rglob("*_suppliers.json"):
        try:
            data = load_json(str(path))
        except Exception as exc:
            print(f"[WARN] 读取供应商文件失败 {path}: {exc}")
            continue

        # 文件名里的 upc 作为兜底
        upc_fallback = path.stem.split("_suppliers")[0]

        if not isinstance(data, dict):
            continue

        for supplier_id, rows in data.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                date_str = row.get("date")
                try:
                    date_obj = datetime.fromisoformat(str(date_str)).date()
                except Exception:
                    continue

                if date_obj < begin_date or date_obj > end_date:
                    continue

                price = (
                    row.get("supplier_price")
                    or row.get("price")
                    or row.get("base_cost_price")
                )
                if price is None:
                    continue

                sku_id = row.get("upc") or row.get("sku_id") or upc_fallback
                record = SupplierPriceRecord(
                    supplier_id=str(supplier_id),
                    sku_id=str(sku_id),
                    date_obj=date_obj,
                    price=float(price),
                )
                record_manager.add_supplier_price(record)
                inserted += 1

    return inserted


def _find_supplier_file_for_sku(config: Dict[str, Any], sku_id: str) -> Optional[Path]:
    """
    在 store_data_path 下查找某个 SKU 对应的 *_suppliers.json 文件。
    """
    store_root = Path(config["store_data_path"]) / str(config["store_id"])
    pattern = f"{sku_id}_suppliers.json"
    for path in store_root.rglob(pattern):
        return path
    return None


def _pick_best_quality_supplier_for_day(
    config: Dict[str, Any],
    sku_id: str,
    target_date: datetime.date,
    top_k: int = 2,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    从 *_suppliers.json 中选出指定日期 quality_score 排名前 top_k 的供给商之一（随机挑一个）。
    返回 (supplier_id, quality_score, supplier_price)；若找不到则返回 (None, None, None)。
    """
    supplier_path = _find_supplier_file_for_sku(config, sku_id)
    if not supplier_path or not supplier_path.exists():
        return None, None, None

    data = load_json(str(supplier_path))
    if not isinstance(data, dict):
        return None, None, None

    date_str = target_date.isoformat()
    candidates: List[Tuple[float, float, str]] = []  # (quality_score, price, supplier_id)
    for supplier_id, rows in data.items():
        if supplier_id.startswith("_") or not isinstance(rows, list):
            continue
        for row in rows:
            if str(row.get("date")) != date_str:
                continue
            q = row.get("quality_score")
            p = row.get("supplier_price") or row.get("price")
            try:
                q_val = float(q)
                p_val = float(p)
            except (TypeError, ValueError):
                continue
            candidates.append((q_val, p_val, str(supplier_id)))

    if not candidates:
        return None, None, None
    
    if top_k == 2:
        candidates.sort(key=lambda t: t[0], reverse=True)
        top = candidates[: max(1, min(top_k, len(candidates)))]
        q_val, p_val, sid = random.choices(top, weights=[0.6, 0.4], k=1)[0]
        return sid, q_val, p_val
    # 按质量从高到低排序，然后在前 top_k 个中随机选择一个
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = candidates[: max(1, min(top_k, len(candidates)))]
    q_val, p_val, sid = random.choice(top)
    return sid, q_val, p_val


def target_rating(
    store_id: str,
    category: str,
    sku_id: str,
    rating_table: Dict[str, Any],
    default: float = 4.0,
) -> float:
    """从 rating_table 中取出目标均分，缺失则使用默认值。"""
    return (
        rating_table.get(str(store_id), {})
        .get(category, {})
        .get(sku_id, default)
    )


def simulate_sales_history(
    config: Dict[str, Any],
    cm: CustomerManager,
    sku_payloads: Dict[str, Dict[str, Any]],
    record_manager: RecordManager,
) -> List[SaleRecord]:
    """
    按历史日期模拟销量，并写入 RecordManager（SQLite）。
    """
    begin_date = datetime.strptime(str(config["begin_time"]), "%m/%d/%y").date()
    end_date = datetime.strptime(str(config["end_time"]), "%m/%d/%y").date()
    customer_by_date = customer_lookup(cm)

    sale_records: List[SaleRecord] = []

    days_range = len(
        [date for date in range((end_date - begin_date).days + 1)]
    )

    current_skus = list(sku_payloads.keys())

    for i in range(days_range):
        for sku_id, payload in sku_payloads.items():
            sku_obj: SKU = payload["sku"]
            price = payload["price_series"][i][1]
            sku_obj.set_price(price)
            sku_obj.compute_attribute_attraction(
                date=begin_date + timedelta(days=i)
            )

        for sku_id, payload in sku_payloads.items():
            sku_obj: SKU = payload["sku"]
            sku_obj.compute_attraction(
                currnet_market_skus_ids=current_skus
            )

        for sku_id, payload in sku_payloads.items():
            sku_obj: SKU = payload["sku"]
            date_obj = payload["price_series"][i][0]
            if date_obj < begin_date or date_obj > end_date:
                continue

            customer_count = customer_by_date.get(date_obj)
            if customer_count is None:
                customer_count = random.randint(50, 200)

            sales = max(0, sku_obj.get_sales(
                current_skus,
                int(customer_count)
            ))
            if sales <= 0:
                continue

            record = SaleRecord(sku_id, date_obj, sales, sku_obj.price, customer_count)
            sale_records.append(record)
            record_manager.add_record(sku_id, record)

    sale_records.sort(key=lambda r: r.upc)

    return sale_records


def build_daily_supply_plan(
    config: Dict[str, Any],
    sale_records: List[SaleRecord],
    record_manager: RecordManager,
) -> Dict[Tuple[datetime.date, str], Dict[str, Any]]:
    """
    基于每日销量，为 (date, sku) 规划唯一的供给商选择：
    - 优先使用 *_suppliers.json 中当日 quality_score 前 1~2 名供应商之一；
    - 若没有质量数据，则回退到 supplier_prices 中当日价格最低的供应商。
    返回:
        {(date, sku): {"supplier_id", "quality_score", "supplier_price", "quantity"}}
    """
    plan: Dict[Tuple[datetime.date, str], Dict[str, Any]] = {}
    for rec in sale_records:
        key = (rec.date, rec.upc)
        if key not in plan:
            supplier_id, q_score, s_price = _pick_best_quality_supplier_for_day(
                config=config,
                sku_id=rec.upc,
                target_date=rec.date,
                top_k=2,
            )

            print(f"supplier_id: {supplier_id}, q_score: {q_score}, s_price: {s_price}")
            # 回退：从 supplier_prices 中找当日最低价
            if supplier_id is None or s_price is None:
                rows = record_manager.read_supplier_prices(
                    supplier_id=None,
                    sku_id=rec.upc,
                    start_date=rec.date,
                    end_date=rec.date,
                )
                print("就如回退")
                print(f"rows: {rows}")
                if rows:
                    cheapest = min(rows, key=lambda r: r.price)
                    supplier_id = cheapest.supplier_id
                    s_price = float(cheapest.price)
            plan[key] = {
                "supplier_id": supplier_id,
                "quality_score": q_score,
                "supplier_price": s_price,
                "quantity": 0,
            }
        plan[key]["quantity"] += rec.move
    return plan


def simulate_supplier_orders(
    config: Dict[str, Any],
    supply_plan: Dict[Tuple[datetime.date, str], Dict[str, Any]],
    record_manager: RecordManager,
) -> List[SupplierOrderRecord]:
    """
    基于每日 supply_plan，按「每天为每个供给商聚合一次补货」的策略，模拟进货订单。
    - 对于每个 (date, supplier_id)：聚合这天所有 SKU 的需求，生成一条 SupplierOrderRecord。
    - 这样可以保证：评论里出现的 supplier_id，一定有对应日期的订单记录。
    """
    if not supply_plan:
        return []

    orders: List[SupplierOrderRecord] = []
    # (order_date, supplier_id) -> {sku_id: {"qty": int, "unit_price": float}}
    grouped: Dict[Tuple[datetime.date, str], Dict[str, Dict[str, float]]] = {}

    for (day, sku_id), info in supply_plan.items():
        supplier_id = info.get("supplier_id")
        unit_price = info.get("supplier_price")
        qty = info.get("quantity", 0)
        if not supplier_id or unit_price is None or qty <= 0:
            continue
        key = (day, supplier_id)
        grouped.setdefault(key, {})
        if sku_id not in grouped[key]:
            grouped[key][sku_id] = {"qty": 0, "unit_price": float(unit_price)}
        grouped[key][sku_id]["qty"] += int(qty)

    for (order_date, supplier_id), sku_map in grouped.items():
        if not sku_map:
            continue
        shipping_days = random.randint(2, 7)
        arrival_date = order_date + timedelta(days=shipping_days)
        items: Dict[str, int] = {}
        cost = 0.0
        for sku_id, meta in sku_map.items():
            qty = int(meta["qty"])
            price = float(meta["unit_price"])
            items[sku_id] = qty
            cost += qty * price

        order_record = SupplierOrderRecord(
            supplier_id=supplier_id,
            order_date=order_date,
            arrival_date=arrival_date,
            shipping_days=shipping_days,
            items=items,
            cost=cost,
        )
        record_manager.add_supplier_order(order_record)
        orders.append(order_record)

    return orders


def simulate_return_rates(
    config: Dict[str, Any],
    sale_records: List[SaleRecord],
    rating_table: Dict[str, Any],
    sku_payloads: Dict[str, Dict[str, Any]],
    record_manager: RecordManager,
) -> List[ReturnRateRecord]:
    """
    基于销量与目标评分，模拟退货率记录并写入 return_rate_records 表。
    近似逻辑：
    - 评分/质量越低，退货率越高；
    - 按 (sku_id, date) 聚合当日销量，再为该日生成一条 ReturnRateRecord。
    """
    if not sale_records:
        return []

    store_id = config.get("store_id", "")
    upc_to_category = {sku_id: payload["category"] for sku_id, payload in sku_payloads.items()}

    # 1) 按 (sku_id, date) 聚合销量
    daily_sales: Dict[Tuple[str, datetime.date], int] = {}
    for rec in sale_records:
        key = (rec.upc, rec.date)
        daily_sales[key] = daily_sales.get(key, 0) + rec.move

    results: List[ReturnRateRecord] = []

    for (sku_id, day), move_total in daily_sales.items():
        if move_total <= 0:
            continue
        category = upc_to_category.get(sku_id, "")
        rating_target = target_rating(store_id, category, sku_id, rating_table)

        # 简单规则：目标评分越低，退货率区间越高
        if rating_target <= 2.5:
            low, high = 0.10, 0.20
        elif rating_target <= 3.5:
            low, high = 0.03, 0.08
        else:
            low, high = 0.01, 0.03

        return_rate = random.uniform(low, high)
        return_number = int(round(move_total * return_rate))
        if return_number <= 0:
            continue

        rec = ReturnRateRecord(
            sku_id=sku_id,
            return_rate=return_rate,
            return_number=return_number,
            date_obj=day,
        )
        record_manager.add_return_rate(rec)
        results.append(rec)

    return results


def simulate_returns(
    config: Dict[str, Any],
    sale_records: List[SaleRecord],
    return_rate_records: List[ReturnRateRecord],
    supply_plan: Dict[Tuple[datetime.date, str], Dict[str, Any]],
    record_manager: RecordManager,
) -> List[ReturnRecord]:
    """
    基于销量、退货率和供给计划，模拟具体的退货记录并写入 return_records 表。
    - 对每条销售记录，根据退货率随机决定是否退货
    - 从 supply_plan 中获取对应的 supplier_id
    - 生成 ReturnRecord 并写入数据库
    """
    if not sale_records or not return_rate_records:
        return []

    # 构建 (sku_id, date) -> return_rate 的映射
    return_rate_map: Dict[Tuple[str, datetime.date], float] = {}
    for rr in return_rate_records:
        key = (rr.sku_id, rr.date)
        return_rate_map[key] = rr.return_rate

    results: List[ReturnRecord] = []

    for sale in sale_records:
        key = (sale.upc, sale.date)
        return_rate = return_rate_map.get(key, 0.0)
        
        if return_rate <= 0:
            continue

        # 从 supply_plan 获取 supplier_id
        plan_key = (sale.date, sale.upc)
        plan_info = supply_plan.get(plan_key, {}) if supply_plan else {}
        supplier_id = plan_info.get("supplier_id")
        
        if not supplier_id:
            continue

        # 对每条销售，根据退货率随机决定是否退货
        for _ in range(sale.move):
            if random.random() < return_rate:
                return_record = ReturnRecord(
                    supplier_id=supplier_id,
                    sku_id=sale.upc,
                    date_obj=sale.date,
                )
                record_manager.add_return(return_record)
                results.append(return_record)

    return results


def simulate_news_history(
    config: Dict[str, Any],
    sku_payloads: Dict[str, Dict[str, Any]],
    record_manager: RecordManager,
) -> List[NewRecord]:
    """
    使用“中立新闻”在整个历史区间内填充 new_records 表。
    - 从 news_source_path 读取原始新闻（JSON 或 JSONL）
    - 过滤掉 impact_direction 为 increase/decrease 的记录，仅保留中立新闻
    - 每天随机采样 news_daily_count 条，写入 new_records
    - 返回所有写入的 NewRecord，便于导出 JSONL
    """
    news_source = config.get("news_source_path")
    daily_count = int(config.get("news_daily_count", 0) or 0)
    if not news_source or daily_count <= 0:
        return []

    p = Path(news_source)
    if not p.exists():
        return []

    text = p.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    def is_neutral(item: Dict[str, Any]) -> bool:
        direction = str(
            item.get("impact_direction")
            or item.get("IMPACT_DIRECTION")
            or ""
        ).lower()
        return direction not in {"increase", "decrease"}

    neutral_items = [r for r in rows if is_neutral(r)]
    if not neutral_items:
        neutral_items = rows  # 若没有显式中立新闻，则退化为全部

    begin_date = datetime.strptime(str(config["begin_time"]), "%m/%d/%y").date()
    end_date = datetime.strptime(str(config["end_time"]), "%m/%d/%y").date()

    rng = random.Random(config.get("news_random_seed", 42))
    written: List[NewRecord] = []

    cur = begin_date
    while cur <= end_date:
        if daily_count > 0:
            picked = [rng.choice(neutral_items) for _ in range(daily_count)]
        else:
            picked = []

        for idx, raw in enumerate(picked):
            news_id = str(
                raw.get("id")
                or raw.get("record_id")
                or raw.get("recordId")
                or uuid4().hex
            )
            title = raw.get("title") or raw.get("TITLE") or ""
            content = raw.get("content") or raw.get("CONTENT") or ""
            record_id = f"{news_id}_{cur.isoformat()}_{idx}"
            rec = NewRecord(
                record_id=record_id,
                news_id=news_id,
                title=title,
                content=content,
                date_obj=cur,
            )
            record_manager.add_news_record(rec)
            written.append(rec)

        cur += timedelta(days=1)

    return written


def simulate_review_history(
    config: Dict[str, Any],
    sale_records: List[SaleRecord],
    rating_table: Dict[str, Any],
    sku_payloads: Dict[str, Dict[str, Any]],
    review_manager: ReviewManager,
    supply_plan: Dict[Tuple[datetime.date, str], Dict[str, Any]],
) -> List[ReviewRecord]:
    """兼容旧函数名，内部委托给 simulate_reviews。"""
    return simulate_reviews(
        config=config,
        sale_records=sale_records,
        rating_table=rating_table,
        sku_payloads=sku_payloads,
        review_manager=review_manager,
        supply_plan=supply_plan,
    )


def simulate_reviews(
    config: Dict[str, Any],
    sale_records: List[SaleRecord],
    rating_table: Dict[str, Any],
    sku_payloads: Dict[str, Dict[str, Any]],
    review_manager: ReviewManager,
    supply_plan: Dict[Tuple[datetime.date, str], Dict[str, Any]],
) -> List[ReviewRecord]:
    """
    根据销量记录模拟评论，并通过 ReviewManager 写入记录库。
    - 对每条销售记录，生成 review_ratio 比例的评论条数
    - 评分按照 rating_table 中的目标均分，通过 ReviewStarSmoothModel 进行采样
    - 评论写入 SQLite（review_records 表），同时返回内存中的 ReviewRecord 列表
    """
    review_ratio = float(config.get("review_ratio", 0.1))
    store_id = config.get("store_id", "")
    upc_to_category = {sku_id: payload["category"] for sku_id, payload in sku_payloads.items()}
    reviews: List[ReviewRecord] = []

    for sale in sale_records:
        category = upc_to_category.get(sale.upc, "")
        rating_target = target_rating(store_id, category, sale.upc, rating_table)

        plan_key = (sale.date, sale.upc)
        plan_info = supply_plan.get(plan_key, {}) if supply_plan else {}
        supplier_id = plan_info.get("supplier_id")
        quality_score = plan_info.get("quality_score")
        supplier_price = plan_info.get("supplier_price")

        # 优先从 supply_plan 中取质量分与成本价；若缺失则回退到目标评分 / 销售价近似
        q = quality_score if quality_score is not None else rating_target
        buy_price = supplier_price if supplier_price is not None else float(sale.price) / 3.0
        sell_price = float(sale.price)

        # 逐件模拟是否产生评论（类似 Inventory 中 gen_prob 的概念）
        for _unit in range(sale.move):
            if random.random() > review_ratio:
                continue

            # 与 Inventory 中逻辑保持一致：使用 ReviewStarSmoothModel.simulate_ratings
            rating = ReviewStarSmoothModel.simulate_ratings(q, n=1)[0]
            is_bad_quality = q < 3

            if rating < 3:
                if is_bad_quality:
                    dim = random.choices(
                        population=["price", "quality", "other"],
                        weights=[0.1, 0.8, 0.1],
                        k=1,
                    )[0]
                elif buy_price * 3 < sell_price:
                    dim = random.choices(
                        population=["price", "quality", "other"],
                        weights=[0.8, 0.1, 0.1],
                        k=1,
                    )[0]
                else:
                    dim = random.choice(["price", "quality", "other"])
            else:
                dim = random.choice(["price", "quality", "other"])

            if supplier_id is None:
                raise ValueError(f"supplier_id is None for sku_id: {sale.upc}, date: {sale.date}")

            inserted = review_manager.add_reviews(
                sku_id=sale.upc,
                category=category,
                rating=rating,
                count=1,
                date_obj=sale.date,
                dimension=dim,
                merchandise_id=uuid4().hex,
                supplier_id=supplier_id
            )
            reviews.extend(inserted)

    return reviews


def dump_jsonl(records: List[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config.get("random_seed", 42))

    rating_table = load_review_targets(config["review_target_path"])
    customer_manager = build_customer_manager(config)
    sku_payloads = build_sku_payloads(config, rating_table)

    print(
        f"Length of SKU: {len(sku_payloads)}"
    )

    record_manager = RecordManager(data_dir=config.get("order_record_dir", "order_records"))
    review_manager = ReviewManager(
        record_manager=record_manager,
        review_model_path=config.get("review_model_path"),
        review_source_path=config.get("review_source_path"),
        enabled=True,
    )

    supplier_inserted = ingest_supplier_prices(
        config=config,
        record_manager=record_manager,
    )

    sale_records = simulate_sales_history(
        config=config,
        cm=customer_manager,
        sku_payloads=sku_payloads,
        record_manager=record_manager,
    )
    # 基于销量构建每日供给计划：决定每个 (date, sku) 用哪个 supplier
    supply_plan = build_daily_supply_plan(
        config=config,
        sale_records=sale_records,
        record_manager=record_manager,
    )

    review_records = simulate_reviews(
        config=config,
        sale_records=sale_records,
        rating_table=rating_table,
        sku_payloads=sku_payloads,
        review_manager=review_manager,
        supply_plan=supply_plan,
    )

    output_dir = Path(config["output_dir"])
    # 3. 基于销量与评分模拟退货率
    return_rate_records = simulate_return_rates(
        config=config,
        sale_records=sale_records,
        rating_table=rating_table,
        sku_payloads=sku_payloads,
        record_manager=record_manager,
    )

    # 基于退货率和供给计划模拟具体退货记录
    return_records = simulate_returns(
        config=config,
        sale_records=sale_records,
        return_rate_records=return_rate_records,
        supply_plan=supply_plan,
        record_manager=record_manager,
    )

    # 基于每日供给计划模拟供给商订单（写入 SQLite）
    supplier_orders = simulate_supplier_orders(
        config=config,
        supply_plan=supply_plan,
        record_manager=record_manager,
    )

    dump_jsonl(sale_records, output_dir / config["sales_output_file"])
    record_manager.dump_to_sql(output_dir / "records.sql")
    dump_jsonl(review_records, output_dir / config["review_output_file"])
    dump_jsonl(return_rate_records, output_dir / "simulated_return_rates.jsonl")
    dump_jsonl(return_records, output_dir / "simulated_returns.jsonl")

    # 可选：导出 supplier_orders 的 JSONL 方便检查
    dump_jsonl(
        [o.to_dict() for o in supplier_orders],
        output_dir / "simulated_supplier_orders.jsonl",
    )

    print(
        f"模拟完成：销量 {len(sale_records)} 条，"
        f"评论 {len(review_records)} 条，"
        f"退货率记录 {len(return_rate_records)} 条，"
        f"退货记录 {len(return_records)} 条，"
        f"订单 {len(supplier_orders)} 条"
    )
    print(f"- SQLite 路径: {record_manager.db_path}")
    print(f"- 供应商价格写入: {supplier_inserted}")
    print(f"- 销量文件: {output_dir / config['sales_output_file']}")
    print(f"- 评论文件: {output_dir / config['review_output_file']}")
    print(f"- 订单文件: {output_dir / 'simulated_supplier_orders.jsonl'}")

    # 计算并打印各 SKU 的平均评价星级
    rating_sum: Dict[str, float] = {}
    rating_cnt: Dict[str, int] = {}
    for r in review_records:
        rating_sum[r.upc] = rating_sum.get(r.upc, 0.0) + float(r.rating)
        rating_cnt[r.upc] = rating_cnt.get(r.upc, 0) + 1

    print("各 SKU 平均评分：")
    if not rating_sum:
        print("- 暂无评论数据。")
    else:
        for sku_id in sorted(rating_sum.keys()):
            avg = rating_sum[sku_id] / rating_cnt[sku_id]
            print(f"- {sku_id}: {avg:.3f} (n={rating_cnt[sku_id]})")


if __name__ == "__main__":
    main()

from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta
import random
from typing import Any, Dict, List, Optional
import json
import time
import argparse
from pathlib import Path
from uuid import uuid4
from util.default_config import create_dynamic_hard_config as create_default_config, create_dynamic_middle_config as create_default_middle_config, create_still_hard_config as create_default_still_hard_config, create_still_middle_config as create_default_still_middle_config
from util.file import load_json
from util.sql_formatter import format_sql_rows
from util.logger import get_logger, set_logger_level
from module.customer_manager import CustomerManager
from module.supplier_manager import SupplierManager
from module.inventory import Inventory
from module.order_manager import OrderManager, Order
from module.news_manager import NewsManager
from module.record_manager import (
    RecordManager,
    SupplierOrderRecord,
)
from module.review_manager import WINDOW_DAYS, ReviewManager
from module.sku import SKU, Merchandise

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# raise ValueError("Unable to find legal sku")
# ValueError(f"No rating found for SKU {sku.sku_id}")
#  raise ValueError("limit must be positive")

class RetailEnvironment:
    """
    零售环境入口。
    只实现图中给出的 5 个方法：
    - __init__
    - reset
    - step
    - get_tools
    - exec_tools
    """

    def __init__(self, config) -> None:
        self.config = config

        # 可选：锁定全局随机种子，保证同一配置下运行可复现
        seed = self.config.get("global_random_seed", None)
        if seed is not None:
            try:
                random.seed(int(seed))
            except Exception:
                # 如果配置不合法，忽略种子设置，继续运行
                pass

        self.debug = bool(self.config.get("debug", False))

        set_logger_level(
            self.debug,
        )

        self.logger = get_logger()

        self.inventory = Inventory(
            capacity=config.get("inventory_capacity")
        )
        self.order_manager = OrderManager()

        self.customer_manager = CustomerManager(
            begin_time=config['data_begin_time'],
            end_time=config['data_end_time'],
            data_path=config['customer_data_path'],
            store_id=config.get("store_id", ""),
        )

        self.record_manager = RecordManager(
            data_dir=config["order_record_dir"],
            init_sql_path=config.get("init_sql_path", None),
        )

        self.store_id = config.get("store_id", "")
        self.funds = config.get("initial_funds", 0)

        self.current_date = datetime.strptime(
            config.get("store_begin_time", date.today()),
            "%m/%d/%y"
        ).date()

        store_data_path = config.get("data_dir")

        # SKU 初始化：读取参数与 UPC 元信息
        self.skus_category_map = self._load_sku_data(store_data_path, self.store_id)

        self.skus_list = []
        for category, sku_list in self.skus_category_map.items():
            self.skus_list.extend(sku_list)

        # self._initialize_inventory_with_skus()

        self.skus_id_map: Dict[str, SKU] = {sku.sku_id: sku for sku in self.skus_list}

        self.rent = self.config.get("everyday_rent")

        # Initialize inventory with SKU information
        self.review_manager = ReviewManager(
            record_manager=self.record_manager,
            review_model_path=config.get("review_model_path"),
            review_source_path=config.get("review_source_path"),
            enabled=config.get("enable_review", False),
            gen_prob=config.get("review_ratio", 0.1),
        )

        self._apply_initial_ratings(
            review_manager=self.review_manager
        )

        self.news_manager = (
            NewsManager(
                news_source_path=config.get("news_source_path"),
                random_seed=config.get("news_random_seed"),
                sample_ratios=config.get("news_sample_ratios"),
                daily_news_count=config.get("news_daily_count", 0),
                allowed_categories=list(self.skus_category_map.keys()),
                allowed_sku_ids=list(self.skus_id_map.keys()),
                current_date=self.current_date,
                impact_mode_weights=config.get("news_impact_mode_weights"),
                impact_base_scale=config.get("news_impact_base_scale", 0.2),
            )
            if config.get("enable_new")
            else None
        )

        self.supplier_manager = SupplierManager(
            store_root=config.get("data_dir", ""),
            store_id=config.get("store_id", ""),
            begin_date=config.get("begin_time"),
            end_date=config.get("end_time"),
            news_manager=self.news_manager if self.config.get("enable_new") else None,
            allowed_sku_ids=list(self.skus_id_map.keys()),
        )

        # Initialize notes storage
        self.notes: List[Dict[str, Any]] = []

        

    def _load_sku_data(self, store_data_path: str, store_id: str = "") -> Dict[str, List[SKU]]:
        """
        从 sku_model_parameter.json 和 upc 元数据构建 SKU 实例。

        - 参数文件：store_data_path/store_id/sku_model_parameter.json
        - 元信息：config['upc_meta_path']（默认 data/upc/upc.json），包含 DESCRIP/COM_CODE 等
        """
        param_path = Path(store_data_path) / str(store_id) / "sku_model_parameter.json"
        if not param_path.exists():
            env.log_debug(f"[WARN] sku_model_parameter.json not found: {param_path}")
            return {}
        sku_params = load_json(str(param_path))
        root = Path(
            self.config.get('data_dir')
        )
            
        upc_meta_path = root / f"{self.config.get('store_id', '')}" / f"upc.json"
        upc_meta = load_json(upc_meta_path) if Path(upc_meta_path).exists() else {}

        self.skus_category_map: Dict[str, str] = {}

        for meta in upc_meta:
            category = meta.get("CATEGORY") or meta.get("category")
            sku_id = meta.get("UPC")
            self.skus_category_map[sku_id] = category

        # 仅保留指定 sku_ids（若配置）
        sku_ids_filter = set(self.config.get("sku_ids", []))
        selected_categories = {c.replace("_", " "): True for c in self.config.get("selected_categories", [])}
        def allow_sku(sku_id: str, category: str) -> bool:
            if sku_ids_filter and sku_id not in sku_ids_filter:
                return False
            if selected_categories and category not in selected_categories:
                return False
            return True

        category_to_skus: Dict[str, List[SKU]] = {}

        for sku_id, params in sku_params.items():
            category = self.skus_category_map[sku_id]
            if not allow_sku(sku_id, category):
                continue

            meta = [item for item in upc_meta if item["UPC"] == sku_id][0]
            category = meta.get("CATEGORY") or meta.get("category") or meta.get("COM_CODE") or ""
            price_range = [meta.get('price_min'), meta.get('price_max')]
           
            category_effect = self.config.get("category_effects", {}).get(category, 0.0)

            init_price = random.uniform(float(price_range[0]), float(price_range[1]))

            description = {
                "description": meta.get("DESCRIPTION"),
                "extra": meta,
            }

            brand = meta.get("BRAND")

            sku_obj = SKU(
                sku_id=sku_id,
                description=description,
                category=category,
                model_parameters=params,
                init_price=init_price,
                category_effect=category_effect,
                brand=brand,
                promotion_day=meta.get('PROMOTION_TIME'),
            )

            category_to_skus.setdefault(category, []).append(sku_obj)

        # 同类目互相关联
        for skus in category_to_skus.values():
            for sku_obj in skus:
                sku_obj.set_same_category_skus([s for s in skus if s is not sku_obj])

        return category_to_skus
    
    def _apply_initial_ratings(self, review_manager=None) -> None:
        """
        为 SKU 注入初始评分：
        - 优先使用 review_manager 已有的历史评分均值；
        - 使用滑动时间窗口向更久之前回溯，尽量找到可用评分；
        - 如仍找不到，则保持默认（由 SKU 自身或其他配置决定）。
        """
        if not review_manager or not getattr(review_manager, "enabled", False):
            return

        window_days = WINDOW_DAYS
        max_windows = 20

        for sku in self.skus_list:
            rating_value = None
            window_end = datetime.strptime(
                self.config.get("store_begin_time", date.today()),
                "%m/%d/%y"
            ).date()
            
            window_start = window_end - timedelta(days=window_days)

            # 模拟 review_manager.compute_sales_impact 中的滑动窗口逻辑：
            # 先查最近 WINDOW_DAYS 天的平均评分，如果没有，就继续向前滚动窗口，
            # 最多尝试 max_windows 次，直到找到任意一个可用均值。
            for _ in range(max_windows):
                rating_value = review_manager.get_average_rating(
                    sku_id=sku.sku_id,
                    start_date=window_start,
                    end_date=window_end,
                )
                if rating_value is not None:
                    break
                window_start -= timedelta(days=window_days)

            if rating_value is not None:
                setattr(sku, "initial_rating", rating_value)
                if isinstance(sku.attributes, dict):
                    print(f"Setting initial rating for SKU {sku.sku_id} to {rating_value}")
                    sku.attributes["initial_rating"] = rating_value
            else:
                raise ValueError(f"No rating found for SKU {sku.sku_id}")
   
    def reset(self) -> Dict[str, Any]:
        """重置整个环境。"""

        config = self.config

        self.inventory = Inventory(capacity=config.get("inventory_capacity"))
        self.order_manager = OrderManager()
        self.customer_manager = CustomerManager(
            begin_time=config['customer']['begin_time'],
            end_time=config['customer']['end_time'],
            data_path=config['customer']['data_path'],
            store_id=config.get("store_id", ""),
        )

        self.funds = config.get("initial_funds", 0)
        
        self.current_date = datetime.strptime(
            config.get("begin_time", "10/01/89"),
            "%m/%d/%y"
        ).date()

        self.config = config

        # Initialize SKU data by reading from store data path
        # If store_id is specified in config, look for that store's data
        store_id = config.get("store_id", "")
        store_data_path = config["data_dir"]
        
        # Look for the specific store's processed data directory
        # The structure appears to be: data_dir / store_id / categories / summary.json
        self.skus_category_map = self._load_sku_data(store_data_path, store_id)

        self.skus_list = []
        for category, sku_list in self.skus_category_map.items():
            self.skus_list.extend(sku_list)

        self.skus_id_map = {sku.sku_id: sku for sku in self.skus_list}

        # Initialize inventory with SKU information
        self._initialize_inventory_with_skus()

        self.review_manager = ReviewManager(
            record_manager=self.record_manager,
            review_model_path=config.get("review_model_path"),
            review_source_path=config.get("review_source_path"),
            enabled=config.get("enable_review", False),
        )

        self.news_manager = (
            NewsManager(
                news_source_path=config.get("news_source_path"),
                random_seed=config.get("news_random_seed"),
                sample_ratios=config.get("news_sample_ratios"),
                daily_news_count=config.get("news_daily_count", 0),
                allowed_categories=list(self.skus_category_map.keys()),
                allowed_sku_ids=list(self.skus_id_map.keys()),
                current_date=self.current_date,
                impact_mode_weights=config.get("news_impact_mode_weights"),
                impact_base_scale=config.get("news_impact_base_scale", 0.2),
            )
            if config.get("enable_new")
            else None
        )
        
        self._apply_initial_ratings()

        
        # Initialize SKU records with synthetic sales data
        # self._init_sku_records()
        
        # Return the initial state after reset
        return {
            "funds": self.funds,
            "current_date": self.current_date,
            "num_skus": len(self.skus_list)
        }

    def step(self) -> Dict[str, Any]:
        """
        环境前进一步：推进一天，并让各 manager 执行各自的 step。
        """

        money_earned, insufficient_skus, sales_by_sku, returns_by_sku, expired_discount_by_sku, waiting_items = self.inventory.step(
            self.current_date,
            self.customer_manager.get_customer_count(self.current_date),
            self.skus_id_map,
            record_manager=self.record_manager,
            review_manager=self.review_manager if self.review_manager.enabled else None,
            new_manager=self.news_manager if self.config.get('enable_new') and self.news_manager else None,
        )

        self.supplier_manager.step(
            current_date=self.current_date,
            record_manager=self.record_manager
        )

        if self.news_manager and self.config.get("enable_new"):
            self.news_manager.step(
                record_manager=self.record_manager,
                current_date=self.current_date
            )

        self.funds += money_earned
        self.current_date += timedelta(days=1)

        arrived_merchanises = self.order_manager.step(
            current_date=self.current_date,
            record_manager=self.record_manager,
        )

        for merchandise in arrived_merchanises:
            self.inventory.add_item(merchandise)

        self.funds -= self.rent

        # 使用 view_inventory 获取库存信息
        inventory_result = self.view_inventory()
        inventory_data = inventory_result.get("result", {})

        step_result = {
            "funds": self.funds,
            "net_worth": self.inventory.compute_net_worth(self.current_date) + self.funds + sum([order.cost for order in self.order_manager.get_current_orders()]),
            "current_date": self.current_date,
            "money_earned": money_earned,
            "insufficient_skus": insufficient_skus,
            "sales_by_sku": sales_by_sku,
            "returns_by_sku": returns_by_sku,
            "expired_discount_by_sku": expired_discount_by_sku,
            "waiting_items": waiting_items,
            "inventory": inventory_data,  # 直接使用 view_inventory 返回的数据
        }
        return self._wrap_tool_result(step_result, self._format_step_result(step_result))

    # =========================
    # 工具函数与 MCP 工具定义
    # =========================

    @staticmethod
    def _wrap_tool_result(result: Any, formatted: str) -> Dict[str, Any]:
        """统一返回格式：同时带原始结果与格式化字符串。"""
        return {"result": result, "formatted": formatted}

    def log_debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def log_info(self, msg: str) -> None:
        self.logger.info(msg)

    def _json_safe(self, obj: Any) -> Any:
        """递归转为可 JSON 序列化的结构，并确保键为字符串。"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        return str(obj)

    def _log_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any, elapsed_time: float) -> None:
        """
        记录工具调用日志（JSONL 格式），包含返回结果与资金/净值快照。
        每行一个 JSON 对象，使用 append 模式写入。
        
        Args:
            tool_name: 工具名称
            args: 工具参数
            result: 工具返回结果
            elapsed_time: 工具执行耗时（秒）
        """

        self.log_debug(
            f"当前 timestamp: {timestamp}"
        )

        log_dir = Path(self.config.get("log_dir", f"logs/{timestamp}"))

        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tool_calls.jsonl"

        inventory_worth = self.inventory.compute_net_worth(current_date=self.current_date)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tool": tool_name,
            "args": self._json_safe(args),
            "result": self._json_safe(result),
            "funds": self.funds,
            "net_worth": inventory_worth + self.funds,
            "current_date": self.current_date.isoformat() if isinstance(self.current_date, date) else str(self.current_date),
            "elapsed_time": round(elapsed_time, 4),  # 保留4位小数
        }
        
        # 使用 append 模式写入 JSONL 格式（每行一个 JSON 对象）
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _format_step_result(self, result: Dict[str, Any]) -> str:
        """格式化 step 返回结果。"""
        date_str = result.get("current_date")
        funds = result.get("funds")
        networth = result.get("net_worth")
        money = result.get("money_earned")
        insufficient = result.get("insufficient_skus") or []
        insufficient_skus = ' , '.join([sku.sku_id for sku in insufficient])
        sales_by_sku = result.get("sales_by_sku") or {}
        returns_by_sku = result.get("returns_by_sku") or {}
        expired_by_sku = result.get("expired_discount_by_sku") or {}

        waiting_by_sku = defaultdict(int)
        for merch in self.inventory.waiting_items:
            waiting_by_sku[merch.sku.sku_id] += 1

        # 从结果中获取库存信息（来自 view_inventory）
        inventory_data = result.get("inventory", {})
        total_items = inventory_data.get("total_items", 0)
        total_waiting = inventory_data.get("waiting_items", 0)
        inventory_by_sku = inventory_data.get("inventory", {})

        lines = [
            f"Current Date: {date_str}",
            f"Current Funds: {funds}",
            f"Current Networth: {networth}",
            f"Money earned yestarday: {money}",
        ]
        
        # 添加库存摘要信息
        if self.inventory.capacity is not None:
            capacity_used = total_items + total_waiting
            capacity_available = self.inventory.capacity - capacity_used
            lines.append(f"Inventory: {total_items} items in stock, {total_waiting} waiting ({capacity_used}/{self.inventory.capacity} used, {capacity_available} available)")
        else:
            lines.append(f"Inventory: {total_items} items in stock, {total_waiting} waiting")
        
        if insufficient:
            lines.append(f"Insufficient SKUs: {insufficient_skus}")
        else:
            lines.append("No stockouts today.")
        if sales_by_sku:
            sales_lines = ", ".join(f"{k}={v}" for k, v in sales_by_sku.items())
            lines.append(f"Sales by SKU: {sales_lines}")
        if returns_by_sku:
            return_lines = ", ".join(f"{k}={v}" for k, v in returns_by_sku.items())
            lines.append(f"Returns by SKU: {return_lines}")
        if expired_by_sku:
            expired_lines = ", ".join(f"{k}={v}" for k, v in expired_by_sku.items())
            lines.append(f"Expired clearance sold: {expired_lines}")
        if waiting_by_sku:
            promo_lines = ", ".join(f"{k}={v}" for k, v in waiting_by_sku.items())
            lines.append(f"Waiting SKU Items: {promo_lines}")
        
        # 添加库存详情（只显示有库存的 SKU，最多10个）
        inventory_summary = []
        for sku_id, data in inventory_by_sku.items():
            qty = data.get("quantity", 0)
            waiting = data.get("waiting", 0)
            if qty > 0 or waiting > 0:
                inventory_summary.append(f"{sku_id}: {qty}" + (f" (waiting {waiting})" if waiting > 0 else ""))
        
        if inventory_summary:
            lines.append(f"Inventory details: {', '.join(inventory_summary[:10])}" + ("..." if len(inventory_summary) > 10 else ""))
        
        return "\n".join(lines)

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """允许 date 或常见字符串格式，其他返回 None。"""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        for fmt in ("%Y-%m-%d", "%m/%d/%y"):
            try:
                return datetime.strptime(str(value), fmt).date()
            except Exception:
                continue
        return None

    def get_sku_rating_report(
        self,
        sku_ids: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return initial and average ratings for given SKUs.
        Not registered as a tool; callable directly.
        """
        try:
            self._ensure_review_mgr()
            target_ids = self._filter_allowed_skus(sku_ids) if sku_ids else list(self.skus_id_map.keys())
            start = self._parse_date(start_date)
            end = self._parse_date(end_date)

            def initial_rating(sku_obj: SKU) -> Optional[float]:
                if hasattr(sku_obj, "initial_rating"):
                    return getattr(sku_obj, "initial_rating")
                if isinstance(getattr(sku_obj, "attributes", None), dict):
                    return sku_obj.attributes.get("initial_rating")
                return None

            report: Dict[str, Dict[str, Any]] = {}
            for sid in target_ids:
                sku_obj = self.skus_id_map.get(sid)
                if not sku_obj:
                    continue
                init_rate = initial_rating(sku_obj)
                avg_rate = self.review_manager.get_average_rating(
                    sku_id=sid,
                    start_date=self.current_date - timedelta(days=WINDOW_DAYS),
                    end_date=self.current_date
                )

                report[sid] = {
                    "initial_rating": init_rate,
                    "avg_rating": avg_rate,
                }
            return report
        except Exception as e:
            err_msg = f"Error executing get_sku_rating_report: {type(e).__name__}: {e}"
            # 这个方法不是工具方法，直接返回错误字典
            return {"error": err_msg}
    # ---------- 工具实现 ----------
    def view_funds_and_date(self) -> Dict[str, Any]:
        payload = {
            "funds": self.funds,
            "current_date": self.current_date,
        }
        formatted = f"Current date: {self.current_date}, funds balance: {self.funds:.2f}"
        return self._wrap_tool_result(payload, formatted)

    def view_inventory(self) -> Dict[str, Any]:
        inventory_info: Dict[str, Dict[str, Any]] = {}
        total_items = 0
        waiting_by_sku = defaultdict(int)

        for merch in self.inventory.waiting_items:
            waiting_by_sku[merch.sku.sku_id] += 1

        total_waiting = sum(waiting_by_sku.values())

        for sku_id, value in self.skus_id_map.items():
            inventory_info[sku_id] = {
                "quantity": 0,
                "waiting": waiting_by_sku.get(sku_id, 0),
            }

        for sku_id, items in self.inventory.items_by_sku.items():
            sku_obj = self.skus_id_map.get(sku_id)
            quantities = len(items)
            total_items += quantities
            inventory_info[sku_id] = {
                "quantity": quantities,
                "waiting": waiting_by_sku.get(sku_id, 0),
            }

        formatted_lines = [
            f"Inventory summary: {len(inventory_info)} SKUs, {total_items} total items"
        ]
        if total_waiting:
            formatted_lines[0] += f" (+{total_waiting} waiting)"

        for sku_id, data in list(inventory_info.items()):
            qty = data["quantity"]
            waiting = data.get("waiting", 0)
            if qty == 0 and waiting == 0:
                continue
            formatted_lines.append(
                f"- {sku_id}: quantity {qty}" + (f" (waiting {waiting})" if waiting else "")
            )

        if len(formatted_lines) == 1:
            formatted_lines.append("Inventory is empty.")
        
        return self._wrap_tool_result(
            {
                "inventory": inventory_info,
                "total_items": total_items,
                "total_skus": len(inventory_info),
                "waiting_items": total_waiting,
            },
            "\n".join(formatted_lines),
        )

    def _ensure_review_mgr(self) -> None:
        if not getattr(self, "review_manager", None):
            raise ValueError("Review manager not initialized.")
        if not self.review_manager.enabled:
            raise ValueError("Reviews feature disabled.")

    def _filter_allowed_skus(self, sku_ids: Optional[List[str]]) -> List[str]:
        """Return only SKUs present in the environment; used to guard tools."""
        if not sku_ids:
            return []
        allowed = set(self.skus_id_map.keys())
        return [sid for sid in sku_ids if sid in allowed]

    def _validate_sku_list(
        self,
        sku_ids: Any,
        allow_empty: bool = False,
    ) -> List[str]:
        """Validate sku_ids input and return the list; raises on invalid input."""
        if sku_ids is None:
            if allow_empty:
                return []
            raise ValueError("sku_ids is required")
        if not isinstance(sku_ids, list):
            raise ValueError("sku_ids must be an array")
        if not sku_ids and not allow_empty:
            raise ValueError("sku_ids cannot be empty")
        for idx, sid in enumerate(sku_ids):
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError(f"sku_ids[{idx}] is invalid")
            if sid not in self.skus_id_map:
                raise ValueError(f"Unknown SKU: {sid}")
        return sku_ids

    def _validate_dates(
        self,
        start_date: Any = None,
        end_date: Any = None,
        allow_none: bool = False,
    ) -> Any:
        """
        Validate date inputs; returns parsed (start, end).
        - When allow_none is False, both start_date and end_date are required.
        - Accepts date objects or strings in YYYY-MM-DD / MM/DD/YY.
        """
        if not allow_none:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date are required")

        start = self._parse_date(start_date) if start_date is not None else None
        end = self._parse_date(end_date) if end_date is not None else None

        if start_date is not None and start is None:
            raise ValueError("Invalid start_date format; use YYYY-MM-DD or MM/DD/YY.")
        if end_date is not None and end is None:
            raise ValueError("Invalid end_date format; use YYYY-MM-DD or MM/DD/YY.")

        if start and end and start > end:
            raise ValueError("start_date cannot be after end_date")

        return start, end

    def view_sku_reviews(
        self,
        sku_ids: List[str],
        start_date: str = None,
        end_date: str = None,
        ratings: List[int] = [1, 2, 3, 4, 5],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_review_mgr()
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("No allowed sku_ids to query.")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        if ratings is None:
            ratings = [1, 2, 3, 4, 5]
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be positive")
        else:
            limit = 20  # align with tool schema default

        payload: Dict[str, Any] = {}
        lines = ["## Reviews"]
        for sku_id in sku_ids:
            reviews = self.review_manager.get_reviews_in_range(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
                ratings=ratings,
            )
            if limit is not None:
                if len(reviews) <= limit:
                    pass
                else:
                    reviews = random.sample(reviews, limit)

            payload[sku_id] = [r.to_dict() for r in reviews]
            lines.append(f"### {sku_id} ({len(reviews)})")
            lines.append("| Date | Rating | Category | Dimension | Supplier | Merchandise | Comment |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in reviews:
                lines.append(
                    f"| {r.date} | {r.rating} | {r.category or '-'} | {r.dimension or '-'} | "
                    f"{getattr(r, 'supplier_id', '-') or '-'} | {getattr(r, 'merchandise_id', '-') or '-'} | "
                    f"{r.comment or '-'} |"
                )
            lines.append("")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_sku_avg_ratings(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        self._ensure_review_mgr()
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("No allowed sku_ids to query.")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        payload: Dict[str, Any] = {}
        lines = ["## Average ratings"]
        for sku_id in sku_ids:
            avg = self.review_manager.get_average_rating(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            payload[sku_id] = avg
            lines.append(f"- {sku_id}: {avg if avg is not None else 'N/A'}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_current_orders(self) -> Dict[str, Any]:
        orders_detail = [order.get_detail() for order in self.order_manager.get_current_orders()]
        formatted_lines = [
            f"## Open orders: {len(orders_detail)}",
            "",
            "| # | Order ID | SKU Count | Expected Delivery | Items |",
            "|---|---|---|---|---|",
        ]

        for idx, detail in enumerate(orders_detail):
            detail_items = detail.get("items") or []
            lines = [
                f"{item.get('sku_id', '')}: {item.get('count', '')}"
                for item in detail_items
            ]

            formatted_lines.append(
                f"| {idx} | {detail['order_id']} | {len(detail['ordered_sku'])} | "
                f"{detail['expected_delivery_date']} | {', '.join(lines) or '-'} |"
            )

        return self._wrap_tool_result(
            {"total_orders": len(orders_detail), "orders": orders_detail},
            "\n".join(formatted_lines),
        )

    def view_sku_sales_history(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        payload: Dict[str, Any] = {}
        formatted_sections: List[str] = []

        for sku_id in sku_ids:
            records = self.record_manager.read_sku(sku_id, start, end)

            # 结构化 payload
            daily = {}
            for day, recs in records.items():
                daily[day] = []
                for rec in recs:
                    item = {
                        "sku_id": rec.upc,
                        "date": rec.date,
                        "move": rec.move,
                        "price": rec.price,
                        "customer_count": rec.customer_count,
                    }
                    daily[day].append(item)

            total_units = sum(rec.move for recs in records.values() for rec in recs)

            payload[sku_id] = {
                "records": daily,
                "total_units": total_units,
                "days": len(records),
            }

            # 📌 Markdown 表格构造
            table_lines = [
                f"### SKU: {sku_id}",
                "",
                "| Date | Move | Price | Customers |",
                "|------|------|-------|-----------|",
            ]

            for day, recs in records.items():
                for rec in recs:
                    table_lines.append(
                        f"| {rec.date} | {rec.move} | {rec.price} | {rec.customer_count} |"
                    )

            formatted_sections.append("\n".join(table_lines))

        formatted = "\n\n".join(formatted_sections)
        return self._wrap_tool_result(payload, formatted)

    def view_current_date_supplier_prices(
        self,
        sku_ids: List[str],
    ) -> Dict[str, Any]:
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        dt = self.current_date

        payload: Dict[str, Any] = {}
        supplier_groups = defaultdict(list)

        for sku_id in sku_ids:
            entries = self.supplier_manager.get_sku_date(sku_id, dt)
            payload[sku_id] = entries

            for entry in entries or []:
                supplier_groups[entry['supplier_id']].append(entry)

        # ------- 按 supplier 分组输出 -------
        lines = [
            f"## Supplier prices on {dt}",
            "",
            "| Supplier | SKU | Price |",
            "|---|---|---|",
        ]
        if not supplier_groups:
            lines.append("| - | - | - |")

        for supplier_id, entries in supplier_groups.items():
            for entry in entries:
                sku = entry["sku_id"]
                price = entry["price"]
                lines.append(f"| {supplier_id} | {sku} | {price} |")

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_supplier_price_history(
        self,
        supplier_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sku_id:
            self._validate_sku_list([sku_id])
        sku_ids = self._filter_allowed_skus([sku_id])
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)
        records = self.record_manager.read_supplier_prices(
            supplier_id=supplier_id,
            sku_id=sku_id,
            start_date=start,
            end_date=end,
        )

        payload = [
            {
                "supplier_id": r.supplier_id,
                "sku_id": r.sku_id,
                "date": r.date,
                "price": r.price,
            }
            for r in records
        ]

        lines = [
            f"## Historical supplier price records ({len(payload)} rows)",
            f"- filters: supplier_id={supplier_id}, sku_id={sku_id}, start={start}, end={end}",
            "",
            "| # | Supplier | SKU | Date | Price |",
            "|---|---|---|---|---|",
        ]
        for idx, row in enumerate(payload):
            lines.append(
                f"| {idx} | {row['supplier_id']} | {row['sku_id']} | {row['date']} | {row['price']} |"
            )

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_return_rates(
        self,
        sku_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """查看 SKU 的退货率记录。"""
        self._validate_sku_list(sku_ids)
        sku_ids = self._filter_allowed_skus(sku_ids)
        if not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        payload: Dict[str, List[Dict[str, Any]]] = {}
        lines = [f"## Return rates {start} ~ {end}"]
        lines.append("| SKU | Date | Return Rate | Return Number |")
        lines.append("|---|---|---|---|")

        for sku_id in sku_ids:
            recs = self.record_manager.read_return_rates(
                sku_id=sku_id,
                start_date=start,
                end_date=end,
            )
            payload[sku_id] = [r.to_dict() for r in recs]
            if not recs:
                lines.append(f"| {sku_id} | - | - | - |")
            else:
                for r in recs:
                    lines.append(f"| {sku_id} | {r.date} | {r.return_rate} | {getattr(r, 'return_number', '-') or '-'} |")

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_returns(
        self,
        supplier_id: Optional[str] = None,
        sku_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """查看退货记录，可按 supplier_id / sku_id / 日期范围过滤。"""
        if sku_id:
            self._validate_sku_list([sku_id])
        sku_ids = self._filter_allowed_skus([sku_id]) if sku_id else []
        if sku_id and not sku_ids:
            raise ValueError("Unable to find legal sku")
        start, end = self._validate_dates(start_date, end_date, allow_none=True)

        records = self.record_manager.read_returns(
            supplier_id=supplier_id,
            sku_id=sku_id,
            start_date=start,
            end_date=end,
        )

        payload = [
            {
                "supplier_id": r.supplier_id,
                "sku_id": r.sku_id,
                "date": r.date,
            }
            for r in records
        ]

        lines = [
            f"## Return records ({len(payload)} rows)",
            f"- filters: supplier_id={supplier_id}, sku_id={sku_id}, start={start}, end={end}",
            "",
            "| # | Supplier | SKU | Date |",
            "|---|---|---|---|",
        ]
        for idx, row in enumerate(payload):
            lines.append(
                f"| {idx} | {row['supplier_id']} | {row['sku_id']} | {row['date']} |"
            )

        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_sku_prices(self, sku_ids: List[str]) -> Dict[str, Any]:
        if sku_ids is not None:
            self._validate_sku_list(sku_ids, allow_empty=True)
        target_ids = self._filter_allowed_skus(sku_ids)
        if sku_ids is not None and not target_ids:
            raise ValueError("Unable to find legal sku")

        prices = {}
        for sku_id in target_ids:
            sku_obj = self.skus_id_map.get(sku_id)
            if sku_obj:
                prices[sku_id] = sku_obj.price
        lines = [
            "## Current prices",
            "",
            "| SKU | Price |",
            "|---|---|",
        ]
        for sku_id, price in list(prices.items()):
            lines.append(f"| {sku_id} | {price} |")
        return self._wrap_tool_result(prices, "\n".join(lines))

    def modify_sku_price(self, sku_id: str, new_price: float) -> Dict[str, Any]:
        if not isinstance(new_price, (int, float)):
            raise ValueError("new_price must be a number")
        if new_price <= 0:
            raise ValueError("new_price must be positive")
        sku_obj = self.skus_id_map.get(sku_id)
        if sku_obj is None:
            raise ValueError(f"SKU {sku_id} 不存在")
        old_price = sku_obj.price
        sku_obj.set_price(new_price)
        formatted = f"Updated SKU {sku_id} price: {old_price} -> {new_price}"
        return self._wrap_tool_result(
            {"sku_id": sku_id, "old_price": old_price, "new_price": new_price},
            formatted,
        )

    def view_today_news(self) -> Dict[str, Any]:
        if not getattr(self, "news_manager", None) or not self.config.get("enable_new"):
            raise ValueError("News manager not initialized.")
        news = self.news_manager.get_today_news()
        payload = [{"id": n.get("id") or n.get("record_id"), "record_id": n.get("record_id"), "title": n.get("title", "")} for n in news]
        lines = ["## Today's news"]
        for item in payload:
            lines.append(f"- {item['id']}: {item['title']}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def view_news_detail(self, news_id: str) -> Dict[str, Any]:
        if not isinstance(news_id, str) or not news_id.strip():
            raise ValueError("news_id is required")
        if not getattr(self, "news_manager", None) or not self.config.get("enable_new"):
            raise ValueError("News manager not initialized.")
        news = self.news_manager.fetch_news_detail(self.record_manager, news_id, current_date=self.current_date)
        if not news:
            raise ValueError(f"News id not found: {news_id}")
        return self._wrap_tool_result(news, json.dumps(news, ensure_ascii=False, indent=2, default=str))

    def view_news_history(self, start_date: str, end_date: str) -> Dict[str, Any]:
        if not getattr(self, "news_manager", None) or not self.config.get("enable_new"):
            raise ValueError("News manager not initialized.")
        start, end = self._validate_dates(start_date, end_date, allow_none=False)

        payload = self.news_manager.fetch_news_history(self.record_manager, start, end)
        lines = [f"## News {start} ~ {end} ({len(payload)} items)"]
        for item in payload:
            lines.append(f"- {item['id']}: {item['title']}")
        return self._wrap_tool_result(payload, "\n".join(lines))

    def add_note(self, content: str) -> Dict[str, Any]:
        """
        Add a note to the store's note system.
        
        Args:
            content: The content of the note to add.
            
        Returns:
            Dict containing the note information.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Note content must be a non-empty string")
        
        note = {
            "id": uuid4().hex,
            "content": content.strip(),
            "date": self.current_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
        
        self.notes.append(note)
        
        formatted = f"Note added successfully.\n- ID: {note['id']}\n- Date: {note['date']}\n- Content: {content.strip()}"
        return self._wrap_tool_result(note, formatted)

    def view_notes(self) -> Dict[str, Any]:
        """
        View all notes stored in the system.
        
        Returns:
            Dict containing all notes.
        """
        payload = {
            "total_notes": len(self.notes),
            "notes": self.notes,
        }
        
        if not self.notes:
            formatted = "No notes found."
        else:
            lines = [f"## Notes ({len(self.notes)} total)"]
            for idx, note in enumerate(self.notes, 1):
                lines.append(f"\n### Note {idx}")
                lines.append(f"- ID: {note['id']}")
                lines.append(f"- Date: {note['date']}")
                lines.append(f"- Content: {note['content']}")
            formatted = "\n".join(lines)
        
        return self._wrap_tool_result(payload, formatted)

    def place_order(
        self,
        items: List[Dict[str, Any]],
        supplier_id: str,
    ) -> Dict[str, Any]:
        """
        Place one order containing multiple SKUs (items = [{sku_id, quantity}, ...]).
        If supplier_id is omitted, each SKU picks the cheapest quote for today; mixed suppliers will be marked as mixed.
        """
        if not items:
            raise ValueError("items cannot be empty")
        
        if not supplier_id:
            raise ValueError("supplier_id is required")

        # 参数校验：必须是合法的 sku_id 且数量为正整数
        if not isinstance(items, list):
            raise ValueError("items must be an array")

        for idx, line in enumerate(items):
            if not isinstance(line, dict):
                raise ValueError(f"items[{idx}] must be an object")

            sku_id = line.get("sku_id")
            qty = line.get("quantity")

            if sku_id is None or qty is None:
                raise ValueError(f"items[{idx}] must include sku_id and quantity")

            if not isinstance(sku_id, str) or not sku_id.strip():
                raise ValueError(f"items[{idx}].sku_id is invalid")

            if not isinstance(qty, int):
                raise ValueError(f"items[{idx}].quantity must be an integer")

            if qty <= 0:
                raise ValueError(f"items[{idx}].quantity must be a positive integer")

            if sku_id not in self.skus_id_map:
                raise ValueError(f"Unknown SKU: {sku_id}")
            
        if supplier_id not in self.supplier_manager.get_available_suppliers():
            raise ValueError(f"Unknown supplier: {supplier_id}")

        merchandises: List[Merchandise] = []
        ordered_sku_map: Dict[SKU, int] = {}
        supplier_used_set = set()
        total_cost = 0.0
        line_results = []
        delivery_days: Optional[int] = None
        arrival_date: Optional[date] = None

        for line in items:
            sku_id = line.get("sku_id")
            qty = line.get("quantity")

            today_quotes = self.supplier_manager.get_sku_date(sku_id, self.current_date)
            chosen: Optional[Dict[str, Any]] = None
            if supplier_id:
                chosen = next((q for q in today_quotes if q.get("supplier_id") == supplier_id), None)

            if chosen is None:
                raise ValueError(f"No quote found for supplier {supplier_id} and SKU {sku_id} on {self.current_date}")

            unit_price = chosen["price"]
            supplier_used = chosen["supplier_id"]
            supplier_used_set.add(supplier_used)

            quality_score = self.supplier_manager.get_quality_score(
                supplier_id=supplier_used,
                sku_id=sku_id,
                target_date=self.current_date,
            )

            transport_range = self.supplier_manager.get_transport_range(
                supplier_id=supplier_used,
                sku_id=sku_id,
                target_date=self.current_date,
            )

            if transport_range:
                shipping_days = random.randint(transport_range[0], transport_range[1])
            else:
                shipping_days = random.randint(3, 7)
            arrival_line = self.current_date + timedelta(days=shipping_days)
            delivery_days = shipping_days
            arrival_date = arrival_line
            sku_obj = self.skus_id_map[sku_id]
            shelf_life = sku_obj.promotion_day

            for _ in range(qty):
                merchandises.append(
                    Merchandise(
                        sku=sku_obj,
                        begin_time=arrival_line,
                        expired_time=arrival_line + timedelta(days=shelf_life),
                        buy_price=unit_price,
                        merch_id=uuid4().hex,
                        quality_score=quality_score,
                        supplier_id=supplier_used,
                    )
                )

            ordered_sku_map[sku_obj] = ordered_sku_map.get(sku_obj, 0) + qty
            cost = unit_price * qty
            total_cost += cost
            line_results.append(
                {
                    "sku_id": sku_id,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "supplier_id": supplier_used,
                    "line_cost": cost,
                }
            )

        order_supplier = supplier_id or (supplier_used_set.pop() if len(supplier_used_set) == 1 else "mixed")
        if delivery_days is None or arrival_date is None:
            delivery_days = random.randint(3, 7)
            arrival_date = self.current_date + timedelta(days=delivery_days)
        order_id = uuid4().hex

        order = Order(
            order_id=order_id,
            ordered_sku=ordered_sku_map,
            customer_id=order_supplier,
            items=merchandises,
            created_at=self.current_date,
            delivery_time=delivery_days,
            cost=total_cost,
        )

        if self.funds < total_cost:
            return self._wrap_tool_result(
                {"error": "Insufficient funds", "required": total_cost, "available": self.funds},
                f"Insufficient funds to place the order. Required: {total_cost:.2f}, Available: {self.funds:.2f}"
            )

        # 扣除资金并添加订单
        self.funds -= total_cost
        self.order_manager.add_order(order)

        # 记录供应商订单
        supplier_order_record = SupplierOrderRecord(
            supplier_id=order_supplier,
            order_date=self.current_date,
            arrival_date=arrival_date,
            shipping_days=delivery_days,
            items={line["sku_id"]: line["quantity"] for line in line_results},
            cost=total_cost,
        )
        self.record_manager.add_supplier_order(supplier_order_record)

        formatted_lines = [
            f"**Order placed**",
            f"- ID: `{order_id}`",
            f"- Expected arrival: {arrival_date}",
            f"- Supplier: {order_supplier}",
            f"- Total cost: {total_cost:.2f}",
            f"- Funds after: {self.funds:.2f}",
            "",
            "| SKU | Qty | Unit Price | Supplier | Line Cost |",
            "|---|---|---|---|---|",
        ]
        for line in line_results:
            formatted_lines.append(
                f"| {line['sku_id']} | {line['quantity']} | {line['unit_price']} | {line['supplier_id']} | {line['line_cost']:.2f} |"
            )

        result = {
            "order_id": order_id,
            "arrival_date": arrival_date,
            "supplier_id": order_supplier,
            "total_cost": total_cost,
            "funds_after": self.funds,
            "lines": line_results,
        }
        return self._wrap_tool_result(result, "\n".join(formatted_lines))
    
    def end_of_day(self) -> Dict[str, Any]:
        return self.step()
    
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        保存当前环境状态到 checkpoint 文件。
        
        Args:
            checkpoint_path: checkpoint 文件路径
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化 checkpoint 数据
        checkpoint_data = {
            "funds": self.funds,
            "current_date": self.current_date.isoformat(),
            "sku_prices": {sku_id: sku.price for sku_id, sku in self.skus_id_map.items()},
            "config": self.config,  # 保存配置以便恢复
        }
        
        # 调用各个 manager 的 save_checkpoint 方法
        checkpoint_data = self.inventory.save_checkpoint(checkpoint_data)
        checkpoint_data = self.order_manager.save_checkpoint(checkpoint_data)
        
        # 保存 news_manager 状态（如果启用）
        if self.news_manager and self.config.get("enable_new"):
            checkpoint_data = self.news_manager.save_checkpoint(checkpoint_data)
        
        # 保存 record_manager 数据库到 SQL 文件
        checkpoint_dir = checkpoint_path.parent
        sql_dump_path = checkpoint_dir / f"{checkpoint_path.stem}_records.sql"
        self.record_manager.dump_to_sql(sql_dump_path)
        checkpoint_data["record_manager_sql_path"] = str(sql_dump_path.relative_to(checkpoint_dir))
        
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    @classmethod
    def recover_from_checkpoint(cls, checkpoint_path: Path) -> "RetailEnvironment":
        """
        从 checkpoint 文件恢复环境状态。
        
        Args:
            checkpoint_path: checkpoint 文件路径
            
        Returns:
            恢复后的 RetailEnvironment 实例
        """
        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        config = checkpoint_data["config"]
        env = cls(config)
        
        # 恢复基本状态
        env.funds = checkpoint_data["funds"]
        env.current_date = datetime.strptime(checkpoint_data["current_date"], "%Y-%m-%d").date()
        
        # 恢复 SKU 价格
        sku_prices = checkpoint_data.get("sku_prices", {})
        for sku_id, price in sku_prices.items():
            if sku_id in env.skus_id_map:
                env.skus_id_map[sku_id].set_price(price)
        
        # 调用各个 manager 的 recover_from_checkpoint 方法
        env.inventory.recover_from_checkpoint(checkpoint_data, env.skus_id_map)
        env.order_manager.recover_from_checkpoint(checkpoint_data, env.skus_id_map)
        
        # 恢复 news_manager 状态（如果启用）
        if env.news_manager and env.config.get("enable_new"):
            env.news_manager.recover_from_checkpoint(checkpoint_data)
        
        # 恢复 record_manager 数据库（从 SQL 文件）
        record_manager_sql_path = checkpoint_data.get("record_manager_sql_path")
        if record_manager_sql_path:
            sql_file_path = checkpoint_path.parent / record_manager_sql_path
            if sql_file_path.exists():
                env.record_manager.restore_from_sql(sql_file_path)
                env.logger.info(f"Record manager database restored from {sql_file_path}")
            else:
                env.logger.warning(f"Record manager SQL file not found: {sql_file_path}")
        
        env.logger.info(f"Environment recovered from checkpoint {checkpoint_path}")
        return env
        

    # ---------- MCP 工具声明 ----------
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        tools: Dict[str, Dict[str, Any]] = {
            "place_order": {
                "name": "place_order",
                "description": "Place an order: multiple SKUs supported; supplier_id required.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku_id": {"type": "string", "description": "SKU to purchase"},
                                    "quantity": {"type": "integer", "minimum": 1, "description": "Units to order"},
                                },
                                "required": ["sku_id", "quantity"],
                            },
                            "description": "Order lines array",
                        },
                        "supplier_id": {"type": "string", "description": "Supplier id to use"},
                    },
                    "required": ["items", "supplier_id"],
                },
            },
            "view_current_orders": {
                "name": "view_current_orders",
                "description": "List all open (not yet delivered) orders.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "view_sku_sales_history": {
                "name": "view_sku_sales_history",
                "description": "View historical sales for SKUs with optional date filters.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of SKU ids to query",
                        },
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            },
            "view_current_date_supplier_prices": {
                "name": "view_current_date_supplier_prices",
                "description": "View supplier quotes for the current date.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of SKU ids to check",
                        },
                    },
                    "required": ["sku_ids"],
                },
            },
            "view_supplier_price_history": {
                "name": "view_supplier_price_history",
                "description": "Query historical supplier price records with optional filters.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "supplier_id": {"type": "string", "description": "Optional supplier id"},
                        "sku_id": {"type": "string", "description": "Optional SKU id"},
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                    },
                    "required": ["supplier_id", "sku_id", "start_date", "end_date"],
                },
            },
            "view_inventory": {
                "name": "view_inventory",
                "description": "View current inventory and quantities.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "view_funds_and_date": {
                "name": "view_funds_and_date",
                "description": "View current date and funds balance.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "view_sku_prices": {
                "name": "view_sku_prices",
                "description": "View current prices for specified SKUs (all by default).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of SKU ids; omit to return all",
                        }
                    },
                    "required": ["sku_ids"],
                },
            },
            "modify_sku_price": {
                "name": "modify_sku_price",
                "description": "Update the selling price of a SKU.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_id": {"type": "string", "description": "SKU id to update"},
                        "new_price": {"type": "number", "description": "New price"},
                    },
                    "required": ["sku_id", "new_price"],
                },
            },
            "end_today": {
                "name": "end_today",
                "description": "End today's operations and advance the store to the next day (calls step).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            "add_note": {
                "name": "add_note",
                "description": "Add a note to the store's note system for recording important information, observations, or reminders.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the note to add.",
                        },
                    },
                    "required": ["content"],
                },
            },
            "view_notes": {
                "name": "view_notes",
                "description": "View all notes stored in the system.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        }

        enable_news = self.config.get("enable_new", False)

        if enable_news:
            
            tools["view_today_news"] = {
                "name": "view_today_news",
                "description": "View today's news list (title + id).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }

            tools["view_news_detail"] = {
                "name": "view_news_detail",
                "description": "View a specific news detail by id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "news_id": {"type": "string", "description": "News id to query"},
                    },
                    "required": ["news_id"],
                },
            }

            tools["view_news_history"] = {
                "name": "view_news_history",
                "description": "View news in a date range (inclusive).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    },
                    "required": ["start_date", "end_date"],
                },
            }

        enable_review = self.config.get("enable_review", False)

        if enable_review:
            tools["view_sku_reviews"] = {
                "name": "view_sku_reviews",
                "description": "Fetch reviews for SKUs within date range; optionally filter by rating list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "SKU 列表",
                        },
                        "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD 或 MM/DD/YY"},
                        "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD 或 MM/DD/YY"},
                        "ratings": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 5},
                            "default": [1,2,3,4,5],
                            "description": "Optional list of ratings to include (e.g., [4,5])",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 20,
                            "description": "Maximum number of reviews to return. Defaults to 20."
                        },
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            }
            tools["view_sku_avg_ratings"] = {
                "name": "view_sku_avg_ratings",
                "description": "Get average rating for SKUs within an optional date range.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "SKU 列表",
                        },
                        "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD 或 MM/DD/YY"},
                        "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD 或 MM/DD/YY"},
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            }

            tools["view_return_rates"] = {
                "name": "view_return_rates",
                "description": "View return rate records for SKUs in a date range.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sku_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of SKU ids",
                        },
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD or MM/DD/YY"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD or MM/DD/YY"},
                    },
                    "required": ["sku_ids", "start_date", "end_date"],
                },
            }


        # 兼容旧的 OpenAI tool schema
        for meta in tools.values():
            meta.setdefault("parameters", meta["input_schema"])
        return tools
    

    def exec_tools(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        dispatch = {
            "place_order": self.place_order,
            "modify_sku_price": self.modify_sku_price,
            "end_today": self.end_of_day,
            "view_current_orders": self.view_current_orders,
            "view_sku_sales_history": self.view_sku_sales_history,
            "view_current_date_supplier_prices": self.view_current_date_supplier_prices,
            "view_supplier_price_history": self.view_supplier_price_history,
            "view_inventory": self.view_inventory,
            "view_sku_reviews": self.view_sku_reviews,
            "view_sku_avg_ratings": self.view_sku_avg_ratings,
            "view_return_rates": self.view_return_rates,
            "view_returns": self.view_returns,
            "view_funds_and_date": self.view_funds_and_date,
            "view_sku_prices": self.view_sku_prices,
            "view_today_news": self.view_today_news,
            "view_news_detail": self.view_news_detail,
            "view_news_history": self.view_news_history,
            "add_note": self.add_note,
            "view_notes": self.view_notes,
        }
        # 记录开始时间
        start_time = time.time()
        
        try:
            if tool_name not in dispatch:
                raise ValueError(f"未知工具：{tool_name}")
            
            result = dispatch[tool_name](**kwargs)
            
            # 验证返回格式
            if not isinstance(result, dict) or "result" not in result or "formatted" not in result:
                self.logger.warning(f"Tool {tool_name} returned unexpected format: {type(result)}")
                # 尝试包装结果
                if isinstance(result, dict):
                    result = self._wrap_tool_result(result, str(result))
                else:
                    result = self._wrap_tool_result({"value": result}, str(result))
            
            # Limit the length of formatted field, maximum 50k characters
            MAX_FORMATTED_LENGTH = 50000
            if isinstance(result, dict) and "formatted" in result:
                formatted_str = result.get("formatted", "")
                if isinstance(formatted_str, str) and len(formatted_str) > MAX_FORMATTED_LENGTH:
                    truncated = formatted_str[:MAX_FORMATTED_LENGTH]
                    result["formatted"] = (
                        truncated + 
                        f"\n\n[Content truncated: original length {len(formatted_str)} characters, truncated to {MAX_FORMATTED_LENGTH} characters]"
                    )
                    self.logger.warning(
                        f"Tool {tool_name} returned formatted string too long ({len(formatted_str)} chars), "
                        f"truncated to {MAX_FORMATTED_LENGTH} chars"
                    )
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = f"Error executing {tool_name}: {type(e).__name__}: {e}"
            result = self._wrap_tool_result({"error": err_msg}, err_msg)
        finally:
            # 计算耗时
            pass
            

        # Make the payload JSON safe for downstream logging
        if isinstance(result, dict) and "result" in result:
            result["result"] = self._json_safe(result.get("result"))

        self._log_tool_call(tool_name, kwargs, result, time.time() - start_time)
        elapsed_time = time.time() - start_time
        self.logger.debug(f"[工具耗时] {tool_name}: {elapsed_time:.4f} 秒")
            
        # 在调试模式下输出耗时信息
        # self.logger.info(f"[工具耗时] {tool_name}: {elapsed_time:.4f} 秒")
        
        return result
    

from datetime import date
from retail_environment import RetailEnvironment  # 注意根据你的文件名调整导入


def _snapshot_env_state_for_test(env: RetailEnvironment) -> Dict[str, Any]:
    """
    对环境做一次深度快照，用于和 checkpoint 恢复后的环境做对比。
    只包含可序列化的基础类型，方便直接做 == 比较。
    """
    state: Dict[str, Any] = {}

    # 基本信息
    state["funds"] = round(float(env.funds), 6)
    state["current_date"] = env.current_date.isoformat()

    # SKU 价格
    sku_prices: Dict[str, float] = {}
    for sku_id, sku in env.skus_id_map.items():
        sku_prices[sku_id] = round(float(sku.price), 6)
    state["sku_prices"] = sku_prices

    # 库存（items_by_sku + waiting_items）
    inv_snapshot: Dict[str, Any] = {"capacity": env.inventory.capacity, "items_by_sku": {}, "waiting_items": []}

    for sku_id, items in env.inventory.items_by_sku.items():
        items_data = []
        for m in items:
            items_data.append(
                {
                    "sku_id": m.sku.sku_id,
                    "begin_time": m.begin_time.isoformat(),
                    "expired_time": m.expired_time.isoformat(),
                    "buy_price": round(float(m.buy_price), 6),
                    "merch_id": m.merch_id,
                    "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                    "supplier_id": getattr(m, "supplier_id", None),
                }
            )
        # 按 merch_id 排序，避免顺序差异
        items_data.sort(key=lambda x: x["merch_id"])
        inv_snapshot["items_by_sku"][sku_id] = items_data

    waiting_data = []
    for m in env.inventory.waiting_items:
        waiting_data.append(
            {
                "sku_id": m.sku.sku_id,
                "begin_time": m.begin_time.isoformat(),
                "expired_time": m.expired_time.isoformat(),
                "buy_price": round(float(m.buy_price), 6),
                "merch_id": m.merch_id,
                "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                "supplier_id": getattr(m, "supplier_id", None),
            }
        )
    waiting_data.sort(key=lambda x: x["merch_id"])
    inv_snapshot["waiting_items"] = waiting_data
    state["inventory"] = inv_snapshot

    # 订单
    orders_snapshot: Dict[str, Any] = {}
    for order_id, order in env.order_manager.orders.items():
        items_data = []
        for m in order.items:
            items_data.append(
                {
                    "sku_id": m.sku.sku_id,
                    "begin_time": m.begin_time.isoformat(),
                    "expired_time": m.expired_time.isoformat(),
                    "buy_price": round(float(m.buy_price), 6),
                    "merch_id": m.merch_id,
                    "quality_score": round(float(getattr(m, "quality_score", 0.0)), 6),
                    "supplier_id": getattr(m, "supplier_id", None),
                }
            )
        items_data.sort(key=lambda x: x["merch_id"])

        ordered_sku_dict = {sku.sku_id: qty for sku, qty in order.ordered_sku.items()}

        orders_snapshot[order_id] = {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "delivery_time": order.delivery_time,
            "cost": round(float(order.cost), 6),
            "items": items_data,
            "ordered_sku": ordered_sku_dict,
        }
    state["orders"] = orders_snapshot

    # 从 record_manager 中抽样一个 SKU 的销售记录快照（如果有 SKU）
    if env.skus_list:
        sample_sku_id = env.skus_list[0].sku_id
        # 取从 data_begin_time 到 current_date 的区间
        cfg_begin = env.config.get("data_begin_time") or env.config.get("begin_time")
        try:
            start_dt = datetime.strptime(cfg_begin, "%m/%d/%y").date()
        except Exception:
            start_dt = env.current_date
        sales = env.record_manager.read_sku(sample_sku_id, start_dt, env.current_date)
        # 按日期聚合 (move_sum, last_price, last_customer_count)
        sales_snapshot: Dict[str, Any] = {}
        for day, recs in sales.items():
            move_sum = sum(r.move for r in recs)
            last = recs[-1]
            sales_snapshot[day] = {
                "total_move": int(move_sum),
                "last_price": round(float(last.price), 6),
                "last_customer_count": int(last.customer_count) if last.customer_count is not None else None,
                "records": len(recs),
            }
        state["sample_sales"] = {"sku_id": sample_sku_id, "by_day": sales_snapshot}

    # 新闻管理器（如启用）
    if getattr(env, "news_manager", None) and env.config.get("enable_new"):
        today_news = env.news_manager.get_today_news()
        rolling = getattr(env.news_manager, "_rolling", [])
        state["news_manager"] = {
            "today_news_count": len(today_news),
            "rolling_count": len(rolling),
        }

    return state


def run_all_tools_once(env = None):
    if env is None:
        config = create_default_config()
        env = RetailEnvironment(config)

    ratings = env.get_sku_rating_report()

    if ratings:
        for rating, rating_item in ratings.items():
            env.log_debug(
                f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
            )
    tools = env.get_tools()
    env.log_debug(f"发现工具 {len(tools)} 个：{list(tools.keys())}")

    skus = env.skus_id_map
    env.log_debug(f"发现 SKU {len(skus)} 个：{list(skus.keys())}")

    # 先准备一个可用的 sku_id（如果有的话）
    sku_id_for_test = None
    if env.skus_list:
        sku_id_for_test = env.skus_list[0].sku_id
        env.log_debug(f"用于测试的 sku_id: {sku_id_for_test}")
    else:
        env.log_debug("警告：当前环境中没有任何 SKU，相关工具会返回报错信息。")

    env.log_debug("\n===== 开始逐个执行工具 =====\n")

    for tool_name in tools.keys():
        env.log_debug(f"\n---- 执行工具：{tool_name} ----")

        try:
            # 根据不同工具准备参数
            if tool_name == "view_funds_and_date":
                result = env.exec_tools(tool_name)

            elif tool_name == "view_sku_sales_history":
                # 需要 sku_ids
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_ids=target_skus,
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_current_date_supplier_prices":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(tool_name, sku_ids=target_skus)

            elif tool_name == "view_sku_prices":
                target_skus = [sku_id_for_test] if sku_id_for_test else []
                result = env.exec_tools(tool_name, sku_ids=target_skus if target_skus else None)

            elif tool_name == "view_supplier_price_history":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_id=target_skus[0],
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_sku_reviews":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_ids=target_skus,
                    ratings=[4, 5],
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_sku_avg_ratings":
                target_skus = [sku_id_for_test] if sku_id_for_test else ["DUMMY_SKU"]
                result = env.exec_tools(
                    tool_name,
                    sku_ids=target_skus,
                    start_date="06/06/91",
                    end_date="06/08/91",
                )

            elif tool_name == "view_current_orders":
                result = env.exec_tools(tool_name)
                

            elif tool_name == "view_inventory":
                result = env.exec_tools(tool_name)

            elif tool_name == "place_order":
                # 需要 sku_id 和 quantity
                result = env.exec_tools(
                    tool_name,
                    items=[{"sku_id": sku_id_for_test or "DUMMY_SKU", "quantity": 5}],
                    supplier_id="supplier_1",
                )

            elif tool_name == "modify_sku_price":
                # 需要 sku_id 和 new_price
                new_price = 9.99
                if sku_id_for_test and sku_id_for_test in env.skus_id_map:
                    new_price = env.skus_id_map[sku_id_for_test].price * 1.1
                result = env.exec_tools(
                    tool_name,
                    sku_id=sku_id_for_test or "DUMMY_SKU",
                    new_price=new_price,
                )

            else:
                # 兜底：如果以后新增工具但这里没特殊处理，就尝试裸调
                result = env.exec_tools(tool_name)

            env.log_debug("执行结果：")
            env.log_debug(result)

        except Exception as e:
            env.log_debug(f"工具 {tool_name} 执行时发生异常：{e}")

    env.log_debug("\n===== 所有工具执行完毕 =====")


def simulate_simple_logic_environment(days: int = 7, sample_size: int = 3, config_type: str = "dynamic_hard", db_path: str = None):
    """
    初始化环境后，模拟正常经营若干天：
    - 每天调用 step() 推进一天
    - 如果某个测试 SKU 库存太低，就自动下单补货
    - 打印每天的资金变化、销量情况和订单数
    - 每日展示新闻、供给商报价，并在日终查看记录管理器写入的数据
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
        db_path: 数据库路径（可选）
    """
    # 根据 config_type 选择配置
    if config_type == "dynamic_hard":
        config = create_default_config()
    elif config_type == "dynamic_middle":
        config = create_default_middle_config()
    elif config_type == "still_hard":
        config = create_default_still_hard_config()
    elif config_type == "still_middle":
        config = create_default_still_middle_config()
    else:
        config = create_default_config()  # 默认使用 dynamic_hard
    
    # 设置 order_record_dir
    if db_path is not None:
        config["order_record_dir"] = db_path
    elif "order_record_dir" not in config:
        config["order_record_dir"] = 'model_run_time'
    
    env = RetailEnvironment(config)

    # 为了避免模拟过程中输出大量 debug 日志（尤其是 Attraction 相关），
    # 这里将日志级别降到 INFO，只保留关键信息与耗时统计。
    set_logger_level(False)
    supplier_choice_by_sku: Dict[str, Optional[str]] = {}

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def log_record_manager_snapshot(dt: date) -> None:
        try:
            today_str = dt.isoformat()
            supplier_rows = []
            for sku_id in env.skus_id_map.keys():
                supplier_rows.extend(
                    env.record_manager.read_supplier_prices(
                        sku_id=sku_id,
                        start_date=dt,
                        end_date=dt,
                    )
                )
            env.log_debug(f"记录管理器-当日供给商价格 ({len(supplier_rows)}): {[r.to_dict() for r in supplier_rows]}")

            news_rows = env.exec_tools(
                "view_news_history",
                start_date=today_str,
                end_date=today_str,
            ).get("result", [])
            env.log_debug(f"记录管理器-当日新闻 ({len(news_rows)}): {news_rows[:5]}")

            order_rows = env.record_manager.read_supplier_orders(
                start_order_date=dt,
                end_order_date=dt,
            )
            preview_orders = [
                {"supplier_id": o.supplier_id, "order_date": o.order_date, "arrival_date": o.arrival_date, "items": o.items}
                for o in order_rows[:5]
            ]
            env.log_debug(f"记录管理器-当日订单 ({len(order_rows)}): {preview_orders}")
        except Exception as e:
            env.log_debug(f"读取记录管理器数据失败: {e}")

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营")
        env.log_debug(f"初始资金 & 日期：{env.view_funds_and_date()['formatted']}")
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            # 整天耗时统计
            day_start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            try:
                for sku_id_for_test in sampled_skus:
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test]
                    )
                    env.log_debug(f"供应商报价（{sku_id_for_test}）: {quotes_resp['formatted']}")
            except Exception as e:
                env.log_debug(f"今日供应商价格获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    sku_start_time = time.time()
                    current_quantity = inventory_map.get(sku_id_for_test, {}).get("quantity", 0)
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity}")

                    # 1. 查看历史销售数据
                    t0 = time.time()
                    sales_resp = env.exec_tools(
                        "view_sku_sales_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )
                    t_sales = time.time() - t0

                    env.log_debug(
                        "历史销售数据:" + sales_resp['formatted']
                    )
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 2. 查看当前供货商价格，选择最低
                    t0 = time.time()
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test]
                    )
                    t_quotes = time.time() - t0

                    env.log_debug("供应商报价:" + quotes_resp['formatted'])
                    supplier_quotes = quotes_resp.get("result", {}).get(sku_id_for_test, [])
                    current_supplier = supplier_choice_by_sku.get(sku_id_for_test)
                    chosen_supplier = None
                    chosen_supplier_price = None

                    def pick_supplier(quotes: List[Dict[str, Any]], prefer: Optional[str] = None):
                        if prefer:
                            preferred = next((q for q in quotes if q.get("supplier_id") == prefer), None)
                            if preferred:
                                return preferred.get("supplier_id"), preferred.get("price")
                        if not quotes:
                            return None, None
                        cheapest = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return cheapest.get("supplier_id"), cheapest.get("price")

                    chosen_supplier, chosen_supplier_price = pick_supplier(
                        supplier_quotes,
                        prefer=current_supplier,
                    )

                    if chosen_supplier_price is None:
                        import pdb; pdb.set_trace()
                    if chosen_supplier:
                        supplier_choice_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(f"选择供应商 {chosen_supplier}，报价 {chosen_supplier_price}")
                    else:
                        env.log_debug("未找到可用供应商报价。")

                    # 3. 基于历史销售选择最优价格（总收入最大）
                    t0 = time.time()
                    revenue_by_price: Dict[float, List[int]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    # 若暂无供给商报价，回退使用销售价作为成本避免 None
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else None

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            effective_cost = cost_price if cost_price is not None else price
                            revenue_by_price.setdefault(price, []).append((price - effective_cost) * move)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price = None
                    expected_move = 0
                    if revenue_by_price:
                        best_price = max(revenue_by_price.items(), key=lambda x: sum(x[1]) / len(x[1]))[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = sum(moves) / len(moves) if moves else None
                        env.log_debug(f"历史最优价格 {best_price}，预期销量 {expected_move}")

                    # 4. 修改价格
                    t_price_block = time.time() - t0
                    t0 = time.time()
                    if best_price is not None:
                        res = env.exec_tools("modify_sku_price", sku_id=sku_id_for_test, new_price=best_price)
                        env.log_debug(f"修改价格结果: {res['formatted']}")
                    t_price_change = time.time() - t0

                    # 5. 库存为空则下单，数量为预测销量的 4 倍
                    t0 = time.time()
                    if current_quantity <= expected_move * 4:
                        open_orders_info = env.exec_tools("view_current_orders")
                        open_orders = open_orders_info.get("result", {}).get("orders", []) or []
                        env.log_debug("当前未完成订单：" + open_orders_info['formatted'])
                        has_pending = any(
                            any(item.get("sku_id") == sku_id_for_test for item in (order or {}).get("items", []))
                            for order in open_orders
                        )
                        if has_pending:
                            env.log_debug(f"已有未完成订单包含 {sku_id_for_test}，跳过下单。")
                            continue

                        predicted_sales = expected_move if expected_move is not None else 10
                        order_qty = max(1, int(predicted_sales * 4))
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty}],
                        }
                        if not chosen_supplier:
                            env.log_debug("未选择到供给商，无法下单，跳过。")
                        else:
                            place_kwargs["supplier_id"] = chosen_supplier
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug("触发补货，下单结果：" + order_res['formatted'])
                    t_order_block = time.time() - t0

                    # 单个 SKU 的耗时拆分日志
                    sku_total = time.time() - sku_start_time
                    # 使用 markdown 表格结构化输出单个 SKU 的性能数据，便于在日志中阅读
                    perf_md = (
                        f"### [PERF] Day {day} - SKU `{sku_id_for_test}`\n"
                        f"| 阶段 | 耗时 |\n"
                        f"|---|---|\n"
                        f"| total | `{sku_total:.4f}s` |\n"
                        f"| sales_hist (view_sku_sales_history) | `{t_sales:.4f}s` |\n"
                        f"| quotes (view_current_date_supplier_prices) | `{t_quotes:.4f}s` |\n"
                        f"| pricing (Python 计算最优价格) | `{t_price_block:.4f}s` |\n"
                        f"| price_change (modify_sku_price) | `{t_price_change:.4f}s` |\n"
                        f"| order_logic (open_orders + place_order) | `{t_order_block:.4f}s` |"
                    )
                    env.log_info(perf_md)

                # 3. 推进一天
                end_time = time.time()
                env.log_info(
                    f"第 {day} 天决策逻辑耗时：{end_time - day_start_time:.4f} 秒（不含 end_today）"
                )
                step_res = env.exec_tools("end_today")
                step_payload = step_res.get("result", {})
                env.log_info("step 结果：")
                env.log_info(step_res.get("formatted", ""))
                if step_payload:
                    if step_payload.get("insufficient_skus"):
                        env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")

                ratings = env.get_sku_rating_report()

                if ratings:
                    for rating, rating_item in ratings.items():
                        env.log_debug(
                            f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                        )
                
        # 4. 查看当前订单数量
        orders_info = env.exec_tools("view_current_orders")
        open_orders = orders_info.get("result", {}).get("total_orders")
        env.log_debug(f"当前未完成订单数：{open_orders}")

        # 5. 日终：查看 record_manager 当日写入的供给商价格、新闻、订单
        # log_record_manager_snapshot(env.current_date - timedelta(days=1))

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug(f"最终资金 & 日期：{env.exec_tools('view_funds_and_date')}")
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")

        # ------------ Checkpoint 一致性测试（逻辑环境） ------------
        try:
            import tempfile
            import shutil

            env.log_info("[CheckpointTest-Logic] 开始保存与恢复环境，用于一致性验证...")
            original_state = _snapshot_env_state_for_test(env)

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="simulate_logic_ckpt_"))
            checkpoint_path = checkpoint_dir / "logic_env.json"
            env.save_checkpoint(checkpoint_path)

            recovered_env = RetailEnvironment.recover_from_checkpoint(checkpoint_path)
            recovered_state = _snapshot_env_state_for_test(recovered_env)

            # 基本字段对比
            assert original_state["funds"] == recovered_state["funds"]
            assert original_state["current_date"] == recovered_state["current_date"]
            assert original_state["sku_prices"] == recovered_state["sku_prices"]
            assert original_state["inventory"] == recovered_state["inventory"]
            assert original_state["orders"] == recovered_state["orders"]

            if "sample_sales" in original_state:
                assert original_state["sample_sales"] == recovered_state.get("sample_sales")
            if "news_manager" in original_state:
                assert original_state["news_manager"] == recovered_state.get("news_manager")

            env.log_info("[CheckpointTest-Logic] 环境恢复后一致性验证通过。")
        except AssertionError as e:
            env.log_info(f"[CheckpointTest-Logic] 一致性验证失败: {e}")
        except Exception as e:
            env.log_info(f"[CheckpointTest-Logic] 执行 checkpoint 测试时发生异常: {e}")
        finally:
            try:
                if "checkpoint_dir" in locals():
                    shutil.rmtree(checkpoint_dir)
            except Exception:
                pass
    finally:
        # 结束时清理容器
        pass


def simulate_simple_review_environment(days: int = 7, sample_size: int = 3, db_path: str = 'data/simulate_data/15/records_no_review/', config_type: str = "dynamic_hard"):
    """
    初始化环境后，模拟正常经营若干天：
    目标逻辑：
    1. 第一次：查询历史数据中客户评价最好的供给商，作为每个 SKU 的"首选供给商"；
    2. 每隔 7 天：对当前有报价的每个供给商做一次小量试单，用于更新该供给商的评价认知；
    3. 常规/大量订货：基于当前对各供给商评价水平（平均评分）的认知，选择最优供给商进行大单补货。
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
    """

    print(
        "========================simulate_simple_review_environment =========================="
    )

    # 根据 config_type 选择配置
    if config_type == "dynamic_hard":
        config = create_default_config()
    elif config_type == "dynamic_middle":
        config = create_default_middle_config()
    elif config_type == "still_hard":
        config = create_default_still_hard_config()
    elif config_type == "still_middle":
        config = create_default_still_middle_config()
    else:
        config = create_default_config()  # 默认使用 dynamic_hard
    
    # 设置 order_record_dir
    config["order_record_dir"] = db_path if db_path else 'model_run_time'
    env = RetailEnvironment(config)

    # 记录每个 SKU 当前“认为最优”的供给商
    best_supplier_by_sku: Dict[str, Optional[str]] = {}

    # 探索相关参数
    exploration_interval = 1000        # 每隔多少天做一次小量试单
    bulk_qty_multiplier = 3         # 大量订货：预计销量的倍数
    
    def calculate_exploration_qty(sku_id: str) -> int:
        """
        根据历史30天的平均销售量计算探索订单数量。
        返回：历史30天平均销售量的1/5，至少为1。
        """
        start_date = env.current_date - timedelta(days=30)
        sales = env.record_manager.read_sku(sku_id, start_date, env.current_date)
        
        # 计算总销量
        total_sales = 0
        for day_records in sales.values():
            for rec in day_records:
                total_sales += rec.move
        
        # 计算平均日销量（30天）
        avg_daily_sales = total_sales / 30.0 if total_sales > 0 else 0
        
        # 探索数量 = 平均日销量的1/5，至少为1
        exploration_qty = max(1, int(avg_daily_sales / 5.0))
        
        return exploration_qty

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def compute_supplier_ratings(
        sku_id: str,
        start_dt: date,
        end_dt: date,
    ) -> Dict[str, float]:
        """
        计算某个 SKU 在一段时间内，不同供给商的加权平均评分。
        使用贝叶斯平均方法，根据评论数量进行加权，避免偶发评论造成的影响。
        公式：weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        返回：{supplier_id: weighted_avg_rating}
        """
        reviews = env.review_manager.get_reviews_in_range(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
            ratings=[1, 2, 3, 4, 5],
        )

        rating_sum: Dict[str, float] = {}
        rating_cnt: Dict[str, int] = {}
        for r in reviews:
            sid = getattr(r, "supplier_id", None)
            if not sid or str(sid).upper() == "UNKNOWN":
                continue
            score = float(getattr(r, "rating", 0) or 0)
            rating_sum[sid] = rating_sum.get(sid, 0.0) + score
            rating_cnt[sid] = rating_cnt.get(sid, 0) + 1

        # 计算所有供应商的平均评分作为先验平均值
        all_ratings = [float(getattr(r, "rating", 0) or 0) for r in reviews if getattr(r, "supplier_id", None) and str(getattr(r, "supplier_id", None)).upper() != "UNKNOWN"]
        prior_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0  # 默认中位数
        prior_count = 5  # 先验评论数量，用于平滑
        
        # 使用贝叶斯平均方法计算加权评分
        # weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        # 评论数量越多，实际评分权重越大，避免偶发评论造成的影响
        avg_by_supplier: Dict[str, float] = {}
        for sid, s in rating_sum.items():
            actual_count = rating_cnt.get(sid, 0)
            if actual_count > 0:
                # 贝叶斯加权：评论数量越多，实际评分权重越大
                weighted_avg = (prior_mean * prior_count + s) / (prior_count + actual_count)
                avg_by_supplier[sid] = weighted_avg
        
        # 打印 rating_map (包含原始平均和加权平均)
        raw_avg = {sid: rating_sum[sid] / rating_cnt[sid] for sid in rating_sum.keys() if rating_cnt.get(sid, 0) > 0}
        env.log_debug(f"[compute_supplier_ratings] SKU {sku_id} ({start_dt} ~ {end_dt}):")
        env.log_debug(f"  - Raw avg ratings: {raw_avg}")
        env.log_debug(f"  - Weighted avg ratings (Bayesian, prior_count={prior_count}): {avg_by_supplier}")
        env.log_debug(f"  - Rating counts: {rating_cnt}")
        
        return avg_by_supplier

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        # ---------- 第一次：用历史评价为每个 SKU 选出"首选供给商" ----------
        if sampled_skus:
            begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
            history_start = parse_date_safe(begin_raw)
            history_end = env.current_date
            for sku_id in sampled_skus:
                rating_map = compute_supplier_ratings(sku_id, history_start, history_end)
                if rating_map:
                    best_sid = max(rating_map.items(), key=lambda x: x[1])[0]
                    best_supplier_by_sku[sku_id] = best_sid
                    env.log_debug(
                        f"[初始化首选供给商] SKU {sku_id} 首选 {best_sid}，历史平均评分 {rating_map[best_sid]:.2f}"
                    )

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sku_sales_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=60)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 1.3 根据最近一段时间的评论，为各供给商打分
                    review_window_start = env.current_date - timedelta(days=60)
                    rating_map_recent = compute_supplier_ratings(
                        sku_id_for_test,
                        # review_window_start,
                        parse_date_safe(begin_raw),
                        parse_date_safe(begin_raw) + timedelta(days=90),
                        # env.current_date,
                    )

                    # 如果之前没有首选供给商，这里再尝试用“最近 60 天评论”初始化一次
                    if sku_id_for_test not in best_supplier_by_sku or not best_supplier_by_sku[sku_id_for_test]:
                        if rating_map_recent:
                            best_sid = max(rating_map_recent.items(), key=lambda x: x[1])[0]
                            best_supplier_by_sku[sku_id_for_test] = best_sid
                            env.log_debug(
                                f"[补充首选供给商] SKU {sku_id_for_test} 选择 {best_sid} 作为首选（最近 60 天平均评分 {rating_map_recent[best_sid]:.2f}）"
                            )

                    # 2. 每隔 exploration_interval 天，对所有供给商小量试单
                    if day > 1 and day % exploration_interval == 1:
                        # 根据历史30天平均销售量计算探索数量
                        exploration_qty = calculate_exploration_qty(sku_id_for_test)
                        env.log_debug(f"[探索] 第 {day} 天，对 SKU {sku_id_for_test} 所有供给商做小量试单（数量：{exploration_qty}，基于历史30天平均销售量的1/5）")
                        for quote in supplier_quotes:
                            sid = quote.get("supplier_id")
                            if not sid:
                                continue
                            try:
                                order_res = env.exec_tools(
                                    "place_order",
                                    items=[{"sku_id": sku_id_for_test, "quantity": exploration_qty}],
                                    supplier_id=sid,
                                )
                                env.log_debug(f"  探索下单：供给商 {sid}，数量 {exploration_qty}，结果：{order_res['formatted']}")
                            except Exception as e:
                                env.log_debug(f"  探索下单失败（{sid}）：{e}")

                    # 3. 基于“当前认知的供给商评价”决定大单的供给商
                    chosen_supplier: Optional[str] = None
                    chosen_supplier_price: Optional[float] = None

                    def cheapest_supplier(quotes: List[Dict[str, Any]]) -> tuple:
                        q = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return q.get("supplier_id"), q.get("price")

                    if rating_map_recent:
                        # 有评论数据：按平均评分排序，评分高者优先，若没有价格信息则再按价格兜底
                        def sort_key(entry: Dict[str, Any]) -> tuple:
                            sid = entry.get("supplier_id")
                            rating = rating_map_recent.get(sid, 0.0)
                            return (-rating, entry.get("price", float("inf")))

                        best_entry = sorted(supplier_quotes, key=sort_key)[0]
                        chosen_supplier = best_entry.get("supplier_id")
                        chosen_supplier_price = best_entry.get("price")
                    else:
                        # 没有任何评论数据，则退化为历史初始化的首选供给商 / 最便宜供给商
                        sid_pref = best_supplier_by_sku.get(sku_id_for_test)
                        if sid_pref:
                            preferred = next(
                                (q for q in supplier_quotes if q.get("supplier_id") == sid_pref),
                                None,
                            )
                            if preferred:
                                chosen_supplier = preferred.get("supplier_id")
                                chosen_supplier_price = preferred.get("price")
                        if not chosen_supplier:
                            chosen_supplier, chosen_supplier_price = cheapest_supplier(supplier_quotes)

                    if chosen_supplier:
                        best_supplier_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(
                            f"[大单供给商选择] SKU {sku_id_for_test} 选 {chosen_supplier} 供货，单价 {chosen_supplier_price}"
                        )
                    else:
                        env.log_debug("未能选出大单供给商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 修改零售价
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 6. 库存不足时触发大单补货：数量为预期销量的 bulk_qty_multiplier 倍
                    target_move = expected_move if expected_move > 0 else 10
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_current_orders")
                        open_orders = open_orders_info.get("result", {}).get("orders", []) or []
                        env.log_debug(f"当前未完成订单：{open_orders_info['formatted']}")
                        
                        # 计算未完成订单中该 SKU 的总数量（而不是简单检查是否存在）
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("quantity", 0)
                        
                        # 计算需要补货的数量
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        # Check available capacity before placing the order
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        
                        # Leave a buffer of 100 units
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)  # Adjust order_qty to fit capacity
                            if order_qty <= pending_qty:  # If adjusted order_qty is less than or equal to pending, no new order needed
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        # 如果未完成订单的数量已经足够（大于等于需要补货的数量），则跳过
                        # 否则，即使有试单等小量订单，仍然可以下大单补货
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(f"[大单补货] 触发补货，下单结果：{order_res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))
            if step_payload:
                if step_payload.get("insufficient_skus"):
                    env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")
                
                # 统计 waiting_items（直接从 inventory 获取，避免序列化问题）
                # 注意：step_payload 中的 waiting_items 已被 _json_safe 序列化为字符串，无法访问 sku 属性
                # 因此直接使用 env.inventory.waiting_items，与 _format_step_result 方法保持一致
                waiting_items = env.inventory.waiting_items
                if waiting_items:
                    waiting_by_sku = defaultdict(int)
                    for merch in waiting_items:
                        waiting_by_sku[merch.sku.sku_id] += 1
                    total_waiting = len(waiting_items)
                    env.log_debug(f"  [库存统计] 等待入库商品总数: {total_waiting}")
                    if waiting_by_sku:
                        waiting_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(waiting_by_sku.items()))
                        env.log_debug(f"  [库存统计] 等待入库商品按SKU分布: {waiting_lines}")
                else:
                    env.log_debug(f"  [库存统计] 等待入库商品总数: 0")
                
                # 统计 expired_discount_by_sku
                expired_by_sku = step_payload.get("expired_discount_by_sku", {})
                if expired_by_sku:
                    total_expired = sum(expired_by_sku.values())
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: {total_expired}")
                    expired_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(expired_by_sku.items()))
                    env.log_debug(f"  [库存统计] 过期清仓商品按SKU分布: {expired_lines}")
                else:
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: 0")
                
                # 统计 returns_by_sku
                returns_by_sku = step_payload.get("returns_by_sku", {})
                if returns_by_sku:
                    total_returns = sum(returns_by_sku.values())
                    env.log_debug(f"  [库存统计] 退货商品总数: {total_returns}")
                    return_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(returns_by_sku.items()))
                    env.log_debug(f"  [库存统计] 退货商品按SKU分布: {return_lines}")
                else:
                    env.log_debug(f"  [库存统计] 退货商品总数: 0")

            ratings = env.get_sku_rating_report()

            if ratings:
                for rating, rating_item in ratings.items():
                    env.log_debug(
                        f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                    )

            # 4. 查看当前订单数量
            orders_info = env.exec_tools("view_current_orders")
            open_orders = orders_info.get("result", {}).get("total_orders")
            env.log_debug(f"当前未完成订单数：{open_orders}")

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug("最终资金 & 日期：" + env.exec_tools("view_funds_and_date")['formatted'])
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")

        # ------------ Checkpoint 一致性测试（评价环境） ------------
        try:
            import tempfile
            import shutil

            env.log_info("[CheckpointTest-Review] 开始保存与恢复环境，用于一致性验证...")
            original_state = _snapshot_env_state_for_test(env)

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="simulate_review_ckpt_"))
            checkpoint_path = checkpoint_dir / "review_env.json"
            env.save_checkpoint(checkpoint_path)

            recovered_env = RetailEnvironment.recover_from_checkpoint(checkpoint_path)
            recovered_state = _snapshot_env_state_for_test(recovered_env)

            # 基本字段对比
            assert original_state["funds"] == recovered_state["funds"]
            assert original_state["current_date"] == recovered_state["current_date"]
            assert original_state["sku_prices"] == recovered_state["sku_prices"]
            assert original_state["inventory"] == recovered_state["inventory"]
            assert original_state["orders"] == recovered_state["orders"]

            if "sample_sales" in original_state:
                assert original_state["sample_sales"] == recovered_state.get("sample_sales")
            if "news_manager" in original_state:
                assert original_state["news_manager"] == recovered_state.get("news_manager")

            env.log_info("[CheckpointTest-Review] 环境恢复后一致性验证通过。")
        except AssertionError as e:
            env.log_info(f"[CheckpointTest-Review] 一致性验证失败: {e}")
        except Exception as e:
            env.log_info(f"[CheckpointTest-Review] 执行 checkpoint 测试时发生异常: {e}")
        finally:
            try:
                if "checkpoint_dir" in locals():
                    shutil.rmtree(checkpoint_dir)
            except Exception:
                pass
    finally:
        # 结束时清理容器
        pass


def simulate_news_aware_environment(days: int = 7, sample_size: int = 3, db_path: str = 'data/simulate_data/15/records_no_review/', config_type: str = "dynamic_hard"):
    """
    基于 simulate_simple_review_environment，但加入新闻影响的决策优化：
    1. 启用新闻模块，从新闻数据中获取对各 SKU 的影响程度
    2. 在计算预期销量时，结合新闻影响调整需求预测
    3. 在补货决策时，使用新闻调整后的预期销量来优化订货量
    4. 保持原有的供给商选择和评价逻辑
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
    """

    print(
        "========================simulate_news_aware_environment =========================="
    )
    # 根据 config_type 选择配置，并确保启用新闻模块用于决策优化
    if config_type == "dynamic_hard":
        config = create_default_config()
    elif config_type == "dynamic_middle":
        config = create_default_middle_config()
    elif config_type == "still_hard":
        config = create_default_still_hard_config()
    elif config_type == "still_middle":
        config = create_default_still_middle_config()
    else:
        config = create_default_config()  # 默认使用 dynamic_hard
    
    # 设置 order_record_dir
    config["order_record_dir"] = db_path if db_path else 'model_run_time'
    env = RetailEnvironment(config)

    # 记录每个 SKU 当前"认为最优"的供给商
    best_supplier_by_sku: Dict[str, Optional[str]] = {}

    # 探索相关参数
    exploration_interval = 7        # 每隔多少天做一次小量试单
    bulk_qty_multiplier = 4       # 大量订货：预计销量的倍数
    
    # 追踪新闻影响倍数：用于统计整个模拟过程中的比例
    # 结构：{day: {sku_id: {"need_mult": float, "supply_mult": float, "ratio": float}}}
    multiplier_tracking: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    def calculate_exploration_qty(sku_id: str) -> int:
        """
        根据历史30天的平均销售量计算探索订单数量。
        返回：历史30天平均销售量的1/5，至少为1。
        """
        start_date = env.current_date - timedelta(days=30)
        sales = env.record_manager.read_sku(sku_id, start_date, env.current_date)
        
        # 计算总销量
        total_sales = 0
        for day_records in sales.values():
            for rec in day_records:
                total_sales += rec.move
        
        # 计算平均日销量（30天）
        avg_daily_sales = total_sales / 30.0 if total_sales > 0 else 0
        
        # 探索数量 = 平均日销量的1/5，至少为1
        exploration_qty = max(1, int(avg_daily_sales / 5.0))
        
        return exploration_qty

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        # 使用当前日期作为窗口上限
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:  # 补足
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def compute_supplier_ratings(
        sku_id: str,
        start_dt: date,
        end_dt: date,
    ) -> Dict[str, float]:
        """
        计算某个 SKU 在一段时间内，不同供给商的加权平均评分。
        使用贝叶斯平均方法，根据评论数量进行加权，避免偶发评论造成的影响。
        公式：weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        返回：{supplier_id: weighted_avg_rating}
        """
        reviews = env.review_manager.get_reviews_in_range(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
            ratings=[1, 2, 3, 4, 5],
        )

        rating_sum: Dict[str, float] = {}
        rating_cnt: Dict[str, int] = {}
        for r in reviews:
            sid = getattr(r, "supplier_id", None)
            if not sid or str(sid).upper() == "UNKNOWN":
                continue
            score = float(getattr(r, "rating", 0) or 0)
            rating_sum[sid] = rating_sum.get(sid, 0.0) + score
            rating_cnt[sid] = rating_cnt.get(sid, 0) + 1

        # 计算所有供应商的平均评分作为先验平均值
        all_ratings = [float(getattr(r, "rating", 0) or 0) for r in reviews if getattr(r, "supplier_id", None) and str(getattr(r, "supplier_id", None)).upper() != "UNKNOWN"]
        prior_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0  # 默认中位数
        prior_count = 5  # 先验评论数量，用于平滑
        
        # 使用贝叶斯平均方法计算加权评分
        # weighted_rating = (prior_mean * prior_count + actual_sum) / (prior_count + actual_count)
        # 评论数量越多，实际评分权重越大，避免偶发评论造成的影响
        avg_by_supplier: Dict[str, float] = {}
        for sid, s in rating_sum.items():
            actual_count = rating_cnt.get(sid, 0)
            if actual_count > 0:
                # 贝叶斯加权：评论数量越多，实际评分权重越大
                weighted_avg = (prior_mean * prior_count + s) / (prior_count + actual_count)
                avg_by_supplier[sid] = weighted_avg
        
        # 打印 rating_map (包含原始平均和加权平均)
        raw_avg = {sid: rating_sum[sid] / rating_cnt[sid] for sid in rating_sum.keys() if rating_cnt.get(sid, 0) > 0}
        env.log_debug(f"[compute_supplier_ratings] SKU {sku_id} ({start_dt} ~ {end_dt}):")
        env.log_debug(f"  - Raw avg ratings: {raw_avg}")
        env.log_debug(f"  - Weighted avg ratings (Bayesian, prior_count={prior_count}): {avg_by_supplier}")
        env.log_debug(f"  - Rating counts: {rating_cnt}")
        
        return avg_by_supplier

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营（新闻感知版本）")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        # ---------- 第一次：用历史评价为每个 SKU 选出"首选供给商" ----------
        if sampled_skus:
            begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
            history_start = parse_date_safe(begin_raw)
            history_end = env.current_date
            for sku_id in sampled_skus:
                rating_map = compute_supplier_ratings(sku_id, history_start, history_end)
                if rating_map:
                    best_sid = max(rating_map.items(), key=lambda x: x[1])[0]
                    best_supplier_by_sku[sku_id] = best_sid
                    env.log_debug(
                        f"[初始化首选供给商] SKU {sku_id} 首选 {best_sid}，历史平均评分 {rating_map[best_sid]:.2f}"
                    )

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0  # 重置计数器
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻（如果启用）
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sku_sales_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 1.3 根据最近一段时间的评论，为各供给商打分
                    review_window_start = env.current_date - timedelta(days=60)
                    rating_map_recent = compute_supplier_ratings(
                        sku_id_for_test,
                        review_window_start,
                        env.current_date,
                    )

                    # 如果之前没有首选供给商，这里再尝试用"最近 60 天评论"初始化一次
                    if sku_id_for_test not in best_supplier_by_sku or not best_supplier_by_sku[sku_id_for_test]:
                        if rating_map_recent:
                            best_sid = max(rating_map_recent.items(), key=lambda x: x[1])[0]
                            best_supplier_by_sku[sku_id_for_test] = best_sid
                            env.log_debug(
                                f"[补充首选供给商] SKU {sku_id_for_test} 选择 {best_sid} 作为首选（最近 60 天平均评分 {rating_map_recent[best_sid]:.2f}）"
                            )

                    # 2. 每隔 exploration_interval 天，对所有供给商小量试单
                    if day % exploration_interval == 1:
                        # 根据历史30天平均销售量计算探索数量
                        exploration_qty = calculate_exploration_qty(sku_id_for_test)
                        env.log_debug(f"[探索] 第 {day} 天，对 SKU {sku_id_for_test} 所有供给商做小量试单（数量：{exploration_qty}，基于历史30天平均销售量的1/5）")
                        for quote in supplier_quotes:
                            sid = quote.get("supplier_id")
                            if not sid:
                                continue
                            try:
                                order_res = env.exec_tools(
                                    "place_order",
                                    items=[{"sku_id": sku_id_for_test, "quantity": exploration_qty}],
                                    supplier_id=sid,
                                )
                                env.log_debug(f"  探索下单：供给商 {sid}，数量 {exploration_qty}，结果：{order_res['formatted']}")
                            except Exception as e:
                                env.log_debug(f"  探索下单失败（{sid}）：{e}")

                    # 3. 基于"当前认知的供给商评价"决定大单的供给商
                    chosen_supplier: Optional[str] = None
                    chosen_supplier_price: Optional[float] = None

                    def cheapest_supplier(quotes: List[Dict[str, Any]]) -> tuple:
                        q = min(quotes, key=lambda x: x.get("price", float("inf")))
                        return q.get("supplier_id"), q.get("price")

                    if rating_map_recent:
                        # 有评论数据：按平均评分排序，评分高者优先，若没有价格信息则再按价格兜底
                        def sort_key(entry: Dict[str, Any]) -> tuple:
                            sid = entry.get("supplier_id")
                            rating = rating_map_recent.get(sid, 0.0)
                            return (-rating, entry.get("price", float("inf")))

                        best_entry = sorted(supplier_quotes, key=sort_key)[0]
                        chosen_supplier = best_entry.get("supplier_id")
                        chosen_supplier_price = best_entry.get("price")
                    else:
                        # 没有任何评论数据，则退化为历史初始化的首选供给商 / 最便宜供给商
                        sid_pref = best_supplier_by_sku.get(sku_id_for_test)
                        if sid_pref:
                            preferred = next(
                                (q for q in supplier_quotes if q.get("supplier_id") == sid_pref),
                                None,
                            )
                            if preferred:
                                chosen_supplier = preferred.get("supplier_id")
                                chosen_supplier_price = preferred.get("price")
                        if not chosen_supplier:
                            chosen_supplier, chosen_supplier_price = cheapest_supplier(supplier_quotes)

                    if chosen_supplier:
                        best_supplier_by_sku[sku_id_for_test] = chosen_supplier
                        env.log_debug(
                            f"[大单供给商选择] SKU {sku_id_for_test} 选 {chosen_supplier} 供货，单价 {chosen_supplier_price}"
                        )
                    else:
                        env.log_debug("未能选出大单供给商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 结合新闻影响 & 供给变化，调整对未来需求和订货策略的预估
                    need_effect = 0.0
                    supply_effect = 0.0
                    if env.config.get("enable_new") and getattr(env, "news_manager", None):
                        try:
                            sku_obj = env.skus_id_map.get(sku_id_for_test)
                            sku_category = getattr(sku_obj, "category", None) if sku_obj else None

                            # 5.1 需求侧：impact_factor = "need"
                            need_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["need"],
                            )
                            need_effect = float(need_info.get("total_effect", 0.0) or 0.0)

                            # 5.2 供给侧：impact_factor = "supply"
                            supply_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["supply"],
                            )
                            supply_effect = float(supply_info.get("total_effect", 0.0) or 0.0)

                            matched_need = need_info.get("matched_news", []) or []
                            matched_supply = supply_info.get("matched_news", []) or []
                            if matched_need or matched_supply:
                                env.log_debug(
                                    f"[NewsImpact] SKU {sku_id_for_test} need_news={len(matched_need)}, "
                                    f"supply_news={len(matched_supply)}, "
                                    f"need_effect={need_effect:.4f}, supply_effect={supply_effect:.4f}"
                                )
                        except Exception as e:
                            env.log_debug(f"[NewsImpact] 计算 SKU {sku_id_for_test} 新闻影响失败: {e}")

                    # 基于历史销量 + 新闻影响得到"有效预期销量"：
                    # - expected_move 为历史均值
                    # - need_effect 为需求侧的相对提升/降低系数（已按 config 加权）
                    # - supply_effect 为供给侧的影响：> 0 表示供给条件变好（更容易/更便宜拿货），应增加订货量
                    base_expected = expected_move if expected_move > 0 else 10.0
                    # 需求侧倍数：news_effect 可能是正负值，例如 0.1 表示需求提升 10%
                    # 限制在合理范围内：0.5 ~ 2.0，避免过度放大或缩小
                    need_multiplier = 1.0 + need_effect
                    # 供给侧倍数：supply_effect > 0 代表供给条件变好（更容易/更便宜拿货），应增加订货量
                    # 例如 supply_effect = 0.2 -> multiplier ≈ 1.2，表示可以多进 20% 的货
                    # 限制在合理范围内：0.5 ~ 2.0
                    supply_multiplier = 1.0 + supply_effect
                    # 总倍数 = 需求侧倍数 * 供给侧倍数，但限制在合理范围内：0.5 ~ 3.0
                    # 避免过度放大（移除之前硬编码的最小值 2）
                    total_multiplier = max(0.5, min(2.0, need_multiplier * supply_multiplier))
                    effective_expected = base_expected * total_multiplier
                    
                    # 计算比例：need_mult / supply_mult（如果 supply_mult > 0）
                    ratio = need_multiplier / supply_multiplier if supply_multiplier > 0 else float('inf')
                    
                    # 记录到追踪数据结构中
                    if day not in multiplier_tracking:
                        multiplier_tracking[day] = {}
                    multiplier_tracking[day][sku_id_for_test] = {
                        "need_mult": need_multiplier,
                        "supply_mult": supply_multiplier,
                        "ratio": ratio,
                        "need_effect": need_effect,
                        "supply_effect": supply_effect,
                    }
                    
                    env.log_debug(
                        f"[NewsImpact] SKU {sku_id_for_test} base_expected={base_expected:.2f}, "
                        f"need_effect={need_effect:.4f}, need_mult={need_multiplier:.3f}, "
                        f"supply_effect={supply_effect:.4f}, supply_mult={supply_multiplier:.3f}, "
                        f"ratio(need/supply)={ratio:.3f}, total_mult={total_multiplier:.3f}, effective_expected={effective_expected:.2f}"
                    )

                    # 6. 修改零售价（可选：也可以根据新闻影响微调价格）
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 7. 库存不足时触发大单补货：数量为"新闻修正后预期销量"的 bulk_qty_multiplier 倍
                    target_move = effective_expected
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_current_orders")
                        open_orders = open_orders_info.get("result", {}).get("orders", []) or []
                        env.log_debug(f"当前未完成订单：{open_orders_info['formatted']}")
                        
                        # 计算未完成订单中该 SKU 的总数量（而不是简单检查是否存在）
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("quantity", 0)
                        
                        # 计算需要补货的数量（基于新闻修正后的预期销量）
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        # Check available capacity before placing the order
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        
                        # Leave a buffer of 100 units
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)  # Adjust order_qty to fit capacity
                            if order_qty <= pending_qty:  # If adjusted order_qty is less than or equal to pending, no new order needed
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        # 如果未完成订单的数量已经足够（大于等于需要补货的数量），则跳过
                        # 否则，即使有试单等小量订单，仍然可以下大单补货
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(
                                f"[大单补货-新闻感知] 触发补货，下单结果：{order_res['formatted']} "
                                f"(基于新闻修正后的预期销量: {effective_expected:.2f})"
                            )
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))
            if step_payload:
                if step_payload.get("insufficient_skus"):
                    env.log_debug(f"  当日缺货 SKU：{step_payload['insufficient_skus']}")
                
                # 统计 waiting_items（直接从 inventory 获取，避免序列化问题）
                # 注意：step_payload 中的 waiting_items 已被 _json_safe 序列化为字符串，无法访问 sku 属性
                # 因此直接使用 env.inventory.waiting_items，与 _format_step_result 方法保持一致
                waiting_items = env.inventory.waiting_items
                if waiting_items:
                    waiting_by_sku = defaultdict(int)
                    for merch in waiting_items:
                        waiting_by_sku[merch.sku.sku_id] += 1
                    total_waiting = len(waiting_items)
                    env.log_debug(f"  [库存统计] 等待入库商品总数: {total_waiting}")
                    if waiting_by_sku:
                        waiting_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(waiting_by_sku.items()))
                        env.log_debug(f"  [库存统计] 等待入库商品按SKU分布: {waiting_lines}")
                else:
                    env.log_debug(f"  [库存统计] 等待入库商品总数: 0")
                
                # 统计 expired_discount_by_sku
                expired_by_sku = step_payload.get("expired_discount_by_sku", {})
                if expired_by_sku:
                    total_expired = sum(expired_by_sku.values())
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: {total_expired}")
                    expired_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(expired_by_sku.items()))
                    env.log_debug(f"  [库存统计] 过期清仓商品按SKU分布: {expired_lines}")
                else:
                    env.log_debug(f"  [库存统计] 过期清仓商品总数: 0")
                
                # 统计 returns_by_sku
                returns_by_sku = step_payload.get("returns_by_sku", {})
                if returns_by_sku:
                    total_returns = sum(returns_by_sku.values())
                    env.log_debug(f"  [库存统计] 退货商品总数: {total_returns}")
                    return_lines = ", ".join(f"{sku_id}={count}" for sku_id, count in sorted(returns_by_sku.items()))
                    env.log_debug(f"  [库存统计] 退货商品按SKU分布: {return_lines}")
                else:
                    env.log_debug(f"  [库存统计] 退货商品总数: 0")

            ratings = env.get_sku_rating_report()

            if ratings:
                for rating, rating_item in ratings.items():
                    env.log_debug(
                        f"SKU: {rating}, InitialRating: {rating_item['initial_rating']}, AvgRating: {rating_item['avg_rating']}"
                    )

            # 4. 查看当前订单数量
            orders_info = env.exec_tools("view_current_orders")
            open_orders = orders_info.get("result", {}).get("total_orders")
            env.log_debug(f"当前未完成订单数：{open_orders}")

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug("最终资金 & 日期：" + env.exec_tools("view_funds_and_date")['formatted'])
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")
        
        # ========== 输出新闻影响倍数统计 ==========
        env.log_debug("\n" + "="*80)
        env.log_debug("新闻影响倍数统计（整个模拟过程）")
        env.log_debug("="*80)
        
        if multiplier_tracking:
            # 按 SKU 聚合统计
            sku_stats: Dict[str, Dict[str, List[float]]] = {}
            all_need_mults: List[float] = []
            all_supply_mults: List[float] = []
            all_ratios: List[float] = []
            
            for day, sku_data in multiplier_tracking.items():
                for sku_id, data in sku_data.items():
                    if sku_id not in sku_stats:
                        sku_stats[sku_id] = {
                            "need_mults": [],
                            "supply_mults": [],
                            "ratios": [],
                        }
                    sku_stats[sku_id]["need_mults"].append(data["need_mult"])
                    sku_stats[sku_id]["supply_mults"].append(data["supply_mult"])
                    if data["ratio"] != float('inf'):
                        sku_stats[sku_id]["ratios"].append(data["ratio"])
                        all_ratios.append(data["ratio"])
                    
                    all_need_mults.append(data["need_mult"])
                    all_supply_mults.append(data["supply_mult"])
            
            # 输出总体统计
            if all_need_mults and all_supply_mults:
                avg_need_mult = sum(all_need_mults) / len(all_need_mults)
                avg_supply_mult = sum(all_supply_mults) / len(all_supply_mults)
                avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
                min_need_mult = min(all_need_mults)
                max_need_mult = max(all_need_mults)
                min_supply_mult = min(all_supply_mults)
                max_supply_mult = max(all_supply_mults)
                min_ratio = min(all_ratios) if all_ratios else 0.0
                max_ratio = max(all_ratios) if all_ratios else 0.0
                
                env.log_debug(f"\n【总体统计】（共 {len(all_need_mults)} 次计算）")
                env.log_debug(f"  need_multiplier: 平均={avg_need_mult:.4f}, 最小={min_need_mult:.4f}, 最大={max_need_mult:.4f}")
                env.log_debug(f"  supply_multiplier: 平均={avg_supply_mult:.4f}, 最小={min_supply_mult:.4f}, 最大={max_supply_mult:.4f}")
                env.log_debug(f"  ratio(need/supply): 平均={avg_ratio:.4f}, 最小={min_ratio:.4f}, 最大={max_ratio:.4f}")
                env.log_debug(f"  比例分布: need_mult 占主导(ratio>1.2)={sum(1 for r in all_ratios if r > 1.2)}, "
                            f"supply_mult 占主导(ratio<0.8)={sum(1 for r in all_ratios if r < 0.8)}, "
                            f"相对平衡(0.8<=ratio<=1.2)={sum(1 for r in all_ratios if 0.8 <= r <= 1.2)}")
            
            # 按 SKU 输出详细统计
            env.log_debug(f"\n【按 SKU 统计】（共 {len(sku_stats)} 个 SKU）")
            for sku_id in sorted(sku_stats.keys()):
                stats = sku_stats[sku_id]
                if stats["need_mults"]:
                    avg_need = sum(stats["need_mults"]) / len(stats["need_mults"])
                    avg_supply = sum(stats["supply_mults"]) / len(stats["supply_mults"])
                    avg_ratio_sku = sum(stats["ratios"]) / len(stats["ratios"]) if stats["ratios"] else 0.0
                    count = len(stats["need_mults"])
                    env.log_debug(f"  SKU {sku_id}: 计算次数={count}, "
                                f"need_mult_avg={avg_need:.4f}, supply_mult_avg={avg_supply:.4f}, "
                                f"ratio_avg={avg_ratio_sku:.4f}")
        else:
            env.log_debug("未收集到新闻影响倍数数据")
        
        env.log_debug("="*80 + "\n")

        # ------------ Checkpoint 一致性测试（新闻感知环境） ------------
        try:
            import tempfile
            import shutil

            env.log_info("[CheckpointTest-NewsAware] 开始保存与恢复环境，用于一致性验证...")
            original_state = _snapshot_env_state_for_test(env)

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="simulate_news_aware_ckpt_"))
            checkpoint_path = checkpoint_dir / "news_aware_env.json"
            env.save_checkpoint(checkpoint_path)

            recovered_env = RetailEnvironment.recover_from_checkpoint(checkpoint_path)
            recovered_state = _snapshot_env_state_for_test(recovered_env)

            # 基本字段对比
            assert original_state["funds"] == recovered_state["funds"]
            assert original_state["current_date"] == recovered_state["current_date"]
            assert original_state["sku_prices"] == recovered_state["sku_prices"]
            assert original_state["inventory"] == recovered_state["inventory"]
            assert original_state["orders"] == recovered_state["orders"]

            if "sample_sales" in original_state:
                assert original_state["sample_sales"] == recovered_state.get("sample_sales")
            if "news_manager" in original_state:
                assert original_state["news_manager"] == recovered_state.get("news_manager")

            env.log_info("[CheckpointTest-NewsAware] 环境恢复后一致性验证通过。")
        except AssertionError as e:
            env.log_info(f"[CheckpointTest-NewsAware] 一致性验证失败: {e}")
        except Exception as e:
            env.log_info(f"[CheckpointTest-NewsAware] 执行 checkpoint 测试时发生异常: {e}")
        finally:
            try:
                if "checkpoint_dir" in locals():
                    shutil.rmtree(checkpoint_dir)
            except Exception:
                pass
    finally:
        # 结束时清理容器
        pass


def simulate_quality_based_environment(days: int = 7, sample_size: int = 3, db_path: str = 'data/simulate_data/15/records_no_review/', config_type: str = "dynamic_hard"):
    """
    基于 simulate_news_aware_environment，但每次订货时直接选择 quality_score 最高的供应商。
    
    主要特点：
    1. 每次订货时，获取所有供应商的 quality_score
    2. 选择 quality_score 最高的供应商进行订货
    3. 如果多个供应商 quality_score 相同，选择价格更低的作为 tie-breaker
    4. 保持新闻影响的需求预测逻辑
    
    Args:
        days: 模拟天数
        sample_size: 每个品类选择的 SKU 数量
        db_path: 数据库路径
        config_type: 配置类型 ("dynamic_hard", "dynamic_middle", "still_hard", "still_middle")
    """

    print(
        "========================simulate_quality_based_environment =========================="
    )
    # 根据 config_type 选择配置
    if config_type == "dynamic_hard":
        config = create_default_config()
    elif config_type == "dynamic_middle":
        config = create_default_middle_config()
    elif config_type == "still_hard":
        config = create_default_still_hard_config()
    elif config_type == "still_middle":
        config = create_default_still_middle_config()
    else:
        config = create_default_config()  # 默认使用 dynamic_hard
    
    # 设置 order_record_dir
    config["order_record_dir"] = db_path if db_path else 'model_run_time'
    env = RetailEnvironment(config)

    # 订货相关参数
    bulk_qty_multiplier = 4       # 大量订货：预计销量的倍数
    
    # 追踪新闻影响倍数：用于统计整个模拟过程中的比例
    multiplier_tracking: Dict[int, Dict[str, Dict[str, float]]] = {}

    def parse_date_safe(raw: str) -> date:
        try:
            return datetime.strptime(raw, "%m/%d/%y").date()
        except Exception:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except Exception:
                return date.today()

    def avg_profit_for_sku(sku_id: str, start_dt: date, end_dt: date) -> float:
        sales = env.record_manager.read_sku(sku_id, start_dt, end_dt)
        all_records = [rec for recs in sales.values() for rec in recs]
        if not all_records:
            return float("-inf")
        price_rows = env.record_manager.read_supplier_prices(
            sku_id=sku_id,
            start_date=start_dt,
            end_date=end_dt,
        )
        min_price_by_date: Dict[str, float] = {}
        for row in price_rows:
            d = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
            min_price_by_date[d] = min(min_price_by_date.get(d, row.price), row.price)

        total_profit = 0.0
        total_units = 0
        for rec in all_records:
            d = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            cost = min_price_by_date.get(d, rec.price)
            profit_per_unit = rec.price - cost
            total_profit += profit_per_unit * rec.move
            total_units += rec.move

        return total_profit

    def select_skus_per_category(n: int) -> List[str]:
        selected: List[str] = []
        begin_raw = config.get("data_begin_time") or config.get("begin_time", "")
        start_dt = parse_date_safe(begin_raw)
        end_dt = env.current_date

        for category, sku_objs in env.skus_category_map.items():
            scores = []
            for sku_obj in sku_objs:
                score = avg_profit_for_sku(sku_obj.sku_id, start_dt, end_dt)
                scores.append((score, sku_obj.sku_id))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = [sku_id for _, sku_id in scores[:n] if _ != float("-inf")]
            if len(top) < n:
                remaining = [s.sku_id for s in sku_objs if s.sku_id not in top]
                top.extend(remaining[: max(0, n - len(top))])
            selected.extend(top)
        return selected

    def select_supplier_by_quality_score(supplier_quotes: List[Dict[str, Any]], sku_id: str) -> tuple:
        """
        根据 quality_score 选择供应商。
        返回: (supplier_id, price, quality_score) 或 (None, None, None)
        """
        if not supplier_quotes:
            return None, None, None
        
        # 为每个供应商获取 quality_score
        candidates: List[Tuple[float, float, str]] = []  # (quality_score, price, supplier_id)
        
        for quote in supplier_quotes:
            supplier_id = quote.get("supplier_id")
            price = quote.get("price", float("inf"))
            
            if not supplier_id:
                continue
            
            # 获取该供应商的 quality_score
            quality_score = env.supplier_manager.get_quality_score(
                supplier_id=supplier_id,
                sku_id=sku_id,
                target_date=env.current_date,
            )
            
            # 如果没有 quality_score，使用 0.0 作为默认值
            if quality_score is None:
                quality_score = 0.0
            
            candidates.append((quality_score, price, supplier_id))
        
        if not candidates:
            return None, None, None
        
        # 按 quality_score 降序排序，如果 quality_score 相同则按价格升序排序
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        best_quality, best_price, best_supplier = candidates[0]
        
        env.log_debug(
            f"[质量优先供应商选择] SKU {sku_id} 选择供应商 {best_supplier}，"
            f"quality_score={best_quality:.4f}，价格={best_price:.4f}"
        )
        
        return best_supplier, best_price, best_quality

    try:
        # 按品类选择 SKU：每个品类挑选 sample_size 个平均收益最高的 SKU
        if env.skus_category_map:
            sampled_skus = select_skus_per_category(sample_size)
        else:
            sampled_skus = []

        env.log_debug(f"开始模拟 {days} 天经营（质量优先版本）")
        env.log_debug("初始资金 & 日期：" + env.view_funds_and_date()['formatted'])
        env.log_debug(f"测试用 SKU: {sampled_skus}\n")

        consecutive_negative_days = 0  # 跟踪连续负资金天数
        
        for day in range(1, days + 1):
            start_time = time.time()
            
            # 检查资金状态
            if env.funds < 0:
                consecutive_negative_days += 1
                if consecutive_negative_days >= 10:
                    env.log_debug(f"连续 {consecutive_negative_days} 天资金为负数，模拟结束。")
                    break
            else:
                consecutive_negative_days = 0
            
            env.log_debug(f"\n====== 第 {day} 天 ======")

            # 0. 查看当前日期与资金
            funds_snapshot = env.view_funds_and_date()
            env.log_debug(f"资金与日期：{funds_snapshot['formatted']}")

            # 0.0 生成并展示今日新闻（如果启用）
            if env.config.get("enable_new") and env.news_manager:
                try:
                    today_news = env.exec_tools("view_today_news")
                    env.log_debug("今日新闻：")
                    env.log_debug(today_news.get("formatted", ""))
                except Exception as e:
                    env.log_debug(f"今日新闻获取失败: {e}")

            # 1. 查看当前库存
            inv_info = env.view_inventory()
            inv_result = inv_info.get("result", {})
            env.log_debug(f"当前库存: {inv_info['formatted']}")
            inventory_map = inv_result.get("inventory", {})
            if not sampled_skus:
                env.log_debug("没有找到可用的测试 SKU，跳过补货逻辑。")
            else:
                for sku_id_for_test in sampled_skus:
                    # 计算当前库存：包括已入库的商品和等待入库的商品
                    sku_inv_info = inventory_map.get(sku_id_for_test, {})
                    base_quantity = sku_inv_info.get("quantity", 0)
                    waiting_count = sku_inv_info.get("waiting", 0)
                    current_quantity = base_quantity + waiting_count
                    env.log_debug(f"SKU {sku_id_for_test} 当前库存：{current_quantity} (已入库: {base_quantity}, 等待入库: {waiting_count})")

                    # 1.1 查看历史销售数据（用于定价 & 需求估计）
                    sales_resp = env.exec_tools(
                        "view_sku_sales_history",
                        sku_ids=[sku_id_for_test],
                        start_date=(env.current_date - timedelta(days=30)).isoformat(),
                        end_date=env.current_date.isoformat(),
                    )

                    env.log_debug(f"历史销售数据: {sales_resp['formatted']}")
                    sales_records = sales_resp.get("result", {}).get(sku_id_for_test, {}).get("records", {})

                    # 1.2 查看当前所有供给商报价
                    quotes_resp = env.exec_tools(
                        "view_current_date_supplier_prices",
                        sku_ids=[sku_id_for_test],
                    )
                    env.log_debug(f"供应商报价: {quotes_resp['formatted']}")
                    supplier_quotes: List[Dict[str, Any]] = quotes_resp.get("result", {}).get(sku_id_for_test, []) or []

                    if not supplier_quotes:
                        env.log_debug("今日该 SKU 无供应商报价，跳过。")
                        continue

                    # 2. 基于 quality_score 选择供应商（核心修改点）
                    chosen_supplier, chosen_supplier_price, chosen_quality_score = select_supplier_by_quality_score(
                        supplier_quotes, sku_id_for_test
                    )

                    if not chosen_supplier:
                        env.log_debug("未能选出供应商，跳过后续补货逻辑。")
                        continue

                    # 4. 基于历史销售选择最优售价（总利润最大）
                    revenue_by_price: Dict[float, List[float]] = {}
                    moves_by_price: Dict[float, List[int]] = {}
                    cost_price = chosen_supplier_price if chosen_supplier_price is not None else 0.0

                    for day_records in sales_records.values():
                        for rec in day_records:
                            price = float(rec.get("price", 0))
                            move = int(rec.get("move", 0))
                            profit = (price - cost_price) * move
                            revenue_by_price.setdefault(price, []).append(profit)
                            moves_by_price.setdefault(price, []).append(move)

                    best_price: Optional[float] = None
                    expected_move = 0.0
                    if revenue_by_price:
                        def avg_profit(item):
                            price, profits = item
                            if not profits:
                                return float("-inf")
                            return sum(profits) / len(profits)

                        best_price = max(revenue_by_price.items(), key=avg_profit)[0]
                        moves = moves_by_price.get(best_price, [])
                        expected_move = (sum(moves) / len(moves)) if moves else 0.0
                        env.log_debug(f"历史最优售价 {best_price}，基于历史预期日销量 {expected_move}")

                    # 5. 结合新闻影响调整对未来需求的预估
                    need_effect = 0.0
                    supply_effect = 0.0
                    if env.config.get("enable_new") and getattr(env, "news_manager", None):
                        try:
                            sku_obj = env.skus_id_map.get(sku_id_for_test)
                            sku_category = getattr(sku_obj, "category", None) if sku_obj else None

                            need_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["need"],
                            )
                            need_effect = float(need_info.get("total_effect", 0.0) or 0.0)

                            supply_info = env.news_manager.evaluate_impact_for_sku(
                                sku_id=sku_id_for_test,
                                sku_category=sku_category,
                                impact_factors=["supply"],
                            )
                            supply_effect = float(supply_info.get("total_effect", 0.0) or 0.0)
                        except Exception as e:
                            env.log_debug(f"[NewsImpact] 计算 SKU {sku_id_for_test} 新闻影响失败: {e}")

                    base_expected = expected_move if expected_move > 0 else 10.0
                    need_multiplier = 1.0 + need_effect
                    supply_multiplier = 1.0 + supply_effect
                    total_multiplier = max(0.5, min(2.0, need_multiplier * supply_multiplier))
                    effective_expected = base_expected * total_multiplier
                    
                    if day not in multiplier_tracking:
                        multiplier_tracking[day] = {}
                    multiplier_tracking[day][sku_id_for_test] = {
                        "need_mult": need_multiplier,
                        "supply_mult": supply_multiplier,
                        "need_effect": need_effect,
                        "supply_effect": supply_effect,
                    }
                    
                    env.log_debug(
                        f"[NewsImpact] SKU {sku_id_for_test} base_expected={base_expected:.2f}, "
                        f"effective_expected={effective_expected:.2f}"
                    )

                    # 6. 修改零售价
                    if best_price is not None and best_price > 0:
                        try:
                            res = env.exec_tools(
                                "modify_sku_price",
                                sku_id=sku_id_for_test,
                                new_price=best_price,
                            )
                            env.log_debug(f"修改价格结果: {res['formatted']}")
                        except Exception as e:
                            env.log_debug(f"修改价格失败: {e}")

                    # 7. 库存不足时触发大单补货
                    target_move = effective_expected
                    reorder_threshold = target_move * bulk_qty_multiplier
                    if current_quantity <= reorder_threshold:
                        open_orders_info = env.exec_tools("view_current_orders")
                        open_orders = open_orders_info.get("result", {}).get("orders", []) or []
                        
                        pending_qty = 0
                        for order in open_orders:
                            items = order.get("items", []) or []
                            for item in items:
                                if item.get("sku_id") == sku_id_for_test:
                                    pending_qty += item.get("quantity", 0)
                        
                        order_qty = max(1, int(target_move * bulk_qty_multiplier))
                        
                        current_total_inventory = sum(len(items) for items in env.inventory.items_by_sku.values()) + len(env.inventory.waiting_items)
                        available_capacity = env.inventory.capacity - current_total_inventory if env.inventory.capacity is not None else float('inf')
                        max_order_qty_considering_capacity = max(0, available_capacity - 100)
                        
                        if order_qty - pending_qty > max_order_qty_considering_capacity:
                            env.log_debug(
                                f"[库存容量限制] SKU {sku_id_for_test} 计划订货 {order_qty - pending_qty}，但可用容量仅 {max_order_qty_considering_capacity}，跳过或减少订货。"
                            )
                            order_qty = max(0, max_order_qty_considering_capacity + pending_qty)
                            if order_qty <= pending_qty:
                                env.log_debug(f"[库存容量限制] SKU {sku_id_for_test} 调整后无需新订货。")
                                continue
                        
                        if pending_qty >= order_qty:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，数量 {pending_qty} >= 需要补货数量 {order_qty}，跳过大单补货。"
                            )
                            continue
                        elif pending_qty > 0:
                            env.log_debug(
                                f"已有未完成订单包含 {sku_id_for_test}，但数量 {pending_qty} < 需要补货数量 {order_qty}，继续补货。"
                            )
                        
                        place_kwargs = {
                            "items": [{"sku_id": sku_id_for_test, "quantity": order_qty - pending_qty}],
                            "supplier_id": chosen_supplier,
                        }

                        try:
                            order_res = env.exec_tools("place_order", **place_kwargs)
                            env.log_debug(
                                f"[大单补货-质量优先] 触发补货，供应商 {chosen_supplier} (quality_score={chosen_quality_score:.4f})，"
                                f"下单结果：{order_res['formatted']} "
                                f"(基于新闻修正后的预期销量: {effective_expected:.2f})"
                            )
                        except Exception as e:
                            env.log_debug(f"[大单补货] 下单失败：{e}")

            # 3. 推进一天
            step_res = env.exec_tools("end_today")
            step_payload = step_res.get("result", {})
            env.log_debug("step 结果：")
            env.log_debug(step_res.get("formatted", ""))

            end_time = time.time()
            env.log_info(
                f"第 {day} 天模拟结束，耗时 {end_time - start_time:.4f} 秒"
            )

        env.log_debug("\n====== 模拟结束 ======")
        env.log_debug("最终资金 & 日期：" + env.exec_tools("view_funds_and_date")['formatted'])
        ratings = env.get_sku_rating_report()
        
        final_inv = env.exec_tools("view_inventory")
        final_total_skus = final_inv.get("result", {}).get("total_skus")
        env.log_debug(f"最终总 SKU 数：{final_total_skus}")
    finally:
        # 结束时清理容器
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="零售环境模拟器 - 选择要执行的模拟函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行简单逻辑环境（365天）
  python retail_environment.py --mode logic --days 365 --sample-size 2 --config-type dynamic_hard
  
  # 运行评价感知环境（180天）
  python retail_environment.py --mode review --days 180 --sample-size 2 --config-type dynamic_middle --db-path middle/simulate_data/15/records_review/
  
  # 运行新闻感知环境（180天）
  python retail_environment.py --mode news --days 180 --sample-size 2 --config-type still_hard --db-path middle/simulate_data/15/records_review_news/
  
  # 运行质量优先环境（180天）
  python retail_environment.py --mode quality --days 180 --sample-size 2 --config-type dynamic_hard --db-path data/simulate_data/15/records_no_review/
  
  # 运行工具自检
  python retail_environment.py --mode tools
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["logic", "review", "news", "quality", "tools"],
        required=True,
        help="选择模拟模式: logic=简单逻辑环境, review=评价感知环境, news=新闻感知环境, quality=质量优先环境, tools=工具自检"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="模拟天数（默认: 180）"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2,
        help="每个品类选择的 SKU 数量（默认: 2）"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="数据库路径（可选，不同模式有默认值）"
    )
    
    parser.add_argument(
        "--config-type",
        type=str,
        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"],
        default="dynamic_hard",
        help="配置类型: 'dynamic_hard', 'dynamic_middle', 'still_hard', or 'still_middle' (默认: 'dynamic_hard')"
    )
    
    args = parser.parse_args()
    
    # 根据模式执行相应的函数
    if args.mode == "tools":
        print("运行工具自检...")
        run_all_tools_once()
    
    elif args.mode == "logic":
        db_path = args.db_path if args.db_path else 'model_run_time'
        print(f"运行简单逻辑环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
        simulate_simple_logic_environment(
            days=args.days,
            sample_size=args.sample_size,
            config_type=args.config_type,
            db_path=db_path
        )
    
    elif args.mode == "review":
        db_path = args.db_path or 'middle/simulate_data/15/records_review/'
        print(f"运行评价感知环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
        simulate_simple_review_environment(
            days=args.days,
            sample_size=args.sample_size,
            db_path=db_path,
            config_type=args.config_type
        )
    
    elif args.mode == "news":
        db_path = args.db_path or 'middle/simulate_data/15/records_review_news/'
        print(f"运行新闻感知环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
        simulate_news_aware_environment(
            days=args.days,
            sample_size=args.sample_size,
            db_path=db_path,
            config_type=args.config_type
        )
    
    elif args.mode == "quality":
        db_path = args.db_path or 'data/simulate_data/15/records_no_review/'
        print(f"运行质量优先环境: days={args.days}, sample_size={args.sample_size}, config_type={args.config_type}, db_path={db_path}")
        simulate_quality_based_environment(
            days=args.days,
            sample_size=args.sample_size,
            db_path=db_path,
            config_type=args.config_type
        )

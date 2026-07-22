#!/usr/bin/env python3
"""
ReAct-style loop for RetailEnvironment using OpenAI tools.

The agent can call environment tools (including the SQL DSL tool) and will receive
the formatted output each turn.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from openai import OpenAI

from retail_environment import RetailEnvironment
from util.default_config import (
    create_dynamic_hard_config,
    create_dynamic_middle_config,
    create_still_hard_config,
    create_still_middle_config,
)
from module.strategy_manager import (
    StrategyManager,
    format_strategy_dict,
    get_strategy_tool_definitions,
)

from module.stream_chat import stream_chat
from util.tool_call_parser import parse_tool_args, parse_tool_calls

DEFAULT_MODEL = 'qwen3-235b-a22b-thinking-2507'

# 策略阶段的系统提示模板（基础版本，不包含新闻）
STRATEGY_SYSTEM_PROMPT_BASE = """You are a retail strategy analyst. Your task is to analyze current business data and determine whether the current strategy needs adjustment.


Environment Characteristics

- The store operates with a large number of SKUs, where products within the same category interact and may substitute or cannibalize each other's demand.
- Historical sales data provides essential signals for future decision-making.
- Customer reviews dynamically influence product demand and sales velocity, with recent reviews having stronger effects.
{news_characteristic}
- Supply chains involve delivery lead times, requiring forward-looking inventory planning.
- Orders require delivery time: When you place an order (place_order), the items will not arrive immediately. The delivery time varies and can take up to 7 days (within 7 days). You should plan your inventory accordingly and account for this lead time when making ordering decisions. Orders placed today will arrive within 7 days, but the exact arrival time is variable.
- Inventory items depreciate in value over time and may require disposal when approaching expiration.
- Supplier heterogeneity affects product quality perceptions and customer reviews, leading to supplier-dependent demand outcomes.
- Daily rent: The store incurs a fixed daily rent cost of {daily_rent} that must be paid each day. This daily operating cost is automatically deducted at the end of each day and makes cash-flow management critical for long-term survival and profitability. You must ensure sufficient funds are available to cover the daily rent. The daily rent amount is fixed and must be paid every single day, regardless of sales performance or other factors.

# Your Role in Strategy Phase

Each day starts with a STRATEGY PHASE where you:
1. Review the current strategy (provided at the start of this phase)
2. Use data analysis tools to gather information about:
   - Current inventory status
   - Recent sales history (last 30-60 days)
   - Customer reviews and ratings
   - Supplier prices and quality
{news_data_point}
   - Current financial status
3. Compare current situation with previous days to identify significant changes
4. Set the strategy using the three separate tools (set_macro_strategy, set_execute_strategy, set_action) to set the three strategy components

# Strategy Format

The strategy consists of three components:

1. **macro_strategy**: A list of broad strategic guidelines (array of strings)
   - Examples: ["Focus on high-margin products", "Maintain competitive pricing", "Prioritize inventory turnover"]

2. **execute_strategy**: An object with seven fields, all values are arrays:
   - **focus_skus**: Array of SKU IDs that need attention (e.g., ["SKU_001", "SKU_002"])
   - **sku_supplier_mapping**: Array of mapping objects (e.g., [{{"sku_id": "SKU_001", "supplier_id": "supplier_A"}}, {{"sku_id": "SKU_002", "supplier_id": "supplier_B"}}])
{news_to_monitor_field}
   - **skus_to_reorder**: Array of SKU IDs that need reordering (e.g., ["SKU_003", "SKU_004"])
   - **price_adjustments**: Array of price adjustment objects (e.g., [{{"sku_id": "SKU_001", "adjustment": "increase by 10%"}}, {{"sku_id": "SKU_002", "adjustment": "decrease by 5%"}}])
   - **sku_to_monitor**: Array of SKU IDs that should be closely monitored (e.g., ["SKU_005", "SKU_006"])
   - **other**: Array of other strategy notes or metadata (e.g., [{{"comment": "..."}}, {{"risk_level": "high"}}])

3. **today_action**: An array of action objects, each representing a concrete action using the parameter format of `place_order` or `modify_sku_price`.
   - Each action MUST be an object of the form:
     - {{"tool": "place_order", "arguments": {{<place_order arguments>}}}}
     - OR {{"tool": "modify_sku_price", "arguments": {{<modify_sku_price arguments>}}}}
   - Example:
     [
       {{"tool": "place_order", "arguments": {{"sku_id": "SKU_001", "supplier_id": "supplier_A", "quantity": 100}}}},
       {{"tool": "modify_sku_price", "arguments": {{"sku_id": "SKU_002", "new_price": 9.99}}}}
     ]

# Strategy Setting Tools

Use three separate tools to set different parts of the strategy:
- **set_macro_strategy**: Set the macro_strategy (array of strings)
  - Parameter: `macro_strategy` (array of strings)
  - Example: set_macro_strategy(macro_strategy=["Focus on high-margin products", "Maintain competitive pricing"])

- **set_execute_strategy**: Set the execute_strategy (object with seven fields, all arrays)
  - Parameter: `execute_strategy` (object with fields: focus_skus, sku_supplier_mapping{news_to_monitor_param}, skus_to_reorder, price_adjustments, sku_to_monitor, other)
  - All field values must be arrays
  - Example: set_execute_strategy(execute_strategy={{"focus_skus": ["SKU_001"], "sku_supplier_mapping": [{{"sku_id": "SKU_001", "supplier_id": "supplier_A"}}], ...}})

- **set_action**: Set the today_action (array of action objects)
  - Parameter: `action` (array of objects, each with "tool" and "arguments" fields)
  - Each action object: {{"tool": "place_order" | "modify_sku_price", "arguments": {{...}}}}
  - Example: set_action(action=[{{"tool": "place_order", "arguments": {{"sku_id": "SKU_001", "supplier_id": "supplier_A", "quantity": 100}}}}])

You can call these tools multiple times to build or modify the strategy. After your analysis, set all three components to reflect your decisions.

# Available Tools for Analysis

The available function signatures are provided within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a JSON object with function name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Important Analysis Tools

Use these tools to gather data:
- view_funds_and_date: Check current funds and date
- view_inventory: Check current inventory levels
- view_sku_sales_history: Analyze sales trends (use last 30-60 days)
- view_sku_avg_ratings: Check customer satisfaction
- view_current_date_supplier_prices: Check supplier availability and prices
{news_tools_list}
- view_current_orders: Check pending orders
- set_macro_strategy: Set the macro strategy (array of strings)
- set_execute_strategy: Set the execute strategy (object with seven fields, all arrays)
- set_action: Set the today action (array of action objects)

Note: You CANNOT use place_order or modify_sku_price in the strategy phase. These tools are only available in the execution phase.

After completing your analysis and updating the strategy, the system will transition to the EXECUTION PHASE.
"""

# 执行阶段的系统提示模板（基础版本，不包含新闻）
EXECUTION_SYSTEM_PROMPT_BASE = """You are a retail operations agent executing daily operations based on the current strategy.

# Your Role in Execution Phase

In the EXECUTION PHASE, you will receive the **final strategy** determined in the Strategy Phase. This strategy includes:
- **macro_strategy**: Broad strategic guidelines (array of strings)
- **execute_strategy**: Specific operational details (object with seven fields, all arrays)
- **today_action**: Concrete actions to take today (array of action objects)

# Important Operational Constraints

- **Daily rent**: The store incurs a fixed daily rent cost of {daily_rent} that must be paid each day. This daily operating cost is automatically deducted at the end of each day. Ensure you have sufficient funds to cover this daily expense. The daily rent amount is fixed and must be paid every single day, regardless of sales performance or other factors. This makes cash-flow management critical for long-term survival and profitability.
- **Order delivery time**: When you place an order using place_order, the items will not arrive immediately. The delivery time varies and can take up to 7 days (within 7 days). Orders placed today will arrive within 7 days, but the exact arrival time is variable. Plan your inventory and ordering decisions accordingly, considering the lead time for items to arrive.

# Strategy Usage Guidelines

**The strategy is provided as REFERENCE, but you can and should make additional actions based on real-time data:**

1. **Reference the strategy** to understand priorities and planned actions:
   - Use macro_strategy for overall decision-making direction
   - Use execute_strategy fields (focus_skus, sku_supplier_mapping{news_to_monitor_ref}, skus_to_reorder, price_adjustments, sku_to_monitor, other) as guidance
   - Consider today_action as suggested actions to take

2. **Perform additional data queries** to validate and refine decisions:
   - Check current inventory levels, sales history, supplier prices{news_impacts_ref}, funds, etc.
   - Use tools like view_inventory, view_sku_sales_history, view_current_date_supplier_prices{news_tools_ref}, etc.

3. **Execute actions flexibly**:
   - You can execute actions from today_action when they still make sense given the latest data
   - You can **adjust, skip, or modify** actions from today_action if your analysis shows better alternatives
   - You can **add additional actions** beyond today_action if needed (e.g., unexpected inventory changes, new supplier prices{news_impacts_example})
   - You can use information from execute_strategy (like focus_skus, sku_supplier_mapping) to make decisions even if not explicitly in today_action

4. **End the day** by calling end_today when you've completed all operations for today.

# Important Constraints

- You MUST NOT modify the stored strategy itself in this phase (strategy can only be changed in the Strategy Phase)
- You CANNOT call any tool that changes macro_strategy / execute_strategy / today_action
- You SHOULD use the strategy as guidance but make final decisions based on current data and analysis

# Available Tools

The available function signatures are provided within <tools></tools> XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a JSON object with function name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Ending the Day

When you have completed all reasonable operations for the day (especially those in today_action, adjusted as needed by current data), you MUST call end_today to advance to the next day. This will trigger a new Strategy Phase for the next day.
"""

DEFAULT_CONFIG_PATH: Path | None = None  # Use default config loader
DEFAULT_MAX_TURNS = 10000


def build_openai_tools(env: RetailEnvironment, exclude_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convert RetailEnvironment.get_tools() to OpenAI tool schema."""
    if exclude_tools is None:
        exclude_tools = []
    tools = []
    for name, meta in env.get_tools().items():
        if name in exclude_tools:
            continue
        params = meta.get("input_schema") or meta.get(
            "parameters",
            {"type": "object", "properties": {}, "required": []},
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def build_strategy_prompt(tool_definitions: str, enable_news: bool, daily_rent: float = 0) -> str:
    """根据是否启用新闻构建策略阶段的系统提示。"""
    if enable_news:
        news_characteristic = "- External news dynamically influences product demand and sales velocity."
        news_data_point = "   - News impacts (if available)"
        news_to_monitor_field = "   - **news_to_monitor**: Array of news items to monitor (e.g., [\"News about category X\", \"Supplier Y quality issues\"])"
        news_to_monitor_param = ", news_to_monitor"
        news_tools_list = "- view_news_history: Check recent news impacts (if enabled)\n- view_today_news: Check today's news list\n- view_news_detail: View specific news details"
    else:
        news_characteristic = ""
        news_data_point = ""
        news_to_monitor_field = ""
        news_to_monitor_param = ""
        news_tools_list = ""
    
    return STRATEGY_SYSTEM_PROMPT_BASE.format(
        tool_definitions=tool_definitions,
        news_characteristic=news_characteristic,
        news_data_point=news_data_point,
        news_to_monitor_field=news_to_monitor_field,
        news_to_monitor_param=news_to_monitor_param,
        news_tools_list=news_tools_list,
        daily_rent=daily_rent,
    )


def build_execution_prompt(tool_definitions: str, enable_news: bool, daily_rent: float = 0) -> str:
    """根据是否启用新闻构建执行阶段的系统提示。"""
    if enable_news:
        news_to_monitor_ref = ", news_to_monitor"
        news_impacts_ref = ", news impacts"
        news_tools_ref = ", view_news_history"
        news_impacts_example = ", or news impacts"
    else:
        news_to_monitor_ref = ""
        news_impacts_ref = ""
        news_tools_ref = ""
        news_impacts_example = ""
    
    return EXECUTION_SYSTEM_PROMPT_BASE.format(
        tool_definitions=tool_definitions,
        news_to_monitor_ref=news_to_monitor_ref,
        news_impacts_ref=news_impacts_ref,
        news_tools_ref=news_tools_ref,
        news_impacts_example=news_impacts_example,
        daily_rent=daily_rent,
    )


def render_sku_descriptions(env: RetailEnvironment) -> str:
    lines = []
    lines.append("SKU catalog (grouped by category). SKU_ID is the unique product identifier. Promotion_Days indicates when the item will be discounted/cleared and should be sold before that window expires.")
    for category, sku_list in env.skus_category_map.items():
        lines.append(f"## Category: {category}")
        for sku in sku_list:
            desc = sku.attributes or {}
            detail = desc.get("description") or desc.get("DESCRIP") or ""
            brand = sku.brand
            promotion_days = getattr(sku, "promotion_day", None) or desc.get("PROMOTION_TIME")
            lines.append(f"- SKU_id={sku.sku_id}, Expiration_Days={promotion_days}, Brand={brand}, Desc={detail}, Category={category}")
        lines.append("")  # spacer

    return "\n".join(lines).strip()


def render_tool_definitions(tools: List[Dict[str, Any]]) -> str:
    """Render tool definitions as newline-delimited JSON objects for the prompt."""
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def safe_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)








def log_message(records: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """Append a timestamped record to the in-memory run log."""
    record = {"ts": datetime.utcnow().isoformat() + "Z", **payload}
    records.append(record)


def write_log_json_array(log_path: Path, records: List[Dict[str, Any]]) -> None:
    """Persist the run log as a JSON array."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)



def save_turn_calls_to_json(
    log_dir: Path,
    phase: str,
    turn_index: int,
    turn_data: Dict[str, Any],
    day: int,
) -> None:
    """Save turn calls to a separate JSON file.
    
    Args:
        log_dir: Log directory path
        phase: Phase name ("strategy" or "execute")
        turn_index: Turn index (global turn number)
        turn_data: Data to save for this turn
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{phase}_{day}_{turn_index}.json"
    filepath = log_dir / str(day) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(turn_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Turn {turn_index}] Saved turn data to {filepath}")

# Default OpenAI client configuration (can be overridden by command line arguments or environment variables)
DEFAULT_API_KEY: str = os.environ.get("OPENAI_API_KEY") or os.environ.get("ZORO_DEFAULT_API_KEY") or ""
DEFAULT_BASE_URL: str = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ZORO_DEFAULT_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

_BRIDGE_ROUTES: list[tuple[tuple[str, ...], str, str, str]] = [
    (
        ("opus", "sonnet", "haiku", "claude"),
        "http://127.0.0.1:8399/v1",
        "ZORO_CC_BRIDGE_SECRET",
        "claude_bridge",
    ),
    (
        (
            "sol", "terra", "luna",
            "gpt-5", "gpt5",
            "codex", "codex-cc", "gpt5-codex-cc",
        ),
        "http://127.0.0.1:8398/v1",
        "ZORO_CX_BRIDGE_SECRET",
        "codex_bridge",
    ),
]


_MODEL_SHORT_MAP: dict[str, str] = {
    # Claude family
    "opus":                            "claude-opus-4.8",
    "claude-opus-4-8":                 "claude-opus-4.8",
    "claude-opus-4.8":                 "claude-opus-4.8",
    "sonnet":                          "claude-sonnet-4.5",
    "claude-sonnet-4-5-20250929":      "claude-sonnet-4.5",
    "claude-sonnet-4.5":               "claude-sonnet-4.5",
    "haiku":                           "claude-haiku-4.5",
    "claude-haiku-4-5-20251001":       "claude-haiku-4.5",
    "claude-haiku-4.5":                "claude-haiku-4.5",
    # GPT / Codex family
    "sol":                             "gpt-5.6",
    "terra":                           "gpt-5.6",
    "luna":                            "gpt-5.6",
    "gpt-5":                           "gpt-5.6",
    "gpt5":                            "gpt-5.6",
    "gpt-5.6":                         "gpt-5.6",
    "gpt-5.6-sol":                     "gpt-5.6",
    "codex":                           "gpt-5.6",
    "codex-cc":                        "gpt-5.6",
    "codex-mini":                      "gpt-5.6",
    "gpt5-codex-cc":                   "gpt-5.6",
}


def _resolve_model_short(model: str) -> str:
    """Return canonical short-name for output-directory nesting."""
    key = (model or "").strip().lower()
    if key in _MODEL_SHORT_MAP:
        return _MODEL_SHORT_MAP[key]
    safe = re.sub(r"[^A-Za-z0-9._\-]+", "_", key)
    return safe or "unknown-model"


def _pick_next_run_dir(model_dir: Path) -> Path:
    """Atomically claim and create the next <model_dir>/run_<N> directory.

    Uses `mkdir(exist_ok=False)` in a retry loop so concurrent workers cannot
    collide on the same run_N. On FileExistsError, advances to the next N.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    used: list[int] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("run_"):
            continue
        suffix = name[len("run_"):]
        if suffix.isdigit():
            used.append(int(suffix))
    next_n = (max(used) + 1) if used else 1
    while True:
        candidate = model_dir / f"run_{next_n}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            next_n += 1


def _latest_run_dir(model_dir: Path) -> Optional[Path]:
    """Return highest-numbered existing <model_dir>/run_<N> directory, or None if none exist.

    Used for --recover_day / --recover_turn in dataset mode so that continuation
    writes back into the SAME run_N directory whose checkpoints we're resuming from,
    rather than allocating a fresh run_(N+1) via _pick_next_run_dir.
    """
    if not model_dir.is_dir():
        return None
    best: Optional[int] = None
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("run_"):
            continue
        suffix = name[len("run_"):]
        if not suffix.isdigit():
            continue
        n = int(suffix)
        if best is None or n > best:
            best = n
    if best is None:
        return None
    return model_dir / f"run_{best}"


def _route_bridge(model: str) -> Optional[tuple[str, str, str]]:
    m = (model or "").strip().lower()
    if not m:
        return None
    for prefixes, base_url, env_var, name in _BRIDGE_ROUTES:
        if any(m == p or m.startswith(p) for p in prefixes):
            return base_url, env_var, name
    return None


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> OpenAI:
    if api_key is not None and base_url is not None:
        return OpenAI(api_key=api_key, base_url=base_url)

    route = _route_bridge(model) if model else None
    if route is not None:
        routed_base_url, env_var, name = route
        if base_url is None:
            base_url = routed_base_url
        if api_key is None:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"model {model!r} routes to {name} at {routed_base_url}, "
                    f"but {env_var} is not set. Export it in this shell "
                    f"(same value as the bridge was started with), or pass --api_key."
                )
        return OpenAI(api_key=api_key, base_url=base_url)

    if api_key is None:
        api_key = DEFAULT_API_KEY
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)

def print_env_status(env: RetailEnvironment, prefix: str = "") -> None:
    """
    打印环境状态信息。
    
    Args:
        env: RetailEnvironment 实例
        prefix: 打印前缀（用于区分不同的 checkpoint）
    """
    # 获取库存信息
    inventory_result = env.exec_tools("view_inventory")
    inventory_data = inventory_result.get("result", {})
    total_items = inventory_data.get("total_items", 0)
    total_waiting = inventory_data.get("waiting_items", 0)
    inventory_by_sku = inventory_data.get("inventory", {})
    
    # 获取订单信息
    orders_result = env.exec_tools("view_current_orders")
    orders_data = orders_result.get("result", {})
    pending_orders = orders_data.get("pending_orders", [])
    
    # 计算库存容量使用情况
    capacity_info = ""
    if env.inventory.capacity is not None:
        capacity_used = total_items + total_waiting
        capacity_available = env.inventory.capacity - capacity_used
        capacity_percent = (capacity_used / env.inventory.capacity * 100) if env.inventory.capacity > 0 else 0
        capacity_info = f" ({capacity_used}/{env.inventory.capacity} used, {capacity_available} available, {capacity_percent:.1f}%)"
    
    print(f"{prefix}Environment Status:")
    print(f"{prefix}  - Current Date: {env.current_date}")
    print(f"{prefix}  - Funds: {env.funds:,.2f}")
    print(f"{prefix}  - Inventory: {total_items} items in stock, {total_waiting} waiting{capacity_info}")
    print(f"{prefix}  - Pending Orders: {len(pending_orders)}")
    
    # 打印库存最多的前5个SKU
    if inventory_by_sku:
        sku_counts = [(sku_id, len(items)) for sku_id, items in inventory_by_sku.items() if items]
        sku_counts.sort(key=lambda x: x[1], reverse=True)
        top_skus = sku_counts[:5]
        if top_skus:
            print(f"{prefix}  - Top SKUs by inventory: {', '.join([f'{sku_id}({count})' for sku_id, count in top_skus])}")


def save_checkpoint(
    checkpoint_dir: Path,
    turn: int,
    messages: List[Dict[str, Any]],
    env: RetailEnvironment,
) -> None:
    """
    保存 checkpoint，包括 messages 和 environment 状态。
    
    Args:
        checkpoint_dir: checkpoint 目录
        turn: 当前 turn 数
        messages: 当前 messages
        env: RetailEnvironment 实例
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 messages
    messages_path = checkpoint_dir / f"messages_turn_{turn}.json"
    with messages_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存 environment checkpoint
    env_checkpoint_path = checkpoint_dir / f"env_checkpoint_turn_{turn}.json"
    env.save_checkpoint(env_checkpoint_path)
    
    print(f"[Checkpoint] Saved checkpoint at turn {turn} to {checkpoint_dir}")
    print_env_status(env, prefix="  ")


def recover_from_checkpoint(
    checkpoint_dir: Path,
    turn: int,
    config: Dict[str, Any],
) -> tuple[RetailEnvironment, List[Dict[str, Any]]]:
    """
    从 checkpoint 恢复环境状态和 messages。
    
    Args:
        checkpoint_dir: checkpoint 目录
        turn: 要恢复的 turn 数
        config: 配置字典
        
    Returns:
        (恢复后的 RetailEnvironment, messages)
    """
    # 恢复 messages
    messages_path = checkpoint_dir / f"messages_turn_{turn}.json"
    if not messages_path.exists():
        raise FileNotFoundError(f"Messages checkpoint not found: {messages_path}")
    
    with messages_path.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    
    # 恢复 environment
    env_checkpoint_path = checkpoint_dir / f"env_checkpoint_turn_{turn}.json"
    if not env_checkpoint_path.exists():
        raise FileNotFoundError(f"Environment checkpoint not found: {env_checkpoint_path}")
    
    env = RetailEnvironment.recover_from_checkpoint(env_checkpoint_path)
    
    print(f"[Checkpoint] Recovered from checkpoint at turn {turn} from {checkpoint_dir}")
    print_env_status(env, prefix="  ")
    return env, messages


def recover_from_day_checkpoint(
    checkpoint_dir: Path,
    day: int,
    config: Dict[str, Any],
) -> tuple[RetailEnvironment, List[Dict[str, Any]], Dict[str, Any], int, int]:
    """
    从 day checkpoint 恢复环境状态、messages、策略和运行状态。
    
    Args:
        checkpoint_dir: checkpoint 目录
        day: 要恢复的 day 数
        config: 配置字典
        
    Returns:
        (恢复后的 RetailEnvironment, messages, strategy, start_day, start_turn)
    """
    # 读取 day checkpoint 元数据
    day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
    if not day_checkpoint_path.exists():
        raise FileNotFoundError(f"Day checkpoint not found: {day_checkpoint_path}")
    
    with day_checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint_metadata = json.load(f)
    
    # 恢复 messages
    messages_path = checkpoint_dir / checkpoint_metadata["messages_path"]
    if not messages_path.exists():
        raise FileNotFoundError(f"Messages checkpoint not found: {messages_path}")
    
    with messages_path.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    
    # 恢复 environment
    env_checkpoint_path = checkpoint_dir / checkpoint_metadata["env_checkpoint_path"]
    if not env_checkpoint_path.exists():
        raise FileNotFoundError(f"Environment checkpoint not found: {env_checkpoint_path}")
    
    env = RetailEnvironment.recover_from_checkpoint(env_checkpoint_path)
    
    # 恢复策略
    strategy = checkpoint_metadata.get("strategy", {})
    
    # 获取 start_day 和 start_turn
    # day checkpoint 是在每天执行结束后保存的，所以恢复时应该从 day+1 开始
    recovered_day = checkpoint_metadata.get("day", day)
    start_day = recovered_day + 1  # 从下一天开始
    start_turn = checkpoint_metadata.get("global_turn", 0)
    
    # 从 day checkpoint 恢复时，messages 是前一天的，应该清空，因为新的一天会重新开始
    # 但保留 messages 的路径信息，以防需要调试
    messages = []  # 清空 messages，新的一天会重新开始
    
    print(f"[Checkpoint] Recovered from day {day} checkpoint:")
    print(f"  - Recovered Day: {recovered_day}")
    print(f"  - Start Day: {start_day} (will continue from next day)")
    print(f"  - Global Turn: {start_turn}")
    print(f"  - Current Date: {checkpoint_metadata.get('current_date', 'unknown')}")
    print(f"  - Strategy Turns: {checkpoint_metadata.get('strategy_turns', 0)}")
    print(f"  - Execution Turns: {checkpoint_metadata.get('execution_turns', 0)}")
    print("  - Messages: cleared (new day will start fresh)")
    print_env_status(env, prefix="  ")
    
    return env, messages, strategy, start_day, start_turn


def _check_negative_funds_and_maybe_terminate(
    env,
    consecutive_negative_days: int,
    previous_day_end_today_result: Optional[Dict[str, Any]],
    end_today_result: Dict[str, Any],
    day: int,
    run_log: list,
    log_path,
) -> tuple[bool, int, Optional[Dict[str, Any]]]:
    """Check funds after end_today and update consecutive_negative_days.

    ALWAYS sets previous_day_end_today_result before any return path — D7 fix.
    Returns (should_terminate, updated_consecutive_negative_days, updated_previous_day_end_today_result).
    """
    previous_day_end_today_result = end_today_result.copy() if isinstance(end_today_result, dict) else end_today_result
    result_data = end_today_result.get("result", {}) if isinstance(end_today_result, dict) else {}
    funds = result_data.get("funds", env.funds) if isinstance(result_data, dict) else env.funds
    if funds < 0:
        consecutive_negative_days += 1
        if consecutive_negative_days >= 5:
            print(f"[运营失败] 连续 {consecutive_negative_days} 天资金为负数")
            log_message(run_log, {"role": "system", "day": day, "message": f"运营失败：连续 {consecutive_negative_days} 天资金为负数", "funds": funds})
            write_log_json_array(log_path, run_log)
            return True, consecutive_negative_days, previous_day_end_today_result
    else:
        consecutive_negative_days = 0
    return False, consecutive_negative_days, previous_day_end_today_result


def run_strategy_execute_loop(
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    log_path: Path = Path("logs/run_env_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    start_day: int = 1,
    initial_strategy: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_strategy_turns_per_day: int = 10,
    max_execution_turns_per_day: int = 20,
    client: Optional[OpenAI] = None,
) -> None:
    # 检查是否启用新闻功能
    enable_news = env.config.get("enable_new", False)
    
    # 构建工具列表（策略阶段和执行阶段使用不同的工具集）
    all_tools = build_openai_tools(env)
    strategy_tools = build_openai_tools(env, exclude_tools=["modify_sku_price", "place_order", "end_today"])
    
    # 如果没有启用新闻，从工具列表中移除新闻相关工具
    if not enable_news:
        news_tools = ["view_news_history", "view_today_news", "view_news_detail"]
        all_tools = [t for t in all_tools if t["function"]["name"] not in news_tools]
        strategy_tools = [t for t in strategy_tools if t["function"]["name"] not in news_tools]

    # 添加策略管理工具到策略阶段工具列表
    strategy_tools.extend(get_strategy_tool_definitions())
    
    sku_desc = render_sku_descriptions(env)
    run_log: List[Dict[str, Any]] = []

    # 从 config 获取 daily rent
    daily_rent = env.config.get('everyday_rent', 0)
    
    # 根据是否启用新闻构建策略阶段和执行阶段的系统提示
    strategy_system_prompt = build_strategy_prompt(
        tool_definitions=render_tool_definitions(strategy_tools),
        enable_news=enable_news,
        daily_rent=daily_rent,
    )
    execution_system_prompt = build_execution_prompt(
        tool_definitions=render_tool_definitions(all_tools),
        enable_news=enable_news,
        daily_rent=daily_rent,
    )

    consecutive_negative_days = 0
    
    # 创建策略管理器，如果提供了初始策略则使用它
    if initial_strategy is not None:
        strategy_manager = StrategyManager(initial_strategy=initial_strategy)
        print("[Checkpoint] Restored strategy from checkpoint")
    else:
        strategy_manager = StrategyManager()
    
    # 使用提供的 client 或创建默认 client
    if client is None:
        client = create_openai_client()

    input_tokens = 0
    global_turn = start_turn  # 全局 turn 计数器
    
    # Token 统计变量
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # 保存前一天的 end_today 结果
    previous_day_end_today_result: Optional[Dict[str, Any]] = None

    # 按天循环，从 start_day 开始
    for day in range(start_day, max_days + 1):
        print(f"\n{'='*80}\nDay {day} - {env.current_date}\n{'='*80}\n")
        log_message(run_log, {"role": "system", "day": day, "current_date": str(env.current_date), "message": f"Day {day} started"})

        # 每天开始时初始化当天的 token 统计
        day_prompt_tokens = 0
        day_completion_tokens = 0
        day_tokens = 0

        # ========== 策略阶段 ==========
        print(f"[Day {day}] === STRATEGY PHASE ===")
        strategy_phase_complete = False
        strategy_turns = 0
        consecutive_no_valid_tool_calls = 0  # 连续没有合规工具调用的次数
        strategy_phase_failed = False
        strategy_default_leak = False
        
        # 切换到策略阶段的系统提示
        strategy_system_msg = {
            "role": "system",
            "content": strategy_system_prompt,
        }
        
        # 构建策略阶段的用户提示
        funds_result = env.exec_tools("view_funds_and_date")
        funds_formatted = funds_result.get("formatted", "")

        current_sku_desc = render_sku_descriptions(env)
        
        current_strategy_text = format_strategy_dict(strategy_manager.strategy_store, "Current Strategy:\n")
        
        # 如果有前一天的 end_today 结果，添加到提示中
        previous_day_info = ""
        if previous_day_end_today_result is not None:
            prev_result_formatted = previous_day_end_today_result.get("formatted", "")
            previous_day_info = f"## Previous Day (Day {day - 1}) End Result\n\n{prev_result_formatted}\n\n---\n\n"
        
        strategy_user_msg = {
            "role": "user",
            "content": f"# Day {day} - Strategy Phase\n\n{previous_day_info}## Current Status\n\n{funds_formatted}\n\n## Current Strategy\n\n{current_strategy_text}\n\n## SKU Catalog\n\n{current_sku_desc}\n\n## Instructions\n\nYou may use data tools to analyze the current business situation of the store; if you believe the strategy needs to be updated, update it accordingly, and continue doing so until you determine the strategy is appropriate, at which point do not call any tool and output END.",
        }
        
        log_message(run_log, {**strategy_system_msg, "day": day, "phase": "strategy"})
        log_message(run_log, {**strategy_user_msg, "day": day, "phase": "strategy"})

        strategy_messages = []

        # 策略阶段循环
        while not strategy_phase_complete and strategy_turns < max_strategy_turns_per_day:
            strategy_turns += 1
            global_turn += 1
            
            try:
                if input_tokens > max_input_tokens:
                    assistant_idxs = [i for i, m in enumerate(strategy_messages) if m.get("role") == "assistant"]
                    if len(assistant_idxs) >= 3:
                        strategy_messages = strategy_messages[assistant_idxs[2]:]
                        assert strategy_messages[0]['role'] == 'assistant'
                    elif len(assistant_idxs) >= 2:
                        strategy_messages = strategy_messages[assistant_idxs[1]:]
                        assert strategy_messages[0]['role'] == 'assistant'
                
                request_messages = [strategy_system_msg] + [strategy_user_msg] + strategy_messages
                
                try:
                    full_content, final_content, reasoning_content, usage = stream_chat(
                        client=client,
                        model=model,
                        messages=request_messages,
                    )

                except Exception as stream_exc:
                    import traceback
                    err_msg = f"LLM stream_chat failed on day {day} strategy turn {strategy_turns}: {type(stream_exc).__name__}: {stream_exc}"
                    print(f"[严重错误] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "phase": "strategy", "turn": strategy_turns, "message": err_msg, "error_type": type(stream_exc).__name__})
                    write_log_json_array(log_path, run_log)
                    strategy_messages.append({"role": "user", "content": f"System error: {err_msg}. Continue."})
                    continue

                # 解析工具调用
                parse_method_tag = "none"
                try:
                    parse_source = final_content or full_content or ''
                    tool_calls_list, parse_method_tag = parse_tool_calls(parse_source)
                except Exception as parse_exc:
                    err_msg = f"Failed to parse tool calls on day {day} strategy turn {strategy_turns}: {type(parse_exc).__name__}: {parse_exc}"
                    print(f"[错误] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "phase": "strategy", "turn": strategy_turns, "message": err_msg, "error_type": type(parse_exc).__name__})
                    tool_calls_list, parse_method_tag = [], "none"

                # 保存策略阶段本轮调用到单独的 JSON 文件
                if log_dir is not None:
                    turn_data = {
                        "day": day,
                        "phase": "strategy",
                        "turn": strategy_turns,
                        "global_turn": global_turn,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "think_content": full_content,
                        "full_content": full_content,
                        "final_content": final_content,
                        "reasoning_content": reasoning_content,
                        "tool_calls": tool_calls_list,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                            "completion_tokens": usage.get("completion_tokens") if usage else None,
                            "total_tokens": usage.get("total_tokens") if usage else None,
                        },
                        "messages": request_messages,
                    }
                    save_turn_calls_to_json(log_dir, "strategy", strategy_turns, turn_data, day)

                log_message(run_log, {"role": "usage", "day": day, "phase": "strategy", "turn": strategy_turns, "prompt_tokens": usage.get("prompt_tokens") if usage else None, "completion_tokens": usage.get("completion_tokens") if usage else None, "total_tokens": usage.get("total_tokens") if usage else None})
                input_tokens = usage.get("prompt_tokens") if usage else input_tokens
                # 累计 token 统计（总累计和当天累计）
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    tokens = usage.get("total_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += tokens
                    day_prompt_tokens += prompt_tokens
                    day_completion_tokens += completion_tokens
                    day_tokens += tokens
                print(f"[Day {day} Strategy Turn {strategy_turns}] tokens: prompt={usage.get('prompt_tokens') if usage else None}, completion={usage.get('completion_tokens') if usage else None}")
                log_message(run_log, {"role": "assistant", "day": day, "phase": "strategy", "turn": strategy_turns, "content": full_content, "full_content": full_content, "final_content": final_content, "reasoning": reasoning_content, "tool_calls": tool_calls_list})

                strategy_messages.append({
                    "role": "assistant",
                    "content": full_content,
                })

                if "END" in (final_content or "") and len(tool_calls_list) == 0:
                    strategy_phase_complete = True
                    print(f"[Day {day}] Strategy phase complete - END called")
                    break

                user_message = ''
                valid_tool_calls_count = 0  # 本次循环中合规工具调用的数量
                for call in tool_calls_list:
                    if not isinstance(call, dict) or not call.get("name"):
                        err_msg = f"Invalid tool call: {call}"
                        print(f"[错误] {err_msg}")
                        log_message(run_log, {"role": "error", "day": day, "phase": "strategy", "turn": strategy_turns, "message": err_msg, "call": str(call)})
                        user_message += f"<tool_response>Error: {err_msg}</tool_response>\n"
                        continue

                    name, args = call.get("name"), parse_tool_args(call.get("arguments"))
                    tool_executed_successfully = False
                    try:
                        if name == "set_macro_strategy":
                            result = strategy_manager.set_macro_strategy(**args)
                            tool_executed_successfully = True
                        elif name == "set_execute_strategy":
                            result = strategy_manager.set_execute_strategy(**args)
                            tool_executed_successfully = True
                        elif name == "set_action":
                            result = strategy_manager.set_action(**args)
                            tool_executed_successfully = True
                        else:
                            result = env.exec_tools(name, **args)
                            tool_executed_successfully = True
                        if not isinstance(result, dict) or "formatted" not in result:
                            result = {"formatted": str(result), "result": result}
                    except Exception as exc:
                        import traceback
                        err_msg = f"Error executing tool {name}: {type(exc).__name__}: {exc}"
                        print(f"[错误] {err_msg}")
                        if hasattr(env, 'logger'):
                            env.logger.error(f"{err_msg}\n{traceback.format_exc()}")
                        result = {"formatted": err_msg, "result": {"error": str(exc), "error_type": type(exc).__name__}}

                    log_message(run_log, {"role": "tool", "day": day, "phase": "strategy", "turn": strategy_turns, "name": name, "args": args, "content": result.get("formatted", safe_dump(result)), "raw": result.get("result", safe_dump(result))})
                
                    if name in ("set_macro_strategy", "set_execute_strategy", "set_action"):
                        print(f"[Day {day}] Strategy set: {name}")
                        log_message(run_log, {"role": "system", "day": day, "phase": "strategy", "message": f"Strategy set: {name}", "strategy": json.loads(json.dumps(strategy_manager.strategy_store, default=str))})
                    
                    if tool_executed_successfully:
                        valid_tool_calls_count += 1
                    
                    user_message += f"<tool_response>{result.get('formatted', safe_dump(result))}</tool_response>\n"

                # 检查是否有合规工具调用
                if valid_tool_calls_count > 0:
                    consecutive_no_valid_tool_calls = 0  # 重置计数器
                else:
                    consecutive_no_valid_tool_calls += 1
                    print(f"[Day {day} Strategy Turn {strategy_turns}] No valid tool calls (consecutive: {consecutive_no_valid_tool_calls}/5)")
                
                # 如果连续5次没有合规工具调用，则 break
                if consecutive_no_valid_tool_calls >= 5:
                    print(f"[Day {day}] Strategy phase ended: 5 consecutive turns with no valid tool calls")
                    strategy_phase_complete = True
                    logging.warning(
                        f"[Day {day}] Strategy phase timed out on 5 consecutive no-tool-call "
                        "turns — execution will run with stale/default strategy."
                    )
                    strategy_phase_failed = True
                    break

                if tool_calls_list and user_message:
                    strategy_messages.append({"role": "user", "content": user_message})
                    log_message(run_log, {"role": "user", "day": day, "phase": "strategy", "turn": strategy_turns, "content": user_message, "parse_method": parse_method_tag})
                else:
                    strategy_messages.append({"role": "user", "content": "No valid tool call detected. Continue analysis."})

                write_log_json_array(log_path, run_log)

            except Exception as exc:
                import traceback
                err_msg = f"Strategy phase failed on day {day} turn {strategy_turns}: {type(exc).__name__}: {str(exc)}"
                print(f"[严重错误] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "strategy", "turn": strategy_turns, "message": err_msg, "error_type": type(exc).__name__})
                write_log_json_array(log_path, run_log)
                strategy_phase_complete = True

        if not strategy_phase_complete or not strategy_manager.strategy_store.get("macro_strategy"):
            if not strategy_manager.strategy_store.get("macro_strategy") or len(strategy_manager.strategy_store.get("macro_strategy", [])) == 0:
                strategy_manager.strategy_store["macro_strategy"] = ["Default strategy: Maintain current operations and monitor performance"]
            print(f"[Day {day}] Strategy phase ended")

        print("\nFinal strategy after strategy phase:")
        print(format_strategy_dict(strategy_manager.strategy_store))
        # T9: detect if strategy still equals the initial default placeholder
        strategy_default_leak = False
        if strategy_manager.strategy_store.get("macro_strategy") == [
            "Focus on maintaining inventory levels and competitive pricing"
        ]:
            logging.warning(
                f"[Day {day}] Strategy phase produced default placeholder — indicates silent "
                "strategy failure. Execution will run on stale/default strategy."
            )
            strategy_default_leak = True

        # ========== 执行阶段 ==========
        print(f"\n[Day {day}] === EXECUTION PHASE ===")
        execution_phase_complete = False
        execution_turns = 0
        consecutive_no_valid_tool_calls_exec = 0  # 连续没有合规工具调用的次数
        
        # 切换到执行阶段的系统提示
        execution_system_msg = {
            "role": "system",
            "content": execution_system_prompt,
        }
        
        strategy_text = format_strategy_dict(strategy_manager.strategy_store, "Strategy for today:\n")
        
        sku_desc = render_sku_descriptions(env)

        execution_user_msg = {
            "role": "user",
            "content": f"# Day {day} - Execution Phase\n\n## Final Strategy for Today\n\n{strategy_text}\n\n## SKU Catalog\n\n{sku_desc}\n\n## Instructions\n\nThe strategy above is provided as **reference and guidance**. You should:\n1. **Review the strategy** to understand priorities and planned actions\n2. **Perform additional data queries** (inventory, sales, suppliers, news, funds) to validate and refine decisions\n3. **Execute actions flexibly**:\n   - You can execute actions from today_action when they make sense\n   - You can adjust, skip, or modify actions if your analysis shows better alternatives\n   - You can add additional actions beyond today_action if needed\n   - Use execute_strategy information (focus_skus, sku_supplier_mapping, etc.) to guide decisions\n4. **End the day** by calling end_today when all operations are complete.",
        }

        
        log_message(run_log, {**execution_system_msg, "day": day, "phase": "execution"})
        log_message(run_log, {**execution_user_msg, "day": day, "phase": "execution"})

        # 执行阶段独立的 messages 列表
        execution_messages = []

        # 执行阶段循环
        while not execution_phase_complete and execution_turns < max_execution_turns_per_day:
            execution_turns += 1
            global_turn += 1
            
            try:
                if input_tokens > max_input_tokens:
                    assistant_idxs = [i for i, m in enumerate(execution_messages) if m.get("role") == "assistant"]
                    if len(assistant_idxs) >= 3:
                        execution_messages = execution_messages[assistant_idxs[2]:]
                        assert execution_messages[0]['role'] == 'assistant'
                    elif len(assistant_idxs) >= 2:
                        execution_messages = execution_messages[assistant_idxs[1]:]
                        assert execution_messages[0]['role'] == 'assistant'
                
                request_messages = [execution_system_msg] + [execution_user_msg] + execution_messages
                
                try:
                    full_content, final_content, reasoning_content, usage = stream_chat(
                        client=client,
                        model=model,
                        messages=request_messages,
                    )
                except Exception as stream_exc:
                    import traceback
                    err_msg = f"LLM stream_chat failed on day {day} execution turn {execution_turns}: {type(stream_exc).__name__}: {stream_exc}"
                    print(f"[严重错误] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(stream_exc).__name__})
                    write_log_json_array(log_path, run_log)
                    execution_messages.append({"role": "user", "content": f"System error: {err_msg}. Continue."})
                    continue

                parse_method_tag = "none"
                try:
                    parse_source = final_content or full_content or ''
                    tool_calls_list, parse_method_tag = parse_tool_calls(parse_source)
                except Exception as parse_exc:
                    err_msg = f"Failed to parse tool calls on day {day} execution turn {execution_turns}: {type(parse_exc).__name__}: {parse_exc}"
                    print(f"[错误] {err_msg}")
                    log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(parse_exc).__name__})
                    tool_calls_list, parse_method_tag = [], "none"

                # 保存执行阶段本轮调用到单独的 JSON 文件
                if log_dir is not None:
                    turn_data = {
                        "day": day,
                        "phase": "execution",
                        "turn": execution_turns,
                        "global_turn": global_turn,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "think_content": full_content,
                        "full_content": full_content,
                        "final_content": final_content,
                        "reasoning_content": reasoning_content,
                        "tool_calls": tool_calls_list,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                            "completion_tokens": usage.get("completion_tokens") if usage else None,
                            "total_tokens": usage.get("total_tokens") if usage else None,
                        },
                        "messages": request_messages,
                    }
                    save_turn_calls_to_json(log_dir, "execute", execution_turns, turn_data, day)

                log_message(run_log, {"role": "usage", "day": day, "phase": "execution", "turn": execution_turns, "prompt_tokens": usage.get("prompt_tokens") if usage else None, "completion_tokens": usage.get("completion_tokens") if usage else None, "total_tokens": usage.get("total_tokens") if usage else None})
                input_tokens = usage.get("prompt_tokens") if usage else input_tokens
                # 累计 token 统计（总累计和当天累计）
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    tokens = usage.get("total_tokens", 0)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += tokens
                    day_prompt_tokens += prompt_tokens
                    day_completion_tokens += completion_tokens
                    day_tokens += tokens
                print(f"[Day {day} Execution Turn {execution_turns}] tokens: prompt={usage.get('prompt_tokens') if usage else None}, completion={usage.get('completion_tokens') if usage else None}")
                log_message(run_log, {"role": "assistant", "day": day, "phase": "execution", "turn": execution_turns, "content": full_content, "full_content": full_content, "final_content": final_content, "reasoning": reasoning_content, "tool_calls": tool_calls_list})

                execution_messages.append({
                    "role": "assistant",
                    "content": full_content,
                })

                user_message = ''
                valid_tool_calls_count = 0  # 本次循环中合规工具调用的数量
                for call in tool_calls_list:
                    if not isinstance(call, dict) or not call.get("name"):
                        err_msg = f"Invalid tool call: {call}"
                        print(f"[错误] {err_msg}")
                        log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "call": str(call)})
                        user_message += f"<tool_response>Error: {err_msg}</tool_response>\n"
                        continue

                    name, args = call.get("name"), parse_tool_args(call.get("arguments"))
                    tool_executed_successfully = False
                    try:
                        result = env.exec_tools(name, **args)
                        tool_executed_successfully = True
                        if not isinstance(result, dict) or "formatted" not in result:
                            result = {"formatted": str(result), "result": result}
                    except Exception as exc:
                        import traceback
                        err_msg = f"Error executing tool {name}: {type(exc).__name__}: {exc}"
                        print(f"[错误] {err_msg}")
                        if hasattr(env, 'logger'):
                            env.logger.error(f"{err_msg}\n{traceback.format_exc()}")
                        result = {"formatted": err_msg, "result": {"error": str(exc), "error_type": type(exc).__name__}}

                    log_message(run_log, {"role": "tool", "day": day, "phase": "execution", "turn": execution_turns, "name": name, "args": args, "content": result.get("formatted", safe_dump(result)), "raw": result.get("result", safe_dump(result))})
                    
                    if name == 'end_today':
                        execution_phase_complete = True
                        print(f"[Day {day}] Execution phase complete - end_today called")
                        # 保存 end_today 的结果，供下一天使用
                        _exit, consecutive_negative_days, previous_day_end_today_result = (
                            _check_negative_funds_and_maybe_terminate(
                                env, consecutive_negative_days, previous_day_end_today_result,
                                result, day, run_log, log_path,
                            )
                        )
                        if _exit:
                            return
                    
                    if tool_executed_successfully:
                        valid_tool_calls_count += 1
                    
                    user_message += f"<tool_response>{result.get('formatted', safe_dump(result))}</tool_response>\n"

                # 检查是否有合规工具调用
                if valid_tool_calls_count > 0:
                    consecutive_no_valid_tool_calls_exec = 0  # 重置计数器
                else:
                    consecutive_no_valid_tool_calls_exec += 1
                    print(f"[Day {day} Execution Turn {execution_turns}] No valid tool calls (consecutive: {consecutive_no_valid_tool_calls_exec}/5)")
                
                # 如果连续5次没有合规工具调用，则 break（但 end_today 已经调用的情况除外）
                if consecutive_no_valid_tool_calls_exec >= 5 and not execution_phase_complete:
                    print(f"[Day {day}] Execution phase ended: 5 consecutive turns with no valid tool calls")
                    print(f"[Day {day}] Executing end_today due to consecutive invalid tool calls")
                    try:
                        end_today_result = env.exec_tools("end_today")
                        execution_phase_complete = True
                        _exit, consecutive_negative_days, previous_day_end_today_result = (
                            _check_negative_funds_and_maybe_terminate(
                                env, consecutive_negative_days, previous_day_end_today_result,
                                end_today_result, day, run_log, log_path,
                            )
                        )
                        if _exit:
                            return
                        log_message(run_log, {"role": "tool", "day": day, "phase": "execution", "turn": execution_turns, "name": "end_today", "content": end_today_result.get("formatted", safe_dump(end_today_result)), "raw": end_today_result.get("result", safe_dump(end_today_result))})
                    except Exception as end_today_exc:
                        import traceback
                        err_msg = f"Failed to execute end_today after consecutive invalid tool calls: {type(end_today_exc).__name__}: {end_today_exc}"
                        print(f"[错误] {err_msg}")
                        log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(end_today_exc).__name__})
                    break

                if tool_calls_list and user_message:
                    execution_messages.append({"role": "user", "content": user_message})
                    if not execution_phase_complete:
                        note = '[Note: Use <tool_call> tags] ' if parse_method_tag == "json" else ''
                        execution_messages.append({"role": "user", "content": f"{note}Continue operations. Call end_today when done."})
                    log_message(run_log, {"role": "user", "day": day, "phase": "execution", "turn": execution_turns, "content": user_message, "parse_method": parse_method_tag})
                else:
                    execution_messages.append({"role": "user", "content": "No valid tool call detected. Continue or call end_today."})

                write_log_json_array(log_path, run_log)
                
                # 每 checkpoint_interval 步保存一次 checkpoint
                if checkpoint_dir is not None and global_turn % checkpoint_interval == 0:
                    # 保存时合并两个阶段的 messages（用于恢复）
                    combined_messages = strategy_messages + execution_messages
                    save_checkpoint(checkpoint_dir, global_turn, combined_messages, env)

            except Exception as exc:
                import traceback
                err_msg = f"Execution phase failed on day {day} turn {execution_turns}: {type(exc).__name__}: {str(exc)}"
                print(f"[严重错误] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(exc).__name__})
                write_log_json_array(log_path, run_log)
                
                if isinstance(exc, KeyError) and 'arguments' in str(exc):
                    print("[警告] 工具调用格式错误，但尝试继续执行...")
                    execution_messages.append({"role": "user", "content": f"Previous tool call had format error: {err_msg}. Please retry with correct format."})
                    continue
                else:
                    print("[停止] 遇到严重错误，停止执行")
                    return

        if not execution_phase_complete:
            print(f"[Day {day}] Forcing end_today after {max_execution_turns_per_day} turns")
            try:
                end_today_result = env.exec_tools("end_today")
                execution_phase_complete = True
                # 保存 end_today 的结果，供下一天使用
                _exit, consecutive_negative_days, previous_day_end_today_result = (
                    _check_negative_funds_and_maybe_terminate(
                        env, consecutive_negative_days, previous_day_end_today_result,
                        end_today_result, day, run_log, log_path,
                    )
                )
                if _exit:
                    return
                log_message(run_log, {"role": "system", "day": day, "message": "Forced end_today"})
            except Exception as e:
                print(f"[错误] Failed to force end_today: {e}")

        # 写入当天最终策略到单独 JSON 文件
        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                day_strategy_path = log_dir / f"day_{day}_final_strategy.json"
                day_strategy_data = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "strategy": json.loads(json.dumps(strategy_manager.strategy_store, default=str)),
                    "strategy_phase_failed": strategy_phase_failed,
                    "strategy_default_leak": strategy_default_leak,
                }
                with day_strategy_path.open("w", encoding="utf-8") as f:
                    json.dump(day_strategy_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to write day {day} final strategy to json: {exc}")

        # 保存当天的 token 使用信息
        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                day_token_path = log_dir / f"day_{day}_token_usage.json"
                day_token_data = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "prompt_tokens": day_prompt_tokens,
                    "completion_tokens": day_completion_tokens,
                    "total_tokens": day_tokens,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_token_path.open("w", encoding="utf-8") as f:
                    json.dump(day_token_data, f, ensure_ascii=False, indent=2, default=str)
                print(f"[Day {day}] Token usage: prompt={day_prompt_tokens:,}, completion={day_completion_tokens:,}, total={day_tokens:,}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to write day {day} token usage to json: {exc}")

        # 保存每天执行结束的 checkpoint
        if checkpoint_dir is not None:
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                # 合并两个阶段的 messages（用于恢复）
                combined_messages = strategy_messages + execution_messages
                day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
                day_messages_path = checkpoint_dir / f"day_{day}_messages.json"
                day_env_checkpoint_path = checkpoint_dir / f"day_{day}_env_checkpoint.json"
                
                # 保存 messages
                with day_messages_path.open("w", encoding="utf-8") as f:
                    json.dump(combined_messages, f, ensure_ascii=False, indent=2, default=str)
                
                # 保存 environment checkpoint
                env.save_checkpoint(day_env_checkpoint_path)
                
                # 保存 checkpoint 元数据
                checkpoint_metadata = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "global_turn": global_turn,
                    "strategy_turns": strategy_turns,
                    "execution_turns": execution_turns,
                    "messages_path": str(day_messages_path.relative_to(checkpoint_dir)),
                    "env_checkpoint_path": str(day_env_checkpoint_path.relative_to(checkpoint_dir)),
                    "strategy": json.loads(json.dumps(strategy_manager.strategy_store, default=str)),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
                print_env_status(env, prefix="  ")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to save day {day} checkpoint: {exc}")

        write_log_json_array(log_path, run_log)

    # 所有天数结束后，输出当前最终策略
    print("\nFinal strategy after simulation:")
    print(format_strategy_dict(strategy_manager.strategy_store))

    # 输出 token 统计
    print("\n" + "="*80)
    print("Token Usage Statistics:")
    print("="*80)
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print("="*80 + "\n")
    
    # 保存 token 统计到日志目录
    if log_dir is not None:
        try:
            token_stats_path = log_dir / "token_statistics.json"
            token_stats = {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            with token_stats_path.open("w", encoding="utf-8") as f:
                json.dump(token_stats, f, ensure_ascii=False, indent=2, default=str)
            print(f"[INFO] Token statistics saved to {token_stats_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to save token statistics: {exc}")

    # 同时把最终策略和 token 统计写入日志
    log_message(
        run_log,
        {
            "role": "system",
            "phase": "summary",
            "message": "Final strategy after simulation",
            "strategy": json.loads(json.dumps(strategy_manager.strategy_store, default=str)),
            "token_statistics": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
        },
    )
    write_log_json_array(log_path, run_log)

def build_log_path(base_dir: str = "logs") -> Path:
    """Generate a log file path using current timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(os.path.join(base_dir, f"run_env_{timestamp}/run_env_{timestamp}.json")), base_dir + f'/run_env_{timestamp}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retail environment with checkpoint support")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--recover_turn", type=int, help="Turn number to recover from (if recovering)")
    parser.add_argument("--recover_day", type=int, help="Day number to recover from (if recovering from day checkpoint)")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N turns")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS, help=f"Maximum number of turns (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records (default: 'model_run_time')")
    parser.add_argument("--config_type", type=str, choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"], default="dynamic_hard", help="Configuration type: 'dynamic_hard', 'dynamic_middle', 'still_hard', or 'still_middle' (default: 'dynamic_hard')")
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window (default: 60000)")
    parser.add_argument("--max_days", type=int, default=None, help="Maximum number of days to simulate (default: 180, or taken from dataset.json when --dataset is used)")
    parser.add_argument("--max_strategy_turns", type=int, default=10, help="Maximum turns per day in strategy phase (default: 10)")
    parser.add_argument("--max_execution_turns", type=int, default=20, help="Maximum turns per day in execution phase (default: 20)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key ")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI API base URL")
    parser.add_argument("--dataset", type=str, default=None,
        help="Path to dataset.json (UUID5-named task). Mutually exclusive with --config_type.")
    parser.add_argument("--out_dir", type=str, default=None,
        help="Output directory for tool_calls.jsonl and artifacts. Required when --dataset is used.")
    parser.add_argument("--framework", type=str, default="strategy_execute",
        choices=["react", "reflection", "strategy_execute"],
        help="Agent framework to use (default: strategy_execute)")
    parser.add_argument("--goal", type=str,
        default="Maximize net worth over the simulation horizon.",
        help="Goal string for plan_and_act framework. Ignored by react.")
    args = parser.parse_args()
    
    # -- Config loading: dataset.json path OR legacy --config_type path --
    if args.dataset:
        if not args.out_dir:
            raise ValueError("--out_dir is required when --dataset is provided")
        with open(args.dataset, encoding="utf-8") as _f:
            ds = json.load(_f)

        task_id = ds.get("task_id")
        if not task_id:
            task_id = Path(args.dataset).stem
        model_short = _resolve_model_short(args.model)

        model_dir = Path(args.out_dir) / str(task_id) / model_short
        if args.recover_day is not None or args.recover_turn is not None:
            out = _latest_run_dir(model_dir)
            if out is None:
                raise FileNotFoundError(
                    f"Recovery requested but no existing run_N directory found under {model_dir}"
                )
        else:
            out = _pick_next_run_dir(model_dir)

        config = dict(ds)                            # flat schema — dataset IS the env_config
        config["log_dir"]          = str(out)        # MUST be set before RetailEnvironment(config)
        config["order_record_dir"] = str(out / "order_records")

        (out / "dataset.json").write_text(json.dumps(ds, indent=2))
        (out / "config.json").write_text(json.dumps(config, indent=2))

        args.max_days = args.max_days or 180
        env_log_path = str(out)
        log_path = out / "run_env.json"
    else:
        # existing --config_type path — COMPLETELY UNCHANGED
        if args.config_type == "dynamic_hard":
            config = create_dynamic_hard_config()
        elif args.config_type == "dynamic_middle":
            config = create_dynamic_middle_config()
        elif args.config_type == "still_hard":
            config = create_still_hard_config()
        elif args.config_type == "still_middle":
            config = create_still_middle_config()
        else:
            config = create_dynamic_hard_config()
        config["order_record_dir"] = args.db_path if args.db_path is not None else 'model_run_time'
        log_path, env_log_path = build_log_path()
        config['log_dir'] = env_log_path
        args.max_days = args.max_days or 180  # apply default for legacy path

    # 把运行时使用的 config 复制到日志目录，便于复现
    try:
        os.makedirs(env_log_path, exist_ok=True)
        with open(os.path.join(env_log_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to write config to log dir {env_log_path}: {exc}")
    
    # Save command line arguments to log directory
    try:
        args_dict = {
            "checkpoint_dir": args.checkpoint_dir,
            "recover_turn": args.recover_turn,
            "recover_day": args.recover_day,
            "checkpoint_interval": args.checkpoint_interval,
            "model": args.model,
            "max_turns": args.max_turns,
            "db_path": args.db_path,
            "config_type": args.config_type,
            "dataset": args.dataset,
            "out_dir": args.out_dir,
            "framework": args.framework,
            "goal": args.goal,
            "max_input_tokens": args.max_input_tokens,
            "api_key": "***" if args.api_key else None,  # Don't save actual API key
            "base_url": args.base_url,
        }
        args_file = Path(env_log_path) / "args.json"
        with args_file.open("w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"[INFO] Saved command line arguments to {args_file}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to write args to log dir {env_log_path}: {exc}")
    
    # 设置 checkpoint 目录（默认使用 env_log_path）
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(env_log_path) / "checkpoints"
    
    # 创建 OpenAI client
    client = create_openai_client(api_key=args.api_key, base_url=args.base_url, model=args.model)
    
    # 如果指定了恢复 day，则从 day checkpoint 恢复（优先级高于 recover_turn）
    initial_messages = None
    start_turn = 0
    start_day = 1
    initial_strategy = None
    
    if args.recover_day is not None:
        print(f"[Checkpoint] Recovering from day {args.recover_day} checkpoint...")
        env, recovered_messages, recovered_strategy, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir, args.recover_day, config
        )
        initial_messages = recovered_messages
        initial_strategy = recovered_strategy
    elif args.recover_turn is not None:
        print(f"[Checkpoint] Recovering from turn {args.recover_turn}...")
        env, recovered_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn, config)
        initial_messages = recovered_messages
        start_turn = args.recover_turn
    else:
        env = RetailEnvironment(config)

    run_strategy_execute_loop(
        env=env,
        model=args.model,
        log_path=log_path,
        max_input_tokens=args.max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        start_day=start_day,
        initial_strategy=initial_strategy,
        log_dir=Path(env_log_path),  # Pass log directory for turn files
        max_days=args.max_days,
        max_strategy_turns_per_day=args.max_strategy_turns,
        max_execution_turns_per_day=args.max_execution_turns,
        client=client,
    )

if __name__ == "__main__":
    main()

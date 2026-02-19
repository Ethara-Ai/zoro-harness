#!/usr/bin/env python3
"""
ReAct-style loop for RetailEnvironment using OpenAI tools.

The agent can call environment tools (including the SQL DSL tool) and will receive
the formatted output each turn.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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
    format_strategy_dict,
)

def parse_tool_calls(text: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Parse tool calls from text. Returns (tool_calls_list, parse_method_tag).
    
    Supports two formats:
    1. XML format: <tool_call>{"name": "...", "arguments": {...}}</tool_call> (tag: "xml")
    2. Direct JSON format: {"name": "...", "arguments": {...}} (tag: "json")
    
    Returns:
        tuple: (list of tool calls, parse method tag)
    """
    tool_calls = []
    parse_method = "none"
    
    # First, try XML format parsing
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for json_str in matches:
        try:
            tool_calls.append(json.loads(json_str))
            parse_method = "xml"
        except json.JSONDecodeError:
            continue
    
    # If XML parsing found results, return them
    if tool_calls:
        return tool_calls, parse_method
    
    # Fallback: try to find standalone JSON objects (non-standard format)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = text[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    # Validate it has the expected structure for a tool call
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        tool_calls.append(parsed)
                        parse_method = "json"
                except (json.JSONDecodeError, KeyError):
                    pass
                start_idx = -1
    
    return tool_calls, parse_method

DEFAULT_MODEL = 'qwen3-235b-a22b-thinking-2507'

# 执行阶段的系统提示
EXECUTION_SYSTEM_PROMPT = """You are a retail operations agent executing daily operations based on a fixed strategy.

# Your Role

You will receive a **fixed strategy** that includes:
- **macro_strategy**: Broad strategic guidelines (array of strings)
- **execute_strategy**: Specific operational details (object with seven fields, all arrays)
- **today_action**: Concrete actions to take today (array of action objects)

# Strategy Usage Guidelines

**The strategy is provided as REFERENCE, but you can and should make additional actions based on real-time data:**

1. **Reference the strategy** to understand priorities and planned actions:
   - Use macro_strategy for overall decision-making direction
   - Use execute_strategy fields (focus_skus, sku_supplier_mapping, news_to_monitor, skus_to_reorder, price_adjustments, sku_to_monitor, other) as guidance
   - Consider today_action as suggested actions to take

2. **Perform additional data queries** to validate and refine decisions:
   - Check current inventory levels, sales history, supplier prices, news impacts, funds, etc.
   - Use tools like view_inventory, view_sku_sales_history, view_current_date_supplier_prices, view_news_history, etc.

3. **Execute actions flexibly**:
   - You can execute actions from today_action when they still make sense given the latest data
   - You can **adjust, skip, or modify** actions from today_action if your analysis shows better alternatives
   - You can **add additional actions** beyond today_action if needed (e.g., unexpected inventory changes, new supplier prices, or news impacts)
   - You can use information from execute_strategy (like focus_skus, sku_supplier_mapping) to make decisions even if not explicitly in today_action

4. **End the day** by calling end_today when you've completed all operations for today.

# Important Constraints

- The strategy is FIXED and cannot be modified
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

When you have completed all reasonable operations for the day (especially those in today_action, adjusted as needed by current data), you MUST call end_today to advance to the next day.
"""

DEFAULT_GOAL = (
    "Please optimize inventory assortment and turnover for long-term store viability: minimize stockouts, shrink, and cash risk while covering rent/operating costs and growing gross margin via data-driven, proactive decisions."
) 

DEFAULT_CONFIG_PATH: Path | None = None  # Use default config loader
DEFAULT_MAX_TURNS = 10000


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return create_default_config()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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



def parse_tool_args(raw: Any) -> Dict[str, Any]:
    """Safely parse tool arguments coming from the model."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def stream_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
) -> tuple[str, str, Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Stream a chat completion with tools; returns (final_content, reasoning_content, tool_calls_by_id, usage_dict).
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["\n<tool_response>", "<tool_response>"],
        extra_body={"enable_thinking": True},
        stream=True,
        stream_options={"include_usage": True},
        top_p=0.95,
        temperature=0.6,
        max_tokens=10000,
        presence_penalty=1.1,
    )

    content_parts: List[str] = []
    reasoning_content = ""
    aggregated_calls: Dict[str, Dict[str, Any]] = {}
    usage: Optional[Dict[str, Any]] = None

    for chunk in stream:
        usage_obj = getattr(chunk, "usage", None)
        if usage_obj:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }

        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            delta_reason = getattr(delta, "reasoning_content", None)
            if delta_reason:
                reasoning_content += delta_reason

            delta_content = getattr(delta, "content", None)
            if delta_content:
                if isinstance(delta_content, list):
                    content_parts.extend([str(c) for c in delta_content])
                else:
                    content_parts.append(str(delta_content))

            for tc in getattr(delta, "tool_calls", []) or []:
                tc_id = getattr(tc, "id", None) or f"call_{len(aggregated_calls)}"
                fn = getattr(tc, "function", None) or {}
                fn_name = getattr(fn, "name", "") if hasattr(fn, "name") else fn.get("name", "")
                fn_args = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else fn.get("arguments")

                entry = aggregated_calls.get(
                    tc_id,
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": ""},
                    },
                )
                if fn_name:
                    entry["function"]["name"] = fn_name
                if fn_args is not None:
                    if isinstance(fn_args, str):
                        entry["function"]["arguments"] = str(entry["function"].get("arguments", "")) + fn_args
                    else:
                        entry["function"]["arguments"] = fn_args

                aggregated_calls[tc_id] = entry

    final_content = "".join(content_parts).strip()
    return final_content, reasoning_content, aggregated_calls, usage


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

def build_goal(base_goal: str, config: Dict[str, Any]) -> str:
    """Blend the base goal with readable operational context from config."""
    store_id = config.get("store_id", "")
    begin = config.get("data_begin_time", "")
    end = config.get("data_end_time", "")
    current_date = config.get("store_begin_time", "")
    funds = config.get("initial_funds", "")

    context = (
        f"You are operating store {store_id} with initial funds of {funds}. "
        f"Your available operational data ranges from {begin} to {end}. "
        f"Today is {current_date}. "
        f"Daily Rent is {config.get('everyday_rent', 0)}."
        f"Make decisions grounded in this specific business context."
    )

    return f"{context}\n\n{base_goal}"

# Default OpenAI client configuration (can be overridden by command line arguments or environment variables)
DEFAULT_API_KEY = ''
DEFAULT_BASE_URL = ''

def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client with configurable API key and base URL.
    
    Args:
        api_key: OpenAI API key. If None, uses DEFAULT_API_KEY.
        base_url: Base URL for API. If None, uses DEFAULT_BASE_URL.
    
    Returns:
        Configured OpenAI client instance.
    """
    # Use provided values or defaults (no environment variable fallback)
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    
    return OpenAI(api_key=api_key, base_url=base_url)

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
    return env, messages


def run_react_loop(
    goal: str,
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    log_path: Path = Path("logs/run_env_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_execution_turns_per_day: int = 20,
    fixed_strategy: Dict[str, Any] = None,
    client: Optional[OpenAI] = None,
) -> None:
    # 构建工具列表
    all_tools = build_openai_tools(env)
    
    sku_desc = render_sku_descriptions(env)
    run_log: List[Dict[str, Any]] = []

    # 构建执行阶段的系统提示
    execution_system_prompt = EXECUTION_SYSTEM_PROMPT.format(
        tool_definitions=render_tool_definitions(all_tools),
    )

    consecutive_negative_days = 0
    
    # 验证固定策略
    if fixed_strategy is None:
        raise ValueError("fixed_strategy must be provided")
    
    print(f"[INFO] Using fixed strategy")
    print("Fixed strategy:")
    print(format_strategy_dict(fixed_strategy))

    input_tokens = 0
    global_turn = start_turn  # 全局 turn 计数器
    
    # Token 统计变量
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # 按天循环
    for day in range(1, max_days + 1):
        print(f"\n{'='*80}\nDay {day} - {env.current_date}\n{'='*80}\n")
        log_message(run_log, {"role": "system", "day": day, "current_date": str(env.current_date), "message": f"Day {day} started"})

        # 每天开始时初始化当天的 token 统计
        day_prompt_tokens = 0
        day_completion_tokens = 0
        day_tokens = 0

        # ========== 执行阶段 ==========
        print(f"\n[Day {day}] === EXECUTION PHASE ===")
        execution_phase_complete = False
        execution_turns = 0
        
        # 切换到执行阶段的系统提示
        execution_system_msg = {
            "role": "system",
            "content": execution_system_prompt,
        }
        
        strategy_text = format_strategy_dict(fixed_strategy, "Strategy for today:\n")
        
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
                    if len(assistant_idxs) >= 2:
                        execution_messages = execution_messages[assistant_idxs[2]:]
                        assert execution_messages[0]['role'] == 'assistant'
                
                request_messages = [execution_system_msg] + [execution_user_msg] + execution_messages
                
                try:
                    full_content, reasoning_content, aggregated_calls, usage = stream_chat(
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

                think_content = (reasoning_content or '') + (full_content or '')

                # 解析工具调用
                parse_method_tag = "none"
                try:
                    tool_calls_list, parse_method_tag = parse_tool_calls(full_content or '')
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
                        "think_content": think_content,
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
                log_message(run_log, {"role": "assistant", "day": day, "phase": "execution", "turn": execution_turns, "content": think_content, "full_content": full_content, "reasoning": reasoning_content, "tool_calls": tool_calls_list})

                execution_messages.append({
                    "role": "assistant",
                    "content": think_content,
                })

                user_message = ''
                for call in tool_calls_list:
                    if not isinstance(call, dict) or not call.get("name"):
                        err_msg = f"Invalid tool call: {call}"
                        print(f"[错误] {err_msg}")
                        log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "call": str(call)})
                        user_message += f"<tool_response>Error: {err_msg}</tool_response>\n"
                        continue

                    name, args = call.get("name"), parse_tool_args(call.get("arguments"))
                    try:
                        result = env.exec_tools(name, **args)
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
                        result_data = result.get("result", {})
                        funds = result_data.get('funds', env.funds) if isinstance(result_data, dict) else env.funds
                        if funds < 0:
                            consecutive_negative_days += 1
                            if consecutive_negative_days >= 5:
                                print(f"[运营失败] 连续 {consecutive_negative_days} 天资金为负数")
                                log_message(run_log, {"role": "system", "day": day, "message": f"运营失败：连续 {consecutive_negative_days} 天资金为负数", "funds": funds})
                                write_log_json_array(log_path, run_log)
                                return
                        else:
                            consecutive_negative_days = 0
                    
                    user_message += f"<tool_response>{result.get('formatted', safe_dump(result))}</tool_response>\n"

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
                    save_checkpoint(checkpoint_dir, global_turn, execution_messages, env)

            except Exception as exc:
                import traceback
                err_msg = f"Execution phase failed on day {day} turn {execution_turns}: {type(exc).__name__}: {str(exc)}"
                print(f"[严重错误] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "execution", "turn": execution_turns, "message": err_msg, "error_type": type(exc).__name__})
                write_log_json_array(log_path, run_log)
                
                if isinstance(exc, KeyError) and 'arguments' in str(exc):
                    print(f"[警告] 工具调用格式错误，但尝试继续执行...")
                    execution_messages.append({"role": "user", "content": f"Previous tool call had format error: {err_msg}. Please retry with correct format."})
                    continue
                else:
                    print(f"[停止] 遇到严重错误，停止执行")
                    return

        if not execution_phase_complete:
            print(f"[Day {day}] Forcing end_today after {max_execution_turns_per_day} turns")
            try:
                env.exec_tools("end_today")
                log_message(run_log, {"role": "system", "day": day, "message": "Forced end_today"})
            except Exception as e:
                print(f"[错误] Failed to force end_today: {e}")


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
                day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
                day_messages_path = checkpoint_dir / f"day_{day}_messages.json"
                day_env_checkpoint_path = checkpoint_dir / f"day_{day}_env_checkpoint.json"
                
                # 保存 messages
                with day_messages_path.open("w", encoding="utf-8") as f:
                    json.dump(execution_messages, f, ensure_ascii=False, indent=2, default=str)
                
                # 保存 environment checkpoint
                env.save_checkpoint(day_env_checkpoint_path)
                
                # 保存 checkpoint 元数据
                checkpoint_metadata = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "global_turn": global_turn,
                    "execution_turns": execution_turns,
                    "messages_path": str(day_messages_path.relative_to(checkpoint_dir)),
                    "env_checkpoint_path": str(day_env_checkpoint_path.relative_to(checkpoint_dir)),
                    "strategy": json.loads(json.dumps(fixed_strategy, default=str)),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to save day {day} checkpoint: {exc}")

        write_log_json_array(log_path, run_log)

    # 所有天数结束后，输出使用的固定策略
    print("\nFixed strategy used throughout simulation:")
    print(format_strategy_dict(fixed_strategy))

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

    # 同时把固定策略和 token 统计写入日志
    log_message(
        run_log,
        {
            "role": "system",
            "phase": "summary",
            "message": "Fixed strategy used throughout simulation",
            "strategy": json.loads(json.dumps(fixed_strategy, default=str)),
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


def load_strategy_from_file(strategy_file_path: str | Path) -> Dict[str, Any]:
    """Load strategy from a JSON file.
    
    Args:
        strategy_file_path: Path to the strategy JSON file
        
    Returns:
        Strategy dictionary extracted from the file
    """
    strategy_path = Path(strategy_file_path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
    
    with strategy_path.open("r", encoding="utf-8") as f:
        strategy_data = json.load(f)
    
    # 提取 strategy 字段，如果文件格式是 {day, current_date, strategy: {...}}
    if "strategy" in strategy_data:
        return strategy_data["strategy"]
    # 如果文件直接就是策略格式
    elif "macro_strategy" in strategy_data or "execute_strategy" in strategy_data or "today_action" in strategy_data:
        return strategy_data
    else:
        raise ValueError(f"Invalid strategy file format. Expected 'strategy' field or strategy object directly. Got keys: {list(strategy_data.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retail environment with fixed strategy (execution only)")
    parser.add_argument("--strategy_file", type=str, required=True, help="Path to the fixed strategy JSON file (e.g., logs/run_env_2025-12-25_02-46-37/day_1_final_strategy.json)")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--recover_turn", type=int, help="Turn number to recover from (if recovering)")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N turns")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS, help=f"Maximum number of turns (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records (default: 'model_run_time')")
    parser.add_argument("--config_type", type=str, choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"], default="dynamic_hard", help="Configuration type: 'dynamic_hard', 'dynamic_middle', 'still_hard', or 'still_middle' (default: 'dynamic_hard')")
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window (default: 60000)")
    parser.add_argument("--max_days", type=int, default=30, help="Maximum number of days to simulate (default: 30)")
    parser.add_argument("--max_execution_turns", type=int, default=20, help="Maximum turns per day in execution phase (default: 20)")
    parser.add_argument("--api_key", type=str, default=None, help=f"OpenAI API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--base_url", type=str, default=None, help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()
    
    # 加载固定策略
    try:
        fixed_strategy = load_strategy_from_file(args.strategy_file)
        print(f"[INFO] Loaded fixed strategy from {args.strategy_file}")
    except Exception as e:
        print(f"[ERROR] Failed to load strategy file: {e}")
        return
    
    # Load config based on config_type
    if args.config_type == "dynamic_hard":
        config = create_dynamic_hard_config()
    elif args.config_type == "dynamic_middle":
        config = create_dynamic_middle_config()
    elif args.config_type == "still_hard":
        config = create_still_hard_config()
    elif args.config_type == "still_middle":
        config = create_still_middle_config()
    else:
        # Default fallback
        config = create_dynamic_hard_config()
    
    # Use db_path if provided, otherwise use default
    config["order_record_dir"] = args.db_path if args.db_path is not None else 'model_run_time'
    log_path, env_log_path = build_log_path()  # 🚀自动生成日志路径

    goal = build_goal(DEFAULT_GOAL, config)
    config['log_dir'] = env_log_path

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
            "strategy_file": args.strategy_file,
            "checkpoint_dir": args.checkpoint_dir,
            "recover_turn": args.recover_turn,
            "checkpoint_interval": args.checkpoint_interval,
            "model": args.model,
            "max_turns": args.max_turns,
            "db_path": args.db_path,
            "config_type": args.config_type,
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
    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)
    
    # 如果指定了恢复 turn，则从 checkpoint 恢复
    initial_messages = None
    start_turn = 0
    if args.recover_turn is not None:
        print(f"[Checkpoint] Recovering from turn {args.recover_turn}...")
        env, recovered_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn, config)
        initial_messages = recovered_messages
        start_turn = args.recover_turn
    else:
        env = RetailEnvironment(config)

    run_react_loop(
        goal=goal,
        env=env,
        model=args.model,
        max_turns=args.max_turns,
        log_path=log_path,
        max_input_tokens=args.max_input_tokens,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        initial_messages=initial_messages,
        start_turn=start_turn,
        log_dir=Path(env_log_path),  # Pass log directory for turn files
        max_days=args.max_days,
        max_execution_turns_per_day=args.max_execution_turns,
        fixed_strategy=fixed_strategy,  # Pass fixed strategy
        client=client,
    )

if __name__ == "__main__":
    main()

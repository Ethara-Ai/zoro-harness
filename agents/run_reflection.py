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
from util.tool_call_parser import parse_tool_args, parse_tool_calls

from module.stream_chat import stream_chat

DEFAULT_MODEL = 'qwen3-235b-a22b-thinking-2507'

# 执行阶段的系统提示
EXECUTION_SYSTEM_PROMPT = """You are a retail operations agent executing daily operations.

# Your Role

You should:
1. **Analyze current business situation** using data tools (inventory, sales, suppliers, news, funds, etc.)
2. **Make data-driven decisions** about inventory management, pricing, and ordering
3. **Execute actions** such as placing orders and adjusting prices based on your analysis
4. **End the day** by calling end_today when all operations are complete

Your ultimate goal is to earn more profit while keeping the store operating stably.

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

When you have completed all reasonable operations for the day, you MUST call end_today to advance to the next day.
"""

DEFAULT_GOAL = (
    "Please optimize inventory assortment and turnover for long-term store viability: minimize stockouts, shrink, and cash risk while covering rent/operating costs and growing gross margin via data-driven, proactive decisions."
) 

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


def truncate_text(text: str, max_length: int = 500) -> str:
    """截断文本，如果超过最大长度则截断并添加省略号。"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def format_interaction_history(messages: List[Dict[str, Any]], max_tool_response_length: int = 1000) -> str:
    """
    格式化交互历史，截断工具返回内容。
    
    Args:
        messages: 今日的交互消息列表
        max_tool_response_length: 工具返回内容的最大长度
        
    Returns:
        格式化后的交互历史文本
    """
    history_lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "assistant":
            # 截断 assistant 的内容
            truncated_content = truncate_text(content, max_length=1000)
            history_lines.append(f"[Assistant]: {truncated_content}")
        elif role == "user":
            # 检查是否是工具响应
            if "<tool_response>" in content:
                # 提取并截断工具响应
                tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL)
                formatted_responses = []
                for resp in tool_responses:
                    truncated_resp = truncate_text(resp.strip(), max_length=max_tool_response_length)
                    formatted_responses.append(truncated_resp)
                if formatted_responses:
                    history_lines.append(f"[Tool Responses]:\n" + "\n---\n".join(formatted_responses))
            else:
                # 普通用户消息，截断
                truncated_content = truncate_text(content, max_length=500)
                history_lines.append(f"[User]: {truncated_content}")
    
    return "\n\n".join(history_lines)

def log_message(records: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """Append a timestamped record to the in-memory run log."""
    record = {"ts": datetime.utcnow().isoformat() + "Z", **payload}
    records.append(record)


def write_log_json_array(log_path: Path, records: List[Dict[str, Any]]) -> None:
    """Persist the run log as a JSON array."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)



def save_turn_calls_to_json(log_dir: Path, phase: str, turn_index: int, turn_data: Dict[str, Any], day: int) -> None:
    """Save turn calls to a separate JSON file."""
    filepath = log_dir / str(day) / f"{phase}_{day}_{turn_index}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(turn_data, f, ensure_ascii=False, indent=2, default=str)

def build_goal(base_goal: str, config: Dict[str, Any]) -> str:
    """Blend the base goal with readable operational context from config."""
    context = (
        f"You are operating store {config.get('store_id', '')} with initial funds of {config.get('initial_funds', '')}. "
        f"Your available operational data ranges from {config.get('data_begin_time', '')} to {config.get('data_end_time', '')}. "
        f"Today is {config.get('store_begin_time', '')}. Daily Rent is {config.get('everyday_rent', 0)}."
    )
    return f"{context}\n\n{base_goal}"

# Default OpenAI client configuration (can be overridden by command line arguments or environment variables)
DEFAULT_API_KEY = ''
DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

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

def save_checkpoint(checkpoint_dir: Path, turn: int, messages: List[Dict[str, Any]], env: RetailEnvironment) -> None:
    """保存 checkpoint，包括 messages 和 environment 状态。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / f"messages_turn_{turn}.json").open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
    env.save_checkpoint(checkpoint_dir / f"env_checkpoint_turn_{turn}.json")
    print(f"[Checkpoint] Saved checkpoint at turn {turn}")


def recover_from_checkpoint(checkpoint_dir: Path, turn: int, config: Dict[str, Any]) -> tuple[RetailEnvironment, List[Dict[str, Any]]]:
    """从 checkpoint 恢复环境状态和 messages。"""
    messages_path = checkpoint_dir / f"messages_turn_{turn}.json"
    env_checkpoint_path = checkpoint_dir / f"env_checkpoint_turn_{turn}.json"
    if not messages_path.exists() or not env_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for turn {turn}")
    
    with messages_path.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    env = RetailEnvironment.recover_from_checkpoint(env_checkpoint_path)
    print(f"[Checkpoint] Recovered from turn {turn}")
    return env, messages


def recover_from_day_checkpoint(checkpoint_dir: Path, day: int, config: Dict[str, Any]) -> tuple[RetailEnvironment, List[Dict[str, Any]], List[str], int, int]:
    """从 day checkpoint 恢复环境状态、messages、记忆和运行状态。"""
    day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
    if not day_checkpoint_path.exists():
        raise FileNotFoundError(f"Day checkpoint not found: {day_checkpoint_path}")
    
    with day_checkpoint_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    env = RetailEnvironment.recover_from_checkpoint(checkpoint_dir / metadata["env_checkpoint_path"])
    reflection_memory = metadata.get("reflection_memory", [])
    start_day = metadata.get("day", day) + 1
    start_turn = metadata.get("global_turn", 0)
    
    print(f"[Checkpoint] Recovered from day {day}, will start from day {start_day}, memory: {len(reflection_memory)} entries")
    return env, [], reflection_memory, start_day, start_turn


def generate_reflection(
    client: OpenAI,
    model: str,
    day: int,
    task_spec: str,
    end_today_result: Dict[str, Any],
    interaction_history: List[Dict[str, Any]],
    previous_memory: List[str],
) -> str:
    """
    生成反思文本。
    
    Args:
        client: OpenAI client
        model: 模型名称
        day: 当前天数
        task_spec: 任务描述
        end_today_result: end_today 的结果
        interaction_history: 今日的交互过程（messages 列表）
        previous_memory: 之前的记忆
        
    Returns:
        反思文本
    """
    memory_context = ""
    if previous_memory:
        # 由于每次只保留最新的反思，previous_memory 应该只有一条
        memory_context = "\n\n## Previous Reflection (Base your new reflection on this, but generate a complete new version):\n"
        memory_context += previous_memory[0] if previous_memory else ""
        memory_context += "\n\nNote: You should generate a NEW, COMPLETE reflection that incorporates learnings from the previous reflection but is a fresh, comprehensive analysis of today's performance."
    
    # 格式化交互历史，截断工具返回内容
    interaction_summary = format_interaction_history(interaction_history, max_tool_response_length=500)
    
    reflection_prompt = f"""You are a retail operations analyst reflecting on the day's performance.

# Task Goal
{task_spec}

# Day {day} End Result
{end_today_result.get('formatted', safe_dump(end_today_result))}

# Day {day} Interaction History
{interaction_summary}
{memory_context}

# Your Task

Generate a comprehensive reflection on today's performance. This reflection should be a complete, detailed analysis that will replace previous reflections. Include:

1. **Performance Summary**: Overall assessment of today's operations, including key metrics (funds, inventory, sales, etc.)

2. **Issue Identification**: What specific problems or challenges occurred? Be specific about what went wrong.

3. **Root Cause Analysis**: Why did these problems happen? Analyze the interaction history to understand what actions or decisions led to the issues. Trace back through the day's operations.

4. **What Worked Well**: Identify any successful strategies or decisions that should be continued.

5. **Actionable Improvements**: What should be done differently next time? Provide specific, actionable recommendations for future operations.

6. **Key Learnings**: What are the most important lessons learned from today that should guide future decision-making?

Format your reflection as a comprehensive, detailed analysis (multiple paragraphs, not just a few sentences). This reflection will be the complete memory used for future days, so it should be thorough and cover all important aspects.

Reflection:"""

    try:
        messages = [{"role": "user", "content": reflection_prompt}]
        full_content, final_content, reasoning_content, usage = stream_chat(client, model, messages)
        
        # 提取反思文本（去掉可能的工具调用标记）
        reflection_text = (final_content or "").strip()
        if not reflection_text:
            reflection_text = (full_content or "").strip()
        
        # 如果仍然为空，生成默认反思
        if not reflection_text:
            end_result_str = end_today_result.get('formatted', safe_dump(end_today_result))[:200]
            reflection_text = f"Day {day}: Review end result: {end_result_str}"
        
        return reflection_text
    except Exception as e:
        print(f"[WARN] Failed to generate reflection: {e}")
        end_result_str = end_today_result.get('formatted', safe_dump(end_today_result))[:200]
        return f"Day {day}: Review end result: {end_result_str}"


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
    start_day: int = 1,
    initial_memory: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_execution_turns_per_day: int = 20,
    client: Optional[OpenAI] = None,
) -> None:
    # 构建工具列表
    all_tools = build_openai_tools(env)
    
    # 如果没有启用新闻，从工具列表中移除新闻相关工具
    if not env.config.get("enable_new", False):
        news_tools = ["view_news_history", "view_today_news", "view_news_detail"]
        all_tools = [t for t in all_tools if t["function"]["name"] not in news_tools]
    
    run_log: List[Dict[str, Any]] = []

    # 构建执行阶段的系统提示
    execution_system_prompt = EXECUTION_SYSTEM_PROMPT.format(
        tool_definitions=render_tool_definitions(all_tools),
    )

    consecutive_negative_days = 0
    
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
    
    # 长期记忆：反思条目（每次生成最新的完整版本，替换而非追加）
    if initial_memory is not None and len(initial_memory) > 0:
        # 如果从 checkpoint 恢复，使用最新的记忆
        reflection_memory: List[str] = [initial_memory[-1]] if initial_memory else []
        print(f"[Checkpoint] Restored latest reflection memory from checkpoint")
    else:
        reflection_memory: List[str] = []

    # 按天循环，从 start_day 开始
    for day in range(start_day, max_days + 1):
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
        consecutive_no_valid_tool_calls_exec = 0  # 连续没有合规工具调用的次数
        
        # 切换到执行阶段的系统提示
        execution_system_msg = {
            "role": "system",
            "content": execution_system_prompt,
        }
        
        # 获取当前状态
        funds_result = env.exec_tools("view_funds_and_date")
        funds_formatted = funds_result.get("formatted", "")
        
        sku_desc = render_sku_descriptions(env)
        
        
        # 添加长期记忆到执行阶段提示中
        memory_context = ""
        if reflection_memory:
            memory_context = "\n\n## Long-term Reflection Memory (Learn from past mistakes):\n"
            for i, mem in enumerate(reflection_memory, 1):
                memory_context += f"{i}. {mem}\n"
            memory_context += "\nUse these reflections to avoid repeating past mistakes and improve your decisions.\n"

        execution_user_msg = {
            "role": "user",
            "content": f"# Day {day} - Execution Phase\n\n## Current Status\n\n{funds_formatted}\n\n## SKU Catalog\n\n{sku_desc}{memory_context}\n\n## Instructions\n\nYou should:\n1. **Analyze current business situation** using data tools (inventory, sales, suppliers, news, funds, etc.)\n2. **Make data-driven decisions** about inventory management, pricing, and ordering\n3. **Execute actions** such as placing orders and adjusting prices based on your analysis\n4. **End the day** by calling end_today when all operations are complete.\n\nYour ultimate goal is to earn more profit while keeping the store operating stably and avoid bankruptcy.",
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
                        # 保存 end_today 的结果，供反思阶段和下一天使用
                        previous_day_end_today_result = result.copy()
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
                        result_data = end_today_result.get("result", {})
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
                        previous_day_end_today_result = end_today_result
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

        # 如果执行阶段未完成，强制调用 end_today
        if not execution_phase_complete:
            print(f"[Day {day}] Forcing end_today after {max_execution_turns_per_day} turns")
            try:
                end_today_result = env.exec_tools("end_today")
                # 保存 end_today 的结果，供反思阶段和下一天使用
                if isinstance(end_today_result, dict):
                    previous_day_end_today_result = end_today_result.copy()
                log_message(run_log, {"role": "system", "day": day, "message": "Forced end_today"})
            except Exception as e:
                print(f"[错误] Failed to force end_today: {e}")
        
        # ========== 反思阶段（在执行阶段完全结束后）==========
        print(f"\n[Day {day}] === REFLECTION PHASE ===")
        # 获取 end_today 的结果（可能是正常调用或强制调用的结果）
        end_today_result_for_reflection = previous_day_end_today_result if previous_day_end_today_result is not None else None
        
        if end_today_result_for_reflection is not None:
            try:
                print(f"[Day {day}] Generating reflection...")
                reflection_text = generate_reflection(
                    client=client,
                    model=model,
                    day=day,
                    task_spec=goal,
                    end_today_result=end_today_result_for_reflection,
                    interaction_history=execution_messages,
                    previous_memory=reflection_memory,
                )
                
                # 替换长期记忆（每次生成最新的完整版本）
                reflection_memory = [reflection_text]
                
                print(f"[Day {day}] Reflection generated and replaced memory:")
                print(f"  {reflection_text[:200]}..." if len(reflection_text) > 200 else f"  {reflection_text}")
                log_message(run_log, {"role": "reflection", "day": day, "reflection": reflection_text, "memory_size": len(reflection_memory)})
                    
            except Exception as reflection_exc:
                import traceback
                err_msg = f"Reflection phase failed on day {day}: {type(reflection_exc).__name__}: {reflection_exc}"
                print(f"[WARN] {err_msg}")
                log_message(run_log, {"role": "error", "day": day, "phase": "reflection", "message": err_msg, "error_type": type(reflection_exc).__name__})
        else:
            print(f"[Day {day}] No end_today result available for reflection")
            log_message(run_log, {"role": "reflection", "day": day, "reflection": "No end_today result available"})


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
                    "reflection_memory": reflection_memory.copy(),  # 保存记忆
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to save day {day} checkpoint: {exc}")

        write_log_json_array(log_path, run_log)

    # 所有天数结束后，输出总结
    print("\nSimulation completed.")

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

    # 同时把记忆和 token 统计写入日志
    log_message(
        run_log,
        {
            "role": "system",
            "phase": "summary",
            "message": "Final reflection memory and statistics after simulation",
            "reflection_memory": reflection_memory.copy(),
            "token_statistics": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
        },
    )
    write_log_json_array(log_path, run_log)
    
    # 输出最终记忆
    print("\n" + "="*80)
    print("Final Reflection Memory:")
    print("="*80)
    if reflection_memory:
        for i, mem in enumerate(reflection_memory, 1):
            print(f"{i}. {mem}")
    else:
        print("No reflections recorded.")
    print("="*80 + "\n")

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
    parser.add_argument("--max_turns", type=int, default=50, help=f"Maximum number of turns (default: 50)")
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records (default: 'model_run_time')")
    parser.add_argument("--config_type", type=str, choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"], default="dynamic_hard", help="Configuration type: 'dynamic_hard', 'dynamic_middle', 'still_hard', or 'still_middle' (default: 'dynamic_hard')")
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window (default: 60000)")
    parser.add_argument("--max_days", type=int, default=180, help="Maximum number of days to simulate (default: 30)")
    parser.add_argument("--max_execution_turns", type=int, default=20, help="Maximum turns per day in execution phase (default: 20)")
    parser.add_argument("--api_key", type=str, default=None, help=f"OpenAI API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--base_url", type=str, default=None, help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()
    
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
            "checkpoint_dir": args.checkpoint_dir,
            "recover_turn": args.recover_turn,
            "recover_day": args.recover_day,
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
    
    # 如果指定了恢复 day，则从 day checkpoint 恢复（优先级高于 recover_turn）
    initial_messages = None
    start_turn = 0
    start_day = 1
    initial_memory = []
    
    if args.recover_day is not None:
        print(f"[Checkpoint] Recovering from day {args.recover_day} checkpoint...")
        env, recovered_messages, recovered_memory, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir, args.recover_day, config
        )
        initial_messages = recovered_messages
        initial_memory = recovered_memory
    elif args.recover_turn is not None:
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
        start_day=start_day,
        initial_memory=initial_memory,
        log_dir=Path(env_log_path),  # Pass log directory for turn files
        max_days=args.max_days,
        max_execution_turns_per_day=args.max_execution_turns,
        client=client,
    )

if __name__ == "__main__":
    main()

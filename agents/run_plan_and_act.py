#!/usr/bin/env python3
"""
Plan-and-Act loop for RetailEnvironment using OpenAI tools.

Plan phase: generate a concise plan for the day (no tool calls).
Act phase: execute the plan with tools and end the day with end_today.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


DEFAULT_MODEL = "qwen3-235b-a22b-thinking-2507"

PLAN_SYSTEM_PROMPT = """You are a retail planning agent.

Your job is to produce a concise, actionable plan for today's operations. The plan should be 3-7 steps.
Use the provided current status and yesterday's actions/results to identify priorities, risks, and quick wins.
Do NOT call tools in the plan phase. Output only the plan in plain text.
"""

EXECUTION_SYSTEM_PROMPT = """You are a retail operations agent executing the plan for today.

# Your Role

You should:
1. Analyze the current business situation using data tools (inventory, sales, suppliers, news, funds, etc.)
2. Make data-driven decisions about inventory management, pricing, and ordering
3. Execute actions such as placing orders and adjusting prices based on your analysis
4. End the day by calling end_today when all operations are complete

Follow the provided plan, but adapt if live data suggests a better action.

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
    "Please optimize inventory assortment and turnover for long-term store viability: "
    "minimize stockouts, shrink, and cash risk while covering rent/operating costs "
    "and growing gross margin via data-driven, proactive decisions."
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
    lines.append(
        "SKU catalog (grouped by category). SKU_ID is the unique product identifier. "
        "Promotion_Days indicates when the item will be discounted/cleared and should be sold before that window expires."
    )
    for category, sku_list in env.skus_category_map.items():
        lines.append(f"## Category: {category}")
        for sku in sku_list:
            desc = sku.attributes or {}
            detail = desc.get("description") or desc.get("DESCRIP") or ""
            brand = sku.brand
            promotion_days = getattr(sku, "promotion_day", None) or desc.get("PROMOTION_TIME")
            lines.append(
                f"- SKU_id={sku.sku_id}, Expiration_Days={promotion_days}, "
                f"Brand={brand}, Desc={detail}, Category={category}"
            )
        lines.append("")

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
    """Truncate text if it exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


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
    round_index: Optional[int] = None,
) -> None:
    """Save turn calls to a separate JSON file."""
    if round_index is None:
        filename = f"{phase}_{day}_{turn_index}.json"
    else:
        filename = f"{phase}_{day}_r{round_index}_t{turn_index}.json"
    filepath = log_dir / str(day) / filename
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


DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client with configurable API key and base URL.
    """
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if base_url is None:
        base_url = DEFAULT_BASE_URL

    return OpenAI(api_key=api_key, base_url=base_url)


def save_checkpoint(checkpoint_dir: Path, turn: int, messages: List[Dict[str, Any]], env: RetailEnvironment) -> None:
    """Save checkpoint: messages + environment state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / f"messages_turn_{turn}.json").open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
    env.save_checkpoint(checkpoint_dir / f"env_checkpoint_turn_{turn}.json")
    print(f"[Checkpoint] Saved checkpoint at turn {turn}")


def recover_from_checkpoint(checkpoint_dir: Path, turn: int) -> tuple[RetailEnvironment, List[Dict[str, Any]]]:
    """Recover environment state and messages from checkpoint."""
    messages_path = checkpoint_dir / f"messages_turn_{turn}.json"
    env_checkpoint_path = checkpoint_dir / f"env_checkpoint_turn_{turn}.json"
    if not messages_path.exists() or not env_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for turn {turn}")

    with messages_path.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    env = RetailEnvironment.recover_from_checkpoint(env_checkpoint_path)
    print(f"[Checkpoint] Recovered from turn {turn}")
    return env, messages


def recover_from_day_checkpoint(
    checkpoint_dir: Path,
    day: int,
) -> tuple[RetailEnvironment, List[str], Optional[str], int, int]:
    """Recover environment state and previous-day info from day checkpoint."""
    day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
    if not day_checkpoint_path.exists():
        raise FileNotFoundError(f"Day checkpoint not found: {day_checkpoint_path}")

    with day_checkpoint_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    env = RetailEnvironment.recover_from_checkpoint(checkpoint_dir / metadata["env_checkpoint_path"])
    previous_day_actions = metadata.get("previous_day_actions", [])
    previous_day_end_today_formatted = metadata.get("previous_day_end_today_formatted")
    start_day = metadata.get("day", day) + 1
    start_turn = metadata.get("global_turn", 0)

    print(
        f"[Checkpoint] Recovered from day {day}, will start from day {start_day}, "
        f"previous actions: {len(previous_day_actions)} entries"
    )
    return env, previous_day_actions, previous_day_end_today_formatted, start_day, start_turn


def generate_plan(
    client: OpenAI,
    model: str,
    day: int,
    goal: str,
    status_text: str,
    sku_desc: str,
    previous_day_actions: Optional[List[str]] = None,
    previous_day_end_today_formatted: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """
    Generate a concise plan for the day.
    Returns (plan_text, usage).
    """
    sku_desc = truncate_text(sku_desc, max_length=8000)

    if previous_day_actions:
        previous_actions_text = "\n".join(f"- {action}" for action in previous_day_actions)
        previous_actions_text = truncate_text(previous_actions_text, max_length=2000)
    else:
        previous_actions_text = "None"

    if previous_day_end_today_formatted:
        previous_end_today_text = truncate_text(previous_day_end_today_formatted, max_length=2000)
    else:
        previous_end_today_text = "None"

    plan_prompt = f"""# Task Goal
{goal}

# Day {day} Status
{status_text}

# Yesterday's Actions
{previous_actions_text}

# Yesterday's end_today Result
{previous_end_today_text}

# SKU Catalog (truncated if long)
{sku_desc}

# Instructions
Produce a short plan (3-7 steps). Do NOT call tools.
"""

    messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": plan_prompt},
    ]

    try:
        full_content, final_content, reasoning_content, usage = stream_chat(client, model, messages)
        plan_text = (final_content or "").strip() or (full_content or "").strip()
    except Exception as exc:
        print(f"[WARN] Plan generation failed: {type(exc).__name__}: {exc}")
        plan_text = ""
        usage = {}

    if not plan_text:
        plan_text = "Plan: Review status, check inventory and sales, adjust prices/orders, and end the day."

    return plan_text, usage or {}


def run_plan_and_act_loop(
    goal: str,
    env: RetailEnvironment,
    model: str = DEFAULT_MODEL,
    max_turns: int = 50,
    log_path: Path = Path("logs/run_env_history.json"),
    max_input_tokens: int = 60000,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    start_turn: int = 0,
    start_day: int = 1,
    previous_day_actions: Optional[List[str]] = None,
    previous_day_end_today_formatted: Optional[str] = None,
    log_dir: Optional[Path] = None,
    max_days: int = 30,
    max_execution_turns_per_day: int = 20,
    client: Optional[OpenAI] = None,
) -> None:
    all_tools = build_openai_tools(env)

    if not env.config.get("enable_new", False):
        news_tools = ["view_news_history", "view_today_news", "view_news_detail"]
        all_tools = [t for t in all_tools if t["function"]["name"] not in news_tools]

    run_log: List[Dict[str, Any]] = []

    execution_system_prompt = EXECUTION_SYSTEM_PROMPT.format(
        tool_definitions=render_tool_definitions(all_tools),
    )

    if client is None:
        client = create_openai_client()

    input_tokens = 0
    global_turn = start_turn

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    consecutive_negative_days = 0
    previous_day_actions = previous_day_actions or []
    max_plan_act_rounds_per_day = max_turns if max_turns and max_turns > 0 else 1

    for day in range(start_day, max_days + 1):
        print(f"\n{'='*80}\nDay {day} - {env.current_date}\n{'='*80}\n")
        log_message(run_log, {"role": "system", "day": day, "current_date": str(env.current_date), "message": f"Day {day} started"})

        day_prompt_tokens = 0
        day_completion_tokens = 0
        day_tokens = 0

        day_actions: List[str] = []
        day_round = 0
        day_execution_turns_total = 0
        day_complete = False

        while not day_complete and day_round < max_plan_act_rounds_per_day:
            day_round += 1

            # --- Plan Phase ---
            print(f"\n[Day {day}] === PLAN PHASE (Round {day_round}/{max_plan_act_rounds_per_day}) ===")

            funds_result = env.exec_tools("view_funds_and_date")
            inventory_result = env.exec_tools("view_inventory")
            orders_result = env.exec_tools("view_current_orders")

            funds_formatted = funds_result.get("formatted", "")
            inventory_formatted = truncate_text(inventory_result.get("formatted", ""), max_length=3000)
            orders_formatted = truncate_text(orders_result.get("formatted", ""), max_length=3000)

            status_text = f"{funds_formatted}\n\nInventory:\n{inventory_formatted}\n\nOpen Orders:\n{orders_formatted}"
            sku_desc = render_sku_descriptions(env)

            plan_text, plan_usage = generate_plan(
                client=client,
                model=model,
                day=day,
                goal=goal,
                status_text=status_text,
                sku_desc=sku_desc,
                previous_day_actions=previous_day_actions,
                previous_day_end_today_formatted=previous_day_end_today_formatted,
            )

            print(f"[Day {day}] Plan:\n{plan_text}\n")
            log_message(run_log, {"role": "plan", "day": day, "round": day_round, "content": plan_text})
            log_message(
                run_log,
                {
                    "role": "usage",
                    "day": day,
                    "round": day_round,
                    "phase": "plan",
                    "prompt_tokens": plan_usage.get("prompt_tokens"),
                    "completion_tokens": plan_usage.get("completion_tokens"),
                    "total_tokens": plan_usage.get("total_tokens"),
                },
            )

            if plan_usage:
                prompt_tokens = plan_usage.get("prompt_tokens", 0)
                completion_tokens = plan_usage.get("completion_tokens", 0)
                tokens = plan_usage.get("total_tokens", 0)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += tokens
                day_prompt_tokens += prompt_tokens
                day_completion_tokens += completion_tokens
                day_tokens += tokens

            if log_dir is not None:
                plan_turn_data = {
                    "day": day,
                    "round": day_round,
                    "phase": "plan",
                    "turn": 1,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "plan_text": plan_text,
                    "usage": {
                        "prompt_tokens": plan_usage.get("prompt_tokens"),
                        "completion_tokens": plan_usage.get("completion_tokens"),
                        "total_tokens": plan_usage.get("total_tokens"),
                    },
                    "status": status_text,
                }
                save_turn_calls_to_json(log_dir, "plan", 1, plan_turn_data, day, round_index=day_round)

            # --- Execution Phase ---
            print(f"\n[Day {day}] === ACT PHASE (Round {day_round}/{max_plan_act_rounds_per_day}) ===")
            execution_phase_complete = False
            execution_turns = 0
            consecutive_no_valid_tool_calls = 0

            execution_system_msg = {
                "role": "system",
                "content": execution_system_prompt,
            }

            execution_user_msg = {
                "role": "user",
                "content": (
                    f"# Day {day} - Execution Phase\n\n"
                    f"## Plan\n{plan_text}\n\n"
                    f"## Current Status\n{status_text}\n\n"
                    f"## SKU Catalog\n{sku_desc}\n\n"
                    "## Instructions\n"
                    "Follow the plan, use tools to gather data and take actions, and call end_today when done."
                ),
            }

            log_message(run_log, {**execution_system_msg, "day": day, "round": day_round, "phase": "execution"})
            log_message(run_log, {**execution_user_msg, "day": day, "round": day_round, "phase": "execution"})

            execution_messages: List[Dict[str, Any]] = []

            while not execution_phase_complete and execution_turns < max_execution_turns_per_day:
                execution_turns += 1
                global_turn += 1

                if input_tokens > max_input_tokens:
                    assistant_idxs = [i for i, m in enumerate(execution_messages) if m.get("role") == "assistant"]
                    if len(assistant_idxs) >= 3:
                        execution_messages = execution_messages[assistant_idxs[2]:]
                        assert execution_messages[0]["role"] == "assistant"
                    elif len(assistant_idxs) >= 2:
                        execution_messages = execution_messages[assistant_idxs[1]:]
                        assert execution_messages[0]["role"] == "assistant"

                request_messages = [execution_system_msg] + [execution_user_msg] + execution_messages

                try:
                    full_content, final_content, reasoning_content, usage = stream_chat(
                        client=client,
                        model=model,
                        messages=request_messages,
                    )
                except Exception as stream_exc:
                    err_msg = (
                        f"LLM stream_chat failed on day {day} execution turn {execution_turns}: "
                        f"{type(stream_exc).__name__}: {stream_exc}"
                    )
                    print(f"[ERROR] {err_msg}")
                    log_message(
                        run_log,
                        {
                            "role": "error",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "message": err_msg,
                            "error_type": type(stream_exc).__name__,
                        },
                    )
                    write_log_json_array(log_path, run_log)
                    execution_messages.append({"role": "user", "content": f"System error: {err_msg}. Continue."})
                    continue

                parse_method_tag = "none"
                try:
                    parse_source = final_content or full_content or ""
                    tool_calls_list, parse_method_tag = parse_tool_calls(parse_source)
                except Exception as parse_exc:
                    err_msg = (
                        f"Failed to parse tool calls on day {day} execution turn {execution_turns}: "
                        f"{type(parse_exc).__name__}: {parse_exc}"
                    )
                    print(f"[ERROR] {err_msg}")
                    log_message(
                        run_log,
                        {
                            "role": "error",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "message": err_msg,
                            "error_type": type(parse_exc).__name__,
                        },
                    )
                    tool_calls_list, parse_method_tag = [], "none"

                if log_dir is not None:
                    turn_data = {
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
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
                    save_turn_calls_to_json(log_dir, "execute", execution_turns, turn_data, day, round_index=day_round)

                log_message(
                    run_log,
                    {
                        "role": "usage",
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
                        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                        "completion_tokens": usage.get("completion_tokens") if usage else None,
                        "total_tokens": usage.get("total_tokens") if usage else None,
                    },
                )
                input_tokens = usage.get("prompt_tokens") if usage else input_tokens

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

                print(
                    f"[Day {day} Execution Turn {execution_turns}] tokens: "
                    f"prompt={usage.get('prompt_tokens') if usage else None}, "
                    f"completion={usage.get('completion_tokens') if usage else None}"
                )

                log_message(
                    run_log,
                    {
                        "role": "assistant",
                        "day": day,
                        "round": day_round,
                        "phase": "execution",
                        "turn": execution_turns,
                        "content": full_content,
                        "full_content": full_content,
                        "final_content": final_content,
                        "reasoning": reasoning_content,
                        "tool_calls": tool_calls_list,
                    },
                )

                execution_messages.append({"role": "assistant", "content": full_content})

                user_message = ""
                valid_tool_calls_count = 0

                for call in tool_calls_list:
                    if not isinstance(call, dict) or not call.get("name"):
                        err_msg = f"Invalid tool call: {call}"
                        print(f"[ERROR] {err_msg}")
                        log_message(
                            run_log,
                            {
                                "role": "error",
                                "day": day,
                                "round": day_round,
                                "phase": "execution",
                                "turn": execution_turns,
                                "message": err_msg,
                                "call": str(call),
                            },
                        )
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
                        err_msg = f"Error executing tool {name}: {type(exc).__name__}: {exc}"
                        print(f"[ERROR] {err_msg}")
                        if hasattr(env, "logger"):
                            env.logger.error(err_msg)
                        result = {"formatted": err_msg, "result": {"error": str(exc), "error_type": type(exc).__name__}}

                    log_message(
                        run_log,
                        {
                            "role": "tool",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "name": name,
                            "args": args,
                            "content": result.get("formatted", safe_dump(result)),
                            "raw": result.get("result", safe_dump(result)),
                        },
                    )

                    if name == "end_today":
                        execution_phase_complete = True
                        day_complete = True
                        print(f"[Day {day}] Execution phase complete - end_today called")
                        previous_day_end_today_formatted = result.get("formatted") or safe_dump(result)
                        result_data = result.get("result", {})
                        funds = result_data.get("funds", env.funds) if isinstance(result_data, dict) else env.funds
                        if funds < 0:
                            consecutive_negative_days += 1
                            if consecutive_negative_days >= 5:
                                print(f"[FAILURE] {consecutive_negative_days} consecutive days with negative funds")
                                log_message(
                                    run_log,
                                    {
                                        "role": "system",
                                        "day": day,
                                        "round": day_round,
                                        "message": f"Failure: {consecutive_negative_days} consecutive days with negative funds",
                                        "funds": funds,
                                    },
                                )
                                write_log_json_array(log_path, run_log)
                                return
                        else:
                            consecutive_negative_days = 0

                    if tool_executed_successfully:
                        valid_tool_calls_count += 1
                        day_actions.append(f"{name}({safe_dump(args)})")

                    user_message += f"<tool_response>{result.get('formatted', safe_dump(result))}</tool_response>\n"

                if valid_tool_calls_count > 0:
                    consecutive_no_valid_tool_calls = 0
                else:
                    consecutive_no_valid_tool_calls += 1
                    print(
                        f"[Day {day} Execution Turn {execution_turns}] "
                        f"No valid tool calls (consecutive: {consecutive_no_valid_tool_calls}/5)"
                    )

                if consecutive_no_valid_tool_calls >= 5 and not execution_phase_complete:
                    print(
                        f"[Day {day}] Execution round {day_round} ended: "
                        "5 consecutive turns with no valid tool calls"
                    )
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": "Execution round ended: 5 consecutive turns with no valid tool calls",
                        },
                    )
                    break

                if tool_calls_list and user_message:
                    execution_messages.append({"role": "user", "content": user_message})
                    if not execution_phase_complete:
                        note = "[Note: Use <tool_call> tags] " if parse_method_tag == "json" else ""
                        execution_messages.append({"role": "user", "content": f"{note}Continue operations. Call end_today when done."})
                    log_message(
                        run_log,
                        {
                            "role": "user",
                            "day": day,
                            "round": day_round,
                            "phase": "execution",
                            "turn": execution_turns,
                            "content": user_message,
                            "parse_method": parse_method_tag,
                        },
                    )
                else:
                    execution_messages.append({"role": "user", "content": "No valid tool call detected. Continue or call end_today."})

                write_log_json_array(log_path, run_log)

                if checkpoint_dir is not None and global_turn % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_dir, global_turn, execution_messages, env)

            day_execution_turns_total += execution_turns

            if not execution_phase_complete:
                if execution_turns >= max_execution_turns_per_day:
                    print(
                        f"[Day {day}] Execution round {day_round} ended: "
                        f"reached max turns ({max_execution_turns_per_day})"
                    )
                    log_message(
                        run_log,
                        {
                            "role": "system",
                            "day": day,
                            "round": day_round,
                            "message": f"Execution round ended: reached max turns ({max_execution_turns_per_day})",
                        },
                    )
                print(f"[Day {day}] Round {day_round} ended without end_today; re-planning")
                log_message(
                    run_log,
                    {
                        "role": "system",
                        "day": day,
                        "round": day_round,
                        "message": "Round ended without end_today; re-planning",
                    },
                )

        if not day_complete:
            print(f"[Day {day}] Forcing end_today after {max_plan_act_rounds_per_day} rounds")
            try:
                end_today_result = env.exec_tools("end_today")
                previous_day_end_today_formatted = end_today_result.get("formatted") or safe_dump(end_today_result)
                log_message(run_log, {"role": "system", "day": day, "message": "Forced end_today"})
                _ = end_today_result
            except Exception as e:
                print(f"[ERROR] Failed to force end_today after max rounds: {e}")

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
                print(
                    f"[Day {day}] Token usage: "
                    f"prompt={day_prompt_tokens:,}, completion={day_completion_tokens:,}, total={day_tokens:,}"
                )
            except Exception as exc:
                print(f"[WARN] Failed to write day {day} token usage to json: {exc}")

        write_log_json_array(log_path, run_log)

        # Store previous day summary for next day's plan
        previous_day_actions = day_actions

        # Save day checkpoint
        if checkpoint_dir is not None:
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                day_checkpoint_path = checkpoint_dir / f"day_{day}_checkpoint.json"
                day_messages_path = checkpoint_dir / f"day_{day}_messages.json"
                day_env_checkpoint_path = checkpoint_dir / f"day_{day}_env_checkpoint.json"

                with day_messages_path.open("w", encoding="utf-8") as f:
                    json.dump(execution_messages, f, ensure_ascii=False, indent=2, default=str)

                env.save_checkpoint(day_env_checkpoint_path)

                checkpoint_metadata = {
                    "day": day,
                    "current_date": str(env.current_date),
                    "global_turn": global_turn,
                    "execution_turns": day_execution_turns_total,
                    "messages_path": str(day_messages_path.relative_to(checkpoint_dir)),
                    "env_checkpoint_path": str(day_env_checkpoint_path.relative_to(checkpoint_dir)),
                    "previous_day_actions": previous_day_actions,
                    "previous_day_end_today_formatted": previous_day_end_today_formatted,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                with day_checkpoint_path.open("w", encoding="utf-8") as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2, default=str)

                print(f"[Checkpoint] Saved day {day} checkpoint to {checkpoint_dir}")
            except Exception as exc:
                print(f"[WARN] Failed to save day {day} checkpoint: {exc}")

    print("\nSimulation completed.")

    print("\n" + "=" * 80)
    print("Token Usage Statistics:")
    print("=" * 80)
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print("=" * 80 + "\n")

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
        except Exception as exc:
            print(f"[WARN] Failed to save token statistics: {exc}")


def build_log_path(base_dir: str = "logs") -> tuple[Path, str]:
    """Generate a log file path using current timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(os.path.join(base_dir, f"run_plan_{timestamp}/run_plan_{timestamp}.json")), base_dir + f"/run_plan_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retail environment with Plan-and-Act loop")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--recover_turn", type=int, help="Turn number to recover from (if recovering)")
    parser.add_argument("--recover_day", type=int, help="Day number to recover from (if recovering from day checkpoint)")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N turns")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_turns", type=int, default=50, help="Maximum plan-act rounds per day")
    parser.add_argument("--db_path", type=str, default=None, help="Database path for order records (default: 'model_run_time')")
    parser.add_argument(
        "--config_type",
        type=str,
        choices=["dynamic_hard", "dynamic_middle", "still_hard", "still_middle"],
        default="dynamic_hard",
        help="Configuration type: 'dynamic_hard', 'dynamic_middle', 'still_hard', or 'still_middle'",
    )
    parser.add_argument("--max_input_tokens", type=int, default=50000, help="Maximum input tokens for context window")
    parser.add_argument("--max_days", type=int, default=30, help="Maximum number of days to simulate")
    parser.add_argument("--max_execution_turns", type=int, default=20, help="Maximum turns per execution round")
    parser.add_argument("--api_key", type=str, default=None, help=f"OpenAI API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--base_url", type=str, default=None, help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

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

    config["order_record_dir"] = args.db_path if args.db_path is not None else "model_run_time"
    log_path, env_log_path = build_log_path()

    goal = build_goal(DEFAULT_GOAL, config)
    config["log_dir"] = env_log_path

    try:
        os.makedirs(env_log_path, exist_ok=True)
        with open(os.path.join(env_log_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] Failed to write config to log dir {env_log_path}: {exc}")

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
            "max_days": args.max_days,
            "max_execution_turns": args.max_execution_turns,
            "api_key": "***" if args.api_key else None,
            "base_url": args.base_url,
        }
        args_file = Path(env_log_path) / "args.json"
        with args_file.open("w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"[INFO] Saved command line arguments to {args_file}")
    except Exception as exc:
        print(f"[WARN] Failed to write args to log dir {env_log_path}: {exc}")

    client = create_openai_client(api_key=args.api_key, base_url=args.base_url)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(env_log_path) / "checkpoints"

    initial_messages = None
    start_turn = 0
    start_day = 1
    previous_day_actions: List[str] = []
    previous_day_end_today_formatted: Optional[str] = None

    if args.recover_day is not None:
        print(f"[Checkpoint] Recovering from day {args.recover_day} checkpoint...")
        env, previous_day_actions, previous_day_end_today_formatted, start_day, start_turn = recover_from_day_checkpoint(
            checkpoint_dir, args.recover_day
        )
    elif args.recover_turn is not None:
        print(f"[Checkpoint] Recovering from turn {args.recover_turn}...")
        env, recovered_messages = recover_from_checkpoint(checkpoint_dir, args.recover_turn)
        initial_messages = recovered_messages
        start_turn = args.recover_turn
    else:
        env = RetailEnvironment(config)

    run_plan_and_act_loop(
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
        previous_day_actions=previous_day_actions,
        previous_day_end_today_formatted=previous_day_end_today_formatted,
        log_dir=Path(env_log_path),
        max_days=args.max_days,
        max_execution_turns_per_day=args.max_execution_turns,
        client=client,
    )


if __name__ == "__main__":
    main()

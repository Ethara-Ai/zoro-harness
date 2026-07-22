from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Union

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from claude_bridge.credentials import (
    CredentialProvider,
    CredentialsError,
    MultiAccountCredentialProvider,
    PermanentCredentialsError,
)
from claude_bridge.errors import (
    Classification,
    classify_anthropic_error,
    to_openai_error_body,
)

_LOG = logging.getLogger(__name__)

UPSTREAM_DEFAULT = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
# Upstream rejects requests missing any beta Claude Code normally sends; overridable for forward-compat.
OAUTH_BETA = os.environ.get(
    "ZORO_CC_ANTHROPIC_BETA",
    ",".join([
        "oauth-2025-04-20",
        "claude-code-20250219",
        "interleaved-thinking-2025-05-14",
        "fine-grained-tool-streaming-2025-05-14",
    ]),
)
SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Models that 400 on `thinking.budget_tokens` (non-extended-thinking tiers).
NO_THINKING_MODELS = {
    "claude-haiku-4-5-20251001",
}

DEFAULT_MAX_TOKENS = 8192
DEFAULT_THINKING_BUDGET = 8192

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus":   "claude-opus-4-8",
    "haiku":  "claude-haiku-4-5-20251001",
}

_OPENAI_ONLY_KEYS = {
    "n", "logprobs", "top_logprobs", "logit_bias", "presence_penalty",
    "frequency_penalty", "seed", "response_format", "user", "functions",
    "function_call", "service_tier",
}

ProviderLike = Union[CredentialProvider, MultiAccountCredentialProvider]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (ValueError, TypeError):
        return default


def _upstream_base() -> str:
    return os.environ.get("ZORO_CC_UPSTREAM", UPSTREAM_DEFAULT).rstrip("/")


def _timeout(streaming: bool = False) -> httpx.Timeout:
    connect = _env_float("ZORO_BRIDGE_CONNECT_TIMEOUT", 30.0)
    if streaming:
        read = _env_float("ZORO_BRIDGE_STREAM_READ_TIMEOUT", 600.0)
        return httpx.Timeout(None, connect=connect, read=read, write=None, pool=None)
    total = _env_float("ZORO_BRIDGE_REQUEST_TIMEOUT", 600.0)
    read = _env_float("ZORO_BRIDGE_READ_TIMEOUT", 180.0)
    return httpx.Timeout(total, connect=connect, read=read)


def _map_model(name: str) -> str:
    if not name:
        return MODEL_ALIASES["sonnet"]
    if name.startswith("claude-"):
        return name
    return MODEL_ALIASES.get(name.lower(), name)


def _map_finish_reason(stop_reason: str | None) -> str:
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", "stop")


def _string_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
        return "\n".join(parts)
    return ""


def _translate_tools_to_anthropic(chat_tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(chat_tools, list) or not chat_tools:
        return None
    out: list[dict[str, Any]] = []
    for t in chat_tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        entry: dict[str, Any] = {"name": name}
        desc = fn.get("description")
        if isinstance(desc, str) and desc:
            entry["description"] = desc
        params = fn.get("parameters")
        if isinstance(params, dict):
            schema = dict(params)
            if schema.get("type") != "object":
                schema["type"] = "object"
            entry["input_schema"] = schema
        else:
            entry["input_schema"] = {"type": "object", "properties": {}}
        out.append(entry)
    return out or None


def _translate_tool_choice_to_anthropic(tc: Any, has_tools: bool) -> dict[str, Any] | None:
    if tc is None or not has_tools:
        return None
    if isinstance(tc, str):
        if tc == "auto":
            return {"type": "auto"}
        if tc == "required":
            return {"type": "any"}
        if tc == "none":
            return {"type": "none"}
        return None
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
    return None


def _extract_system_and_messages(
    openai_messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    emitted_tool_use_ids: set[str] = set()

    def _last_user_content() -> list[dict[str, Any]] | None:
        if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list):
            return out[-1]["content"]
        return None

    def _last_assistant_content() -> list[dict[str, Any]] | None:
        if out and out[-1].get("role") == "assistant" and isinstance(out[-1].get("content"), list):
            return out[-1]["content"]
        return None

    for m in openai_messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            text = _string_content(content)
            if text:
                system_parts.append(text)
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _string_content(content)
            if text:
                blocks.append({"type": "text", "text": text})
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict) or tc.get("type") not in (None, "function"):
                        continue
                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    call_id = tc.get("id")
                    if not isinstance(call_id, str) or not call_id:
                        _LOG.warning(
                            "dropping assistant tool_call with missing/empty id (name=%s)",
                            name,
                        )
                        continue
                    raw_args = fn.get("arguments")
                    parsed_input: Any
                    if raw_args is None or raw_args == "":
                        parsed_input = {}
                    elif isinstance(raw_args, str):
                        try:
                            parsed_input = json.loads(raw_args)
                        except (ValueError, TypeError):
                            parsed_input = {}
                    elif isinstance(raw_args, dict):
                        parsed_input = raw_args
                    else:
                        parsed_input = {}
                    if not isinstance(parsed_input, dict):
                        _LOG.warning(
                            "assistant tool_call arguments not JSON-object; coercing to {} (name=%s)",
                            name,
                        )
                        parsed_input = {}
                    emitted_tool_use_ids.add(call_id)
                    blocks.append({
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": parsed_input,
                    })
            if blocks:
                last_assistant = _last_assistant_content()
                if last_assistant is not None:
                    last_assistant.extend(blocks)
                else:
                    out.append({"role": "assistant", "content": blocks})
            elif isinstance(tool_calls, list) and tool_calls:
                _LOG.warning(
                    "dropping assistant message: all %d tool_calls were malformed",
                    len(tool_calls),
                )
            continue

        if role == "tool":
            call_id = m.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                _LOG.warning("dropping role=tool message without tool_call_id")
                continue
            if call_id not in emitted_tool_use_ids:
                _LOG.warning(
                    "dropping orphan role=tool message (tool_call_id=%s has no matching prior assistant tool_use)",
                    call_id,
                )
                continue
            result_text = _string_content(content)
            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": result_text,
            }
            last_content = _last_user_content()
            if last_content is not None:
                insert_pos = 0
                for i, blk in enumerate(last_content):
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        insert_pos = i + 1
                    else:
                        break
                last_content.insert(insert_pos, tool_result_block)
            else:
                out.append({"role": "user", "content": [tool_result_block]})
            continue

        if role in ("user", "function"):
            text = _string_content(content)
            if not text:
                continue
            text_block = {"type": "text", "text": text}
            last_content = _last_user_content()
            if last_content is not None:
                last_content.append(text_block)
            else:
                out.append({"role": "user", "content": [text_block]})
            continue

    return "\n\n".join(p for p in system_parts if p), out


def _inject_system_prefix(system_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_text = ""
    for blk in system_blocks:
        if isinstance(blk, dict) and blk.get("type") == "text":
            first_text = blk.get("text", "")
            break
    if first_text.startswith(SYSTEM_PREFIX):
        return system_blocks
    return [{"type": "text", "text": SYSTEM_PREFIX}, *system_blocks]


def translate_openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    model = _map_model(body.get("model", ""))
    system_text, messages = _extract_system_and_messages(body.get("messages") or [])

    system_blocks: list[dict[str, Any]] = []
    if system_text:
        system_blocks.append({"type": "text", "text": system_text})

    if os.environ.get("ZORO_CC_SKIP_SYSTEM_PREFIX") != "1":
        system_blocks = _inject_system_prefix(system_blocks)

    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    ant: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system_blocks:
        ant["system"] = system_blocks

    # Sampling params are set after the thinking block below, because Anthropic's
    # rules depend on whether extended thinking is enabled.

    stop = body.get("stop")
    if stop is not None:
        ant["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    if body.get("stream"):
        ant["stream"] = True

    extra = body.get("extra_body") or {}
    if isinstance(extra, dict) and extra.get("enable_thinking"):
        if model in NO_THINKING_MODELS:
            _LOG.warning("enable_thinking requested for %s which does not support thinking; dropping", model)
        else:
            budget = int(extra.get("thinking_budget") or DEFAULT_THINKING_BUDGET)
            # Anthropic requires budget_tokens strictly less than max_tokens.
            if budget >= max_tokens:
                budget = max(1024, max_tokens - 1)
                _LOG.warning(
                    "thinking_budget >= max_tokens (%d >= %d); clamped to %d",
                    int(extra.get("thinking_budget") or DEFAULT_THINKING_BUDGET), max_tokens, budget,
                )
            ant["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Anthropic rejects temperature+top_p together, and with extended thinking it
    # requires temperature=1 and forbids top_p. So: with thinking on, drop both
    # (Anthropic defaults temperature to the required 1); otherwise forward exactly
    # one, preferring temperature.
    if "thinking" not in ant:
        if body.get("temperature") is not None:
            ant["temperature"] = body["temperature"]
        elif body.get("top_p") is not None:
            ant["top_p"] = body["top_p"]

    translated_tools = _translate_tools_to_anthropic(body.get("tools"))
    if translated_tools:
        ant["tools"] = translated_tools
        translated_tool_choice: dict[str, Any] | None = _translate_tool_choice_to_anthropic(
            body.get("tool_choice"), has_tools=True,
        )
        parallel = body.get("parallel_tool_calls")
        if translated_tool_choice is None and parallel is False:
            translated_tool_choice = {"type": "auto"}
        if translated_tool_choice is not None:
            if "thinking" in ant and translated_tool_choice.get("type") in ("any", "tool"):
                _LOG.warning(
                    "tool_choice=%s incompatible with extended thinking; demoting to auto",
                    translated_tool_choice.get("type"),
                )
                translated_tool_choice = {"type": "auto"}
            if parallel is False and translated_tool_choice["type"] in ("auto", "any", "tool"):
                translated_tool_choice["disable_parallel_tool_use"] = True
            ant["tool_choice"] = translated_tool_choice

    dropped = [k for k in _OPENAI_ONLY_KEYS if k in body and body[k] is not None]
    if dropped:
        _LOG.warning("dropping unsupported OpenAI-only params: %s", ",".join(sorted(dropped)))

    return ant


def _build_forward_headers(access_token: str, streaming: bool) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": OAUTH_BETA,
        "content-type": "application/json",
        "accept": "text/event-stream" if streaming else "application/json",
    }


def _anthropic_to_openai_nonstream(
    ant_body: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for blk in ant_body.get("content") or []:
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")
        if t == "text" and isinstance(blk.get("text"), str):
            text_parts.append(blk["text"])
        elif t == "thinking" and isinstance(blk.get("thinking"), str):
            thinking_parts.append(blk["thinking"])
        elif t == "tool_use":
            call_id = blk.get("id")
            name = blk.get("name")
            if not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            raw_input = blk.get("input")
            if raw_input is None:
                args_str = "{}"
            elif isinstance(raw_input, str):
                args_str = raw_input
            else:
                try:
                    args_str = json.dumps(raw_input, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_str = "{}"
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })

    usage = ant_body.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_creation
    completion_tokens = output_tokens

    text = "".join(text_parts)
    stripped_text = text.strip()
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text if stripped_text else (None if tool_calls else ""),
    }
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = ant_body.get("stop_reason")
    if tool_calls and stop_reason == "max_tokens":
        finish_reason = "length"
    elif tool_calls:
        finish_reason = "tool_calls"
    elif stop_reason == "tool_use":
        finish_reason = "stop"
    else:
        finish_reason = _map_finish_reason(stop_reason)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _openai_chunk_skeleton(chunk_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }


def _parse_sse_event(block: str) -> tuple[str | None, dict[str, Any] | None]:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in block.split("\n"):
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    if not data_lines:
        return event_name, None
    data_str = "\n".join(data_lines)
    if not data_str or data_str == "[DONE]":
        return event_name, None
    try:
        return event_name, json.loads(data_str)
    except (ValueError, TypeError):
        return event_name, None


async def _iter_sse_blocks(
    byte_iter: AsyncIterator[bytes],
    prebuffered: bytes = b"",
) -> AsyncIterator[str]:
    buf = prebuffered.decode("utf-8", errors="replace")
    async for raw in byte_iter:
        buf += raw.decode("utf-8", errors="replace")
        while True:
            idx = buf.find("\n\n")
            if idx < 0:
                break
            block, buf = buf[:idx], buf[idx + 2:]
            block = block.strip("\r")
            if block:
                yield block
    tail = buf.strip()
    if tail:
        yield tail


def _classify_sse_error_event(data: dict[str, Any]) -> Classification:
    err = data.get("error") if isinstance(data, dict) else None
    err_type = (err or {}).get("type") if isinstance(err, dict) else None
    message = (err or {}).get("message") if isinstance(err, dict) else None
    message = message or err_type or "upstream stream error"
    if err_type in ("overloaded_error", "api_error"):
        return Classification("transient", 503, None, None, err_type, message, None)
    if err_type == "rate_limit_error":
        return Classification("cap", 429, 300.0, time.time() + 300.0, err_type, message, None)
    if err_type in ("authentication_error", "permission_error"):
        return Classification("auth", 401, None, None, err_type, message, None)
    if err_type == "invalid_request_error":
        return Classification("client", 400, None, None, err_type, message, None)
    return Classification("transient", 502, None, None, err_type, message, None)


async def _translate_stream(
    byte_iter: AsyncIterator[bytes],
    requested_model: str,
    *,
    prebuffered: bytes = b"",
    provider: ProviderLike | None = None,
) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    stop_reason: str | None = None
    final_usage: dict[str, int] | None = None
    initial_usage: dict[str, int] = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    sent_role = False
    active_block_types: dict[int, str] = {}
    tool_call_index_map: dict[int, int] = {}
    tool_call_completed: set[int] = set()
    tool_call_saw_delta: set[int] = set()
    saw_tool_call = False

    def _emit_role_chunk() -> bytes:
        payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
        payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        return _sse_chunk(payload)

    def _tool_calls_incomplete() -> bool:
        return bool(tool_call_index_map) and not set(tool_call_index_map.keys()).issubset(tool_call_completed)

    async for block in _iter_sse_blocks(byte_iter, prebuffered=prebuffered):
        event_name, data = _parse_sse_event(block)
        if data is None:
            continue
        t = data.get("type") or event_name

        if t == "error":
            cls = _classify_sse_error_event(data)
            _LOG.error(
                "mid-stream error event: kind=%s type=%s msg=%s",
                cls.kind, cls.error_type, cls.message,
            )
            if provider is not None:
                await _apply_classification(provider, cls)
            if not sent_role:
                yield _emit_role_chunk()
                sent_role = True
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{
                "index": 0,
                "delta": {"content": f"\n\n[bridge] upstream {cls.error_type or 'error'}: {cls.message}"},
                "finish_reason": None,
            }]
            yield _sse_chunk(payload)
            if saw_tool_call and stop_reason == "max_tokens":
                err_fr = "length"
            elif saw_tool_call and not _tool_calls_incomplete():
                err_fr = "tool_calls"
            else:
                err_fr = "stop"
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": err_fr}]
            yield _sse_chunk(payload)
            yield b"data: [DONE]\n\n"
            return

        if t == "message_start":
            msg = data.get("message") or {}
            u = msg.get("usage") or {}
            initial_usage = {
                "input_tokens": int(u.get("input_tokens") or 0),
                "cache_read_input_tokens": int(u.get("cache_read_input_tokens") or 0),
                "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens") or 0),
            }
            if not sent_role:
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                yield _sse_chunk(payload)
                sent_role = True

        elif t == "content_block_start":
            idx = int(data.get("index", 0))
            blk = data.get("content_block") or {}
            blk_type = blk.get("type", "")
            active_block_types[idx] = blk_type
            if blk_type == "tool_use":
                tc_idx = len(tool_call_index_map)
                tool_call_index_map[idx] = tc_idx
                saw_tool_call = True
                if not sent_role:
                    yield _emit_role_chunk()
                    sent_role = True
                call_id = blk.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                name = blk.get("name")
                if not isinstance(name, str):
                    name = ""
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": tc_idx,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]},
                    "finish_reason": None,
                }]
                yield _sse_chunk(payload)

        elif t == "content_block_delta":
            idx = int(data.get("index", 0))
            delta = data.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                text = delta.get("text", "")
                if text:
                    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                    payload["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                    yield _sse_chunk(payload)
            elif dt == "thinking_delta":
                text = delta.get("thinking", "")
                if text:
                    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                    payload["choices"] = [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]
                    yield _sse_chunk(payload)
            elif dt == "input_json_delta":
                partial = delta.get("partial_json", "")
                if idx in tool_call_index_map and isinstance(partial, str) and partial:
                    tool_call_saw_delta.add(idx)
                    tc_idx = tool_call_index_map[idx]
                    saw_tool_call = True
                    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                    payload["choices"] = [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": tc_idx,
                            "function": {"arguments": partial},
                        }]},
                        "finish_reason": None,
                    }]
                    yield _sse_chunk(payload)

        elif t == "content_block_stop":
            idx = int(data.get("index", 0))
            if idx in tool_call_index_map:
                if idx not in tool_call_saw_delta:
                    tc_idx = tool_call_index_map[idx]
                    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                    payload["choices"] = [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": tc_idx,
                            "function": {"arguments": "{}"},
                        }]},
                        "finish_reason": None,
                    }]
                    yield _sse_chunk(payload)
                tool_call_completed.add(idx)
            active_block_types.pop(idx, None)

        elif t == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            u = data.get("usage") or {}
            output_tokens = int(u.get("output_tokens") or 0)
            prompt_tokens = (
                initial_usage["input_tokens"]
                + initial_usage["cache_read_input_tokens"]
                + initial_usage["cache_creation_input_tokens"]
            )
            final_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            }

        elif t == "message_stop":
            if saw_tool_call and stop_reason == "max_tokens":
                fr = "length"
            elif saw_tool_call and not _tool_calls_incomplete():
                fr = "tool_calls"
            else:
                fr = _map_finish_reason(stop_reason)
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": fr}]
            yield _sse_chunk(payload)
            if final_usage is not None:
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = []
                payload["usage"] = final_usage
                yield _sse_chunk(payload)
            yield b"data: [DONE]\n\n"
            return

    _LOG.error(
        "upstream SSE stream ended before message_stop (stop_reason=%s); emitting finish_reason=length",
        stop_reason,
    )
    if provider is not None and stop_reason is None:
        cls = Classification("transient", 502, 60.0, time.time() + 60.0, None, "stream truncated", None)
        await _apply_classification(provider, cls)
    if not sent_role:
        payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
        payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield _sse_chunk(payload)
    if saw_tool_call and stop_reason == "max_tokens":
        trailing_finish = "length"
    elif saw_tool_call and not _tool_calls_incomplete():
        trailing_finish = "tool_calls"
    elif stop_reason is None:
        trailing_finish = "length"
    else:
        trailing_finish = _map_finish_reason(stop_reason)
    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
    payload["choices"] = [{
        "index": 0,
        "delta": {},
        "finish_reason": trailing_finish,
    }]
    yield _sse_chunk(payload)
    if final_usage is not None:
        payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
        payload["choices"] = []
        payload["usage"] = final_usage
        yield _sse_chunk(payload)
    yield b"data: [DONE]\n\n"


async def _get_token(provider: ProviderLike) -> str:
    return await asyncio.to_thread(provider.get_access_token)


async def _apply_classification(
    provider: ProviderLike,
    cls: Classification,
) -> None:
    # `auth` intentionally not handled: only a failed OAuth refresh proves the token is dead.
    if isinstance(provider, MultiAccountCredentialProvider):
        if cls.kind == "cap":
            provider.mark_exhausted(cls.retry_after_s or 300.0)
    if cls.kind == "cap":
        provider.last_cap_reset_at = cls.reset_at_unix or (
            time.time() + (cls.retry_after_s or 300.0)
        )


def _openai_error_response(status_code: int, message: str, type_: str | None = None) -> JSONResponse:
    body = to_openai_error_body(status_code, message, type_)
    headers: dict[str, str] = {}
    return JSONResponse(body, status_code=status_code, headers=headers)


def _openai_error_from_classification(cls: Classification) -> JSONResponse:
    headers: dict[str, str] = {}
    if cls.retry_after_s is not None:
        headers["Retry-After"] = str(max(1, int(cls.retry_after_s)))
    body = to_openai_error_body(cls.status_code, cls.message, cls.error_type)
    return JSONResponse(body, status_code=cls.status_code, headers=headers)


def build_app(provider: ProviderLike | None = None) -> FastAPI:
    if provider is None:
        from claude_bridge.credentials import resolve_provider
        provider = resolve_provider()
    prov: ProviderLike = provider

    bridge_secret = os.environ.get("ZORO_CC_BRIDGE_SECRET", "").strip()
    if not bridge_secret:
        _LOG.warning(
            "ZORO_CC_BRIDGE_SECRET is not set — the bridge is UNAUTHENTICATED; "
            "any local process can spend this subscription."
        )

    max_inline_retries = _env_int("ZORO_CC_MAX_INLINE_RETRIES", 3)
    max_inline_wait = _env_int("ZORO_CC_MAX_INLINE_WAIT", 30)
    max_stream_retries = _env_int("ZORO_CC_STREAM_BUFFER_RETRIES", 3)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=_timeout())
        app.state.http_stream = httpx.AsyncClient(timeout=_timeout(streaming=True))
        try:
            yield
        finally:
            await app.state.http.aclose()
            await app.state.http_stream.aclose()

    app = FastAPI(title="Zoro Claude Code Bridge", version="0.1.0", lifespan=lifespan)

    def _authorized(request: Request) -> bool:
        if not bridge_secret:
            return True
        presented = (
            request.headers.get("x-zoro-bridge-secret")
            or request.headers.get("x-api-key")
            or ""
        )
        if not presented:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                presented = auth[7:].strip()
        return hmac.compare_digest(presented, bridge_secret)

    def _unauthorized() -> JSONResponse:
        return _openai_error_response(401, "missing or invalid bridge secret", "authentication_error")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        tp = None
        sub = None
        try:
            tp = prov.token_prefix()
        except Exception:  # noqa: BLE001
            tp = None
        try:
            sub = prov.subscription_type()
        except Exception:  # noqa: BLE001
            sub = None
        return {"ok": True, "token_prefix": tp, "subscription": sub}

    @app.get("/quota")
    async def quota(request: Request) -> dict[str, Any]:
        if not _authorized(request):
            return {"accounts": []}
        if isinstance(prov, MultiAccountCredentialProvider):
            return {
                "multi_account": True,
                "accounts": prov.snapshot(),
                "next_reset_at_unix": prov.next_reset_at(),
            }
        reset = getattr(prov, "last_cap_reset_at", None)
        if reset is not None and reset <= time.time():
            reset = None
        return {"multi_account": False, "accounts": [], "next_reset_at_unix": reset}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()

        try:
            raw = await request.body()
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, TypeError) as e:
            return _openai_error_response(400, f"invalid JSON body: {e}", "invalid_request_error")

        requested_model = body.get("model", "sonnet")
        streaming = bool(body.get("stream"))
        try:
            ant_body = translate_openai_to_anthropic(body)
        except (ValueError, TypeError, KeyError) as e:
            return _openai_error_response(400, f"translation error: {e}", "invalid_request_error")

        url = f"{_upstream_base()}/v1/messages"

        if streaming:
            return await _handle_streaming(
                prov, url, ant_body, requested_model,
                max_stream_retries, max_inline_retries, max_inline_wait,
                app.state.http_stream,
            )
        return await _handle_nonstream(
            prov, url, ant_body, requested_model,
            max_inline_retries, max_inline_wait,
            app.state.http,
        )

    return app


async def _handle_nonstream(
    provider: ProviderLike,
    url: str,
    ant_body: dict[str, Any],
    requested_model: str,
    max_retries: int,
    max_wait: int,
    client: httpx.AsyncClient,
) -> Response:
    payload = json.dumps(ant_body).encode("utf-8")
    attempt = 0
    auth_retried = False
    tried_tokens: set[str] = set()

    while True:
        try:
            access_token = await _get_token(provider)
        except CredentialsError as e:
            return _openai_error_response(401, str(e), "authentication_error")

        if access_token in tried_tokens and attempt > 0:
            return _openai_error_response(503, "no usable credentials", "api_error")

        headers = _build_forward_headers(access_token, streaming=False)
        try:
            resp = await client.post(url, content=payload, headers=headers)
        except httpx.HTTPError as e:
            if attempt < max_retries:
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, max_wait))
                continue
            return _openai_error_response(502, f"upstream network error: {e}", "api_error")

        if 200 <= resp.status_code < 300:
            provider.last_cap_reset_at = None
            try:
                ant_json = resp.json()
            except ValueError as e:
                return _openai_error_response(502, f"invalid upstream JSON: {e}", "api_error")
            return JSONResponse(_anthropic_to_openai_nonstream(ant_json, requested_model))

        cls = classify_anthropic_error(resp.status_code, resp.content, resp.headers)
        await _apply_classification(provider, cls)
        _LOG.info(
            "upstream error: status=%d kind=%s retry_after=%s",
            resp.status_code, cls.kind, cls.retry_after_s,
        )

        if cls.kind == "auth" and not auth_retried:
            auth_retried = True
            tried_tokens.add(access_token)
            try:
                await asyncio.to_thread(provider.refresh)
            except PermanentCredentialsError as e:
                if isinstance(provider, MultiAccountCredentialProvider):
                    provider.mark_invalid()
                    continue
                return _openai_error_response(401, str(e), "authentication_error")
            except CredentialsError as e:
                return _openai_error_response(503, str(e), "api_error")
            continue

        if cls.kind == "cap" and isinstance(provider, MultiAccountCredentialProvider):
            tried_tokens.add(access_token)
            if provider.next_reset_at() is None:
                continue
            return _openai_error_from_classification(cls)

        if cls.kind == "transient" and attempt < max_retries:
            wait = cls.retry_after_s if cls.retry_after_s is not None else (2 ** (attempt + 1))
            wait = min(wait, max_wait)
            attempt += 1
            await asyncio.sleep(wait)
            continue

        return _openai_error_from_classification(cls)


_HeadState = Literal["started", "error", "closed"]


async def _peek_stream_head(
    byte_iter: AsyncIterator[bytes],
    max_bytes: int = 65536,
) -> tuple[bytes, _HeadState, Classification | None]:
    buf = b""
    text_buf = ""
    try:
        async for raw in byte_iter:
            buf += raw
            text_buf += raw.decode("utf-8", errors="replace")
            while True:
                idx = text_buf.find("\n\n")
                if idx < 0:
                    break
                block, text_buf = text_buf[:idx], text_buf[idx + 2:]
                block = block.strip("\r")
                if not block:
                    continue
                event_name, data = _parse_sse_event(block)
                if data is None:
                    continue
                t = data.get("type") or event_name
                if t == "message_start":
                    return buf, "started", None
                if t == "error":
                    return buf, "error", _classify_sse_error_event(data)
            if len(buf) > max_bytes:
                return buf, "started", None
    except httpx.HTTPError as e:
        _LOG.warning("upstream stream head read error: %s", e)
        return buf, "closed", None
    return buf, "closed", None


async def _handle_streaming(
    provider: ProviderLike,
    url: str,
    ant_body: dict[str, Any],
    requested_model: str,
    max_stream_retries: int,
    max_retries: int,
    max_wait: int,
    client: httpx.AsyncClient,
) -> Response:
    payload = json.dumps(ant_body).encode("utf-8")
    attempt = 0
    auth_retried = False
    tried_tokens: set[str] = set()
    buffer_retries = 0

    while True:
        try:
            access_token = await _get_token(provider)
        except CredentialsError as e:
            return _openai_error_response(401, str(e), "authentication_error")

        if access_token in tried_tokens and attempt > 0:
            return _openai_error_response(503, "no usable credentials", "api_error")

        headers = _build_forward_headers(access_token, streaming=True)

        stream_cm = client.stream("POST", url, content=payload, headers=headers)
        try:
            upstream = await stream_cm.__aenter__()
        except httpx.HTTPError as e:
            if attempt < max_retries:
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, max_wait))
                continue
            return _openai_error_response(502, f"upstream connect error: {e}", "api_error")

        if 200 <= upstream.status_code < 300:
            # Read the response body stream exactly once: httpx forbids calling
            # aiter_bytes() twice, so create one iterator and share it between the
            # head-peek and the full translation below.
            byte_iter = upstream.aiter_bytes()
            head_bytes, head_state, head_cls = await _peek_stream_head(byte_iter)

            if head_state == "error" and head_cls is not None:
                await stream_cm.__aexit__(None, None, None)
                await _apply_classification(provider, head_cls)
                _LOG.info(
                    "pre-commit stream error: kind=%s type=%s msg=%s",
                    head_cls.kind, head_cls.error_type, head_cls.message,
                )
                if head_cls.kind == "auth" and not auth_retried:
                    auth_retried = True
                    tried_tokens.add(access_token)
                    try:
                        await asyncio.to_thread(provider.refresh)
                    except PermanentCredentialsError as e:
                        if isinstance(provider, MultiAccountCredentialProvider):
                            provider.mark_invalid()
                            continue
                        return _openai_error_response(401, str(e), "authentication_error")
                    except CredentialsError as e:
                        return _openai_error_response(503, str(e), "api_error")
                    continue
                if head_cls.kind == "cap" and isinstance(provider, MultiAccountCredentialProvider):
                    tried_tokens.add(access_token)
                    if provider.next_reset_at() is None:
                        continue
                    return _openai_error_from_classification(head_cls)
                if head_cls.kind == "transient":
                    if buffer_retries < max_stream_retries and attempt < max_retries:
                        wait = head_cls.retry_after_s if head_cls.retry_after_s is not None else (2 ** (attempt + 1))
                        wait = min(wait, max_wait)
                        attempt += 1
                        buffer_retries += 1
                        await asyncio.sleep(wait)
                        continue
                return _openai_error_from_classification(head_cls)

            if head_state == "closed":
                await stream_cm.__aexit__(None, None, None)
                _LOG.warning("upstream closed before message_start; treating as transient")
                if isinstance(provider, MultiAccountCredentialProvider):
                    provider.mark_exhausted(60.0)
                    tried_tokens.add(access_token)
                if buffer_retries < max_stream_retries and attempt < max_retries:
                    attempt += 1
                    buffer_retries += 1
                    await asyncio.sleep(min(2 ** attempt, max_wait))
                    continue
                return _openai_error_response(502, "upstream closed before response", "api_error")

            provider.last_cap_reset_at = None
            committed_provider = provider
            committed_bytes = head_bytes

            async def event_stream() -> AsyncIterator[bytes]:
                try:
                    async for chunk in _translate_stream(
                        byte_iter, requested_model,
                        prebuffered=committed_bytes, provider=committed_provider,
                    ):
                        yield chunk
                finally:
                    await stream_cm.__aexit__(None, None, None)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-zoro-bridge": "streaming"},
            )

        body_bytes = b""
        try:
            async for chunk in upstream.aiter_bytes():
                body_bytes += chunk
                if len(body_bytes) > 65536:
                    break
        finally:
            await stream_cm.__aexit__(None, None, None)

        cls = classify_anthropic_error(upstream.status_code, body_bytes, upstream.headers)
        await _apply_classification(provider, cls)
        _LOG.info(
            "upstream stream error: status=%d kind=%s retry_after=%s",
            upstream.status_code, cls.kind, cls.retry_after_s,
        )

        if cls.kind == "auth" and not auth_retried:
            auth_retried = True
            tried_tokens.add(access_token)
            try:
                await asyncio.to_thread(provider.refresh)
            except PermanentCredentialsError as e:
                if isinstance(provider, MultiAccountCredentialProvider):
                    provider.mark_invalid()
                    continue
                return _openai_error_response(401, str(e), "authentication_error")
            except CredentialsError as e:
                return _openai_error_response(503, str(e), "api_error")
            continue

        if cls.kind == "cap" and isinstance(provider, MultiAccountCredentialProvider):
            tried_tokens.add(access_token)
            if provider.next_reset_at() is None:
                continue
            return _openai_error_from_classification(cls)

        if cls.kind == "transient":
            if buffer_retries < max_stream_retries and attempt < max_retries:
                wait = cls.retry_after_s if cls.retry_after_s is not None else (2 ** (attempt + 1))
                wait = min(wait, max_wait)
                attempt += 1
                buffer_retries += 1
                await asyncio.sleep(wait)
                continue

        return _openai_error_from_classification(cls)

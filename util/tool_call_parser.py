"""Shared parsing helpers for model-emitted tool calls."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_TOOL_CALL_XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_args(raw: Any) -> Dict[str, Any]:
    """Safely parse tool arguments coming from the model."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            import logging
            logging.getLogger(__name__).warning("parse_tool_args: JSON parse failed, dropping args. raw=%r", raw)
            return {}
    return {}


def _normalize_tool_call(candidate: Any) -> Optional[Dict[str, Any]]:
    """Normalize multiple tool-call shapes into {'name', 'arguments'}."""
    if not isinstance(candidate, dict):
        return None

    name = candidate.get("name")
    arguments = candidate.get("arguments")

    # Support OpenAI-style payloads:
    # {"type":"function","function":{"name":"...","arguments":"{...}"}}
    function_payload = candidate.get("function")
    if (not name) and isinstance(function_payload, dict):
        name = function_payload.get("name")
        arguments = function_payload.get("arguments")

    if not isinstance(name, str) or not name.strip():
        return None

    return {"name": name.strip(), "arguments": arguments if arguments is not None else {}}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def dedupe_end_today_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate repeated end_today actions while preserving order.

    Keep only the first end_today call for each equivalent arguments payload.
    """
    deduped: List[Dict[str, Any]] = []
    seen_end_today_signatures = set()

    for call in tool_calls:
        name = call.get("name")
        if name != "end_today":
            deduped.append(call)
            continue

        args_signature = _canonical_json(parse_tool_args(call.get("arguments")))
        if args_signature in seen_end_today_signatures:
            continue

        seen_end_today_signatures.add(args_signature)
        deduped.append(call)

    return deduped


def _parse_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed_calls: List[Dict[str, Any]] = []
    for json_str in _TOOL_CALL_XML_PATTERN.findall(text):
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        normalized = _normalize_tool_call(parsed)
        if normalized is not None:
            parsed_calls.append(normalized)
    return parsed_calls


def _parse_standalone_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    parsed_calls: List[Dict[str, Any]] = []
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx : i + 1]
                start_idx = -1
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                normalized = _normalize_tool_call(parsed)
                if normalized is not None:
                    parsed_calls.append(normalized)

    return parsed_calls


def parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse tool calls from text. Returns (tool_calls_list, parse_method_tag).

    Supported formats:
    1) XML blocks: <tool_call>{"name":"...", "arguments": {...}}</tool_call> (tag: "xml")
    2) Standalone JSON objects with tool-call shape (tag: "json")
    """
    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        return dedupe_end_today_calls(xml_calls), "xml"

    return [], "none"

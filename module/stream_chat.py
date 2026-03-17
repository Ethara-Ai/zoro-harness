import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


def stream_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 10.0,
    retry_jitter: float = 0.3,
) -> tuple[str, str, str, Optional[Dict[str, Any]]]:
    """
    Stream a chat completion with automatic retry.

    max_retries is the number of retries after the first attempt.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {408, 409, 429} or status_code >= 500:
                return True

        err_text = f"{type(exc).__name__}: {exc}".lower()
        retry_markers = (
            "timeout",
            "timed out",
            "rate limit",
            "connection",
            "network",
            "temporarily unavailable",
            "service unavailable",
            "try again",
            "overloaded",
            "stream closed",
        )
        return any(marker in err_text for marker in retry_markers)

    def _run_once() -> tuple[str, str, str, Optional[Dict[str, Any]]]:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            # stop=["\n<tool_response>", "<tool_response>"],
            extra_body={"enable_thinking": True},
            stream=True,
            stream_options={"include_usage": True},
            top_p=0.95,
            temperature=0.6,
            max_tokens=10000,
            presence_penalty=1.1,
            timeout=60000,
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
        full_content = reasoning_content + final_content
        return full_content, final_content, reasoning_content, usage

    last_error: Optional[Exception] = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return _run_once()
        except Exception as exc:
            last_error = exc
            should_retry = attempt < total_attempts and _is_retryable_error(exc)
            if not should_retry:
                raise

            delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
            if retry_jitter > 0:
                delay += random.uniform(0, retry_jitter)
            time.sleep(delay)

    raise RuntimeError(f"stream_chat failed after {total_attempts} attempts") from last_error

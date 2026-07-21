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
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from codex_bridge.credentials import (
    CredentialProvider,
    CredentialsError,
    MultiAccountCredentialProvider,
)
from codex_bridge.errors import (
    Classification,
    classify_openai_error,
    classify_stream_rate_limit,
    to_openai_error_body,
)

_LOG = logging.getLogger(__name__)

UPSTREAM_DEFAULT = "https://chatgpt.com/backend-api/wham"
RESPONSES_PATH = "/responses"

_VALID_REASONING_EFFORTS = {"low", "medium", "high"}
DEFAULT_MODEL = "gpt-5.6-sol"

CODEX_CLI_ORIGINATOR = "codex_cli_rs"
CODEX_CLI_VERSION = os.environ.get("ZORO_CX_CLI_VERSION", "").strip() or "0.145.0"
_BRIDGE_SESSION_ID = uuid.uuid4().hex

MODEL_ALIASES = {
    "codex": "gpt-5.6-sol",
    "codex-mini": "gpt-5.6-sol",
    "codex-cc": "gpt-5.6-sol",
    "gpt5-codex-cc": "gpt-5.6-sol",
}

REASONING_CAPABLE_PREFIXES = ("o1", "o3", "o4", "gpt-5", "codex-mini")


def _get_installation_id() -> str:
    from codex_bridge.credentials import _codex_home
    id_path = _codex_home() / "installation_id"
    try:
        if id_path.is_file():
            existing = id_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except OSError:
        pass
    new_id = str(uuid.uuid4())
    try:
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(new_id, encoding="utf-8")
    except OSError:
        pass
    return new_id


_BRIDGE_INSTALLATION_ID = _get_installation_id()

_OPENAI_CHAT_ONLY_KEYS = {
    "n", "logprobs", "top_logprobs", "logit_bias", "presence_penalty",
    "frequency_penalty", "seed", "response_format", "user", "functions",
    "function_call", "parallel_tool_calls", "service_tier",
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
    return os.environ.get("ZORO_CX_UPSTREAM", UPSTREAM_DEFAULT).rstrip("/")


def _default_reasoning_effort() -> str:
    v = (os.environ.get("ZORO_CX_DEFAULT_REASONING_EFFORT") or "medium").strip().lower()
    return v if v in _VALID_REASONING_EFFORTS else "medium"


def _timeout(streaming: bool = False) -> httpx.Timeout:
    connect = _env_float("ZORO_CX_CONNECT_TIMEOUT", 30.0)
    if streaming:
        read = _env_float("ZORO_CX_STREAM_READ_TIMEOUT", 600.0)
        return httpx.Timeout(None, connect=connect, read=read, write=None, pool=None)
    total = _env_float("ZORO_CX_REQUEST_TIMEOUT", 600.0)
    read = _env_float("ZORO_CX_READ_TIMEOUT", 180.0)
    return httpx.Timeout(total, connect=connect, read=read)


def _map_model(name: str) -> str:
    if not name:
        return DEFAULT_MODEL
    lower = name.lower()
    if lower in MODEL_ALIASES:
        return MODEL_ALIASES[lower]
    return name


def _model_supports_reasoning(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) for p in REASONING_CAPABLE_PREFIXES)


def _map_finish_reason(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status == "incomplete":
        details = response.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
        return "stop"
    if status in (None, "completed"):
        return "stop"
    if status == "failed":
        return "stop"
    return "stop"


def _string_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                if isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
        return "\n".join(parts)
    return ""


def _extract_instructions_and_input(
    openai_messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    instruction_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for m in openai_messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _string_content(m.get("content", ""))
        if role == "system":
            if text:
                instruction_parts.append(text)
            continue
        if role == "assistant":
            input_items.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })
            continue
        if role in ("user", "tool", "function"):
            input_items.append({
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
    return "\n\n".join(p for p in instruction_parts if p), input_items


def translate_openai_to_responses(body: dict[str, Any]) -> dict[str, Any]:
    model = _map_model(body.get("model", ""))
    instructions, input_items = _extract_instructions_and_input(body.get("messages") or [])

    out: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
    }
    if instructions:
        out["instructions"] = instructions

    if body.get("max_tokens") is not None:
        _LOG.warning(
            "dropping max_tokens=%r; wham /responses rejects max_output_tokens",
            body.get("max_tokens"),
        )

    stop = body.get("stop")
    if stop is not None:
        out["stop"] = [stop] if isinstance(stop, str) else list(stop)

    if body.get("stream"):
        out["stream"] = True

    extra = body.get("extra_body") or {}
    if isinstance(extra, dict):
        want_reasoning = bool(extra.get("enable_thinking"))
        effort_override = extra.get("reasoning_effort")
        if want_reasoning or effort_override:
            if not _model_supports_reasoning(model):
                _LOG.warning(
                    "reasoning requested for %s which does not support it; dropping",
                    model,
                )
            else:
                effort = (
                    str(effort_override).strip().lower()
                    if effort_override
                    else _default_reasoning_effort()
                )
                if effort not in _VALID_REASONING_EFFORTS:
                    _LOG.warning("invalid reasoning_effort %r; using medium", effort)
                    effort = "medium"
                out["reasoning"] = {"effort": effort}

    if "reasoning" not in out:
        if body.get("temperature") is not None:
            out["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            out["top_p"] = body["top_p"]

    if body.get("tools") or body.get("tool_choice"):
        raise HTTPException(
            status_code=422,
            detail="tools/tool_choice not supported by codex_bridge",
        )

    dropped = [k for k in _OPENAI_CHAT_ONLY_KEYS if k in body and body[k] is not None]
    if dropped:
        _LOG.warning(
            "dropping unsupported Chat-Completions-only params: %s",
            ",".join(sorted(dropped)),
        )

    return out


def _build_forward_headers(
    access_token: str,
    streaming: bool,
    account_id: str | None,
    is_fedramp: bool,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "text/event-stream" if streaming else "application/json",
        "originator": CODEX_CLI_ORIGINATOR,
        "User-Agent": f"{CODEX_CLI_ORIGINATOR}/{CODEX_CLI_VERSION}",
        "session_id": _BRIDGE_SESSION_ID,
        "x-codex-installation-id": _BRIDGE_INSTALLATION_ID,
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    if is_fedramp:
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def _extract_text_and_reasoning(response: dict[str, Any]) -> tuple[str, str]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "message":
            for blk in item.get("content") or []:
                if isinstance(blk, dict):
                    bt = blk.get("type")
                    if bt in ("output_text", "text") and isinstance(blk.get("text"), str):
                        text_parts.append(blk["text"])
        elif t == "reasoning":
            for blk in item.get("summary") or []:
                if isinstance(blk, dict) and blk.get("type") == "summary_text":
                    txt = blk.get("text")
                    if isinstance(txt, str):
                        reasoning_parts.append(txt)
    return "".join(text_parts), "".join(reasoning_parts)


def _responses_to_openai_nonstream(
    resp_body: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    response = resp_body.get("response") if "response" in resp_body else resp_body
    if not isinstance(response, dict):
        response = {}

    text, reasoning = _extract_text_and_reasoning(response)

    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
    }
    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _map_finish_reason(response),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
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
    err_type = None
    message = None
    if isinstance(err, dict):
        err_type = err.get("code") or err.get("type")
        message = err.get("message")
    message = message or err_type or "upstream stream error"
    if err_type in ("rate_limit_exceeded", "insufficient_quota"):
        return Classification("cap", 429, 300.0, time.time() + 300.0, err_type, message, None)
    if err_type in ("invalid_request_error",):
        return Classification("client", 400, None, None, err_type, message, None)
    if err_type in ("authentication_error", "permission_error"):
        return Classification("auth", 401, None, None, err_type, message, None)
    if err_type in ("server_error", "overloaded_error"):
        return Classification("transient", 503, None, None, err_type, message, None)
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
    sent_role = False
    finish_reason: str | None = None
    final_usage: dict[str, int] | None = None
    saw_completed = False

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
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{
                "index": 0,
                "delta": {"content": f"\n\n[bridge] upstream {cls.error_type or 'error'}: {cls.message}"},
                "finish_reason": None,
            }]
            yield _sse_chunk(payload)
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            yield _sse_chunk(payload)
            yield b"data: [DONE]\n\n"
            return

        if t == "response.created":
            if not sent_role:
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                yield _sse_chunk(payload)
                sent_role = True

        elif t == "response.output_text.delta":
            text = data.get("delta") or ""
            if text:
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                yield _sse_chunk(payload)

        elif t == "response.reasoning_summary_text.delta":
            text = data.get("delta") or ""
            if text:
                payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
                payload["choices"] = [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]
                yield _sse_chunk(payload)

        elif t in (
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ):
            continue

        elif t == "response.failed":
            resp = data.get("response") or {}
            err = resp.get("error") or {}
            err_type = None
            if isinstance(err, dict):
                err_type = err.get("code") or err.get("type")
            msg = (err or {}).get("message") if isinstance(err, dict) else None
            cls = Classification("transient", 502, None, None, err_type, msg or "response failed", None)
            if provider is not None:
                await _apply_classification(provider, cls)
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{
                "index": 0,
                "delta": {"content": f"\n\n[bridge] upstream failed: {cls.message}"},
                "finish_reason": None,
            }]
            yield _sse_chunk(payload)
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            yield _sse_chunk(payload)
            yield b"data: [DONE]\n\n"
            return

        elif t == "response.completed":
            saw_completed = True
            response = data.get("response") or {}
            finish_reason = _map_finish_reason(response)
            usage = response.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total = int(usage.get("total_tokens") or (input_tokens + output_tokens))
            final_usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total,
            }
            rl_cls = classify_stream_rate_limit(response.get("rate_limit_status"))
            if rl_cls is not None and provider is not None:
                _LOG.warning(
                    "in-stream rate_limit_status: type=%s retry_after=%.0fs",
                    rl_cls.error_type, rl_cls.retry_after_s or 0.0,
                )
                await _apply_classification(provider, rl_cls)

            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
            yield _sse_chunk(payload)
            payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
            payload["choices"] = []
            payload["usage"] = final_usage
            yield _sse_chunk(payload)
            yield b"data: [DONE]\n\n"
            return

    _LOG.error(
        "upstream SSE stream ended without response.completed (saw_completed=%s); emitting finish_reason=stop",
        saw_completed,
    )
    if provider is not None and not saw_completed:
        cls = Classification("transient", 502, 60.0, time.time() + 60.0, None, "stream truncated", None)
        await _apply_classification(provider, cls)
    if not sent_role:
        payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
        payload["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield _sse_chunk(payload)
    payload = _openai_chunk_skeleton(chunk_id, created, requested_model)
    payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    yield _sse_chunk(payload)
    yield b"data: [DONE]\n\n"


async def _get_token(provider: ProviderLike) -> str:
    return await asyncio.to_thread(provider.get_access_token)


async def _apply_classification(
    provider: ProviderLike,
    cls: Classification,
) -> None:
    if isinstance(provider, MultiAccountCredentialProvider):
        if cls.kind == "cap":
            provider.mark_exhausted(cls.retry_after_s or 300.0)
        elif cls.kind == "auth":
            provider.mark_invalid()
    if cls.kind == "cap":
        provider.last_cap_reset_at = cls.reset_at_unix or (
            time.time() + (cls.retry_after_s or 300.0)
        )


def _openai_error_response(status_code: int, message: str, type_: str | None = None) -> JSONResponse:
    body = to_openai_error_body(status_code, message, type_)
    return JSONResponse(body, status_code=status_code)


def _openai_error_from_classification(cls: Classification) -> JSONResponse:
    headers: dict[str, str] = {}
    if cls.retry_after_s is not None:
        headers["Retry-After"] = str(max(1, int(cls.retry_after_s)))
    body = to_openai_error_body(cls.status_code, cls.message, cls.error_type)
    return JSONResponse(body, status_code=cls.status_code, headers=headers)


def build_app(provider: ProviderLike | None = None) -> FastAPI:
    if provider is None:
        from codex_bridge.credentials import resolve_provider
        provider = resolve_provider()
    prov: ProviderLike = provider

    bridge_secret = os.environ.get("ZORO_CX_BRIDGE_SECRET", "").strip()
    if not bridge_secret:
        _LOG.warning(
            "ZORO_CX_BRIDGE_SECRET is not set — the bridge is UNAUTHENTICATED; "
            "any local process can spend this subscription."
        )

    max_inline_retries = _env_int("ZORO_CX_MAX_INLINE_RETRIES", 3)
    max_inline_wait = _env_int("ZORO_CX_MAX_INLINE_WAIT", 30)
    max_stream_retries = _env_int("ZORO_CX_STREAM_BUFFER_RETRIES", 3)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=_timeout())
        app.state.http_stream = httpx.AsyncClient(timeout=_timeout(streaming=True))
        try:
            yield
        finally:
            await app.state.http.aclose()
            await app.state.http_stream.aclose()

    app = FastAPI(title="Zoro Codex Bridge", version="0.1.0", lifespan=lifespan)

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
        plan = None
        account = None
        try:
            tp = prov.token_prefix()
        except Exception:  # noqa: BLE001
            tp = None
        try:
            plan = prov.plan_type()
        except Exception:  # noqa: BLE001
            plan = None
        try:
            account = prov.account_id()
        except Exception:  # noqa: BLE001
            account = None
        return {"ok": True, "token_prefix": tp, "plan": plan, "account_id": account}

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

        requested_model = body.get("model", DEFAULT_MODEL)
        streaming = bool(body.get("stream"))
        try:
            responses_body = translate_openai_to_responses(body)
        except (ValueError, TypeError, KeyError) as e:
            return _openai_error_response(400, f"translation error: {e}", "invalid_request_error")

        url = f"{_upstream_base()}{RESPONSES_PATH}"

        if streaming:
            return await _handle_streaming(
                prov, url, responses_body, requested_model,
                max_stream_retries, max_inline_retries, max_inline_wait,
                app.state.http_stream,
            )
        return await _handle_nonstream(
            prov, url, responses_body, requested_model,
            max_inline_retries, max_inline_wait,
            app.state.http_stream,
        )

    return app


async def _handle_nonstream(
    provider: ProviderLike,
    url: str,
    responses_body: dict[str, Any],
    requested_model: str,
    max_retries: int,
    max_wait: int,
    client: httpx.AsyncClient,
) -> Response:
    responses_body = {**responses_body, "stream": True}
    payload = json.dumps(responses_body).encode("utf-8")
    attempt = 0
    auth_retried = False
    tried_tokens: set[str] = set()

    while True:
        try:
            access_token = await _get_token(provider)
        except CredentialsError as e:
            return _openai_error_response(401, str(e), "authentication_error")

        if access_token in tried_tokens and attempt > 0:
            return _openai_error_response(503, "no usable credentials", "server_error")

        headers = _build_forward_headers(
            access_token,
            streaming=True,
            account_id=provider.account_id(),
            is_fedramp=provider.is_fedramp(),
        )

        stream_cm = client.stream("POST", url, content=payload, headers=headers)
        try:
            upstream = await stream_cm.__aenter__()
        except httpx.HTTPError as e:
            if attempt < max_retries:
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, max_wait))
                continue
            return _openai_error_response(502, f"upstream network error: {e}", "server_error")

        final_response: dict[str, Any] | None = None
        cls: Classification | None = None
        try:
            if 200 <= upstream.status_code < 300:
                async for block in _iter_sse_blocks(upstream.aiter_bytes()):
                    event_name, data = _parse_sse_event(block)
                    if data is None:
                        continue
                    t = data.get("type") or event_name
                    if t == "response.completed":
                        final_response = data.get("response") or {}
                        break
                    if t == "error":
                        cls = _classify_sse_error_event(data)
                        break
                    if t == "response.failed":
                        resp = data.get("response") or {}
                        err = resp.get("error") if isinstance(resp, dict) else None
                        err_type = None
                        msg = "response failed"
                        if isinstance(err, dict):
                            err_type = err.get("code") or err.get("type")
                            msg = err.get("message") or msg
                        cls = Classification("transient", 502, None, None, err_type, msg, None)
                        break
                if final_response is None and cls is None:
                    cls = Classification(
                        "transient", 502, None, None, None,
                        "upstream stream ended without response.completed", None,
                    )
            else:
                body_bytes = b""
                async for chunk in upstream.aiter_bytes():
                    body_bytes += chunk
                    if len(body_bytes) > 65536:
                        break
                cls = classify_openai_error(upstream.status_code, body_bytes, upstream.headers)
                _LOG.info(
                    "upstream error: status=%d kind=%s retry_after=%s",
                    upstream.status_code, cls.kind, cls.retry_after_s,
                )
        finally:
            await stream_cm.__aexit__(None, None, None)

        if final_response is not None:
            provider.last_cap_reset_at = None
            return JSONResponse(_responses_to_openai_nonstream(final_response, requested_model))

        assert cls is not None
        await _apply_classification(provider, cls)

        if cls.kind == "auth" and not auth_retried:
            auth_retried = True
            tried_tokens.add(access_token)
            try:
                if not isinstance(provider, MultiAccountCredentialProvider):
                    await asyncio.to_thread(provider.refresh)
            except CredentialsError as e:
                return _openai_error_response(401, str(e), "authentication_error")
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
                if t == "response.created":
                    return buf, "started", None
                if t == "error":
                    return buf, "error", _classify_sse_error_event(data)
                if t == "response.failed":
                    resp = data.get("response") or {}
                    err = resp.get("error") or {}
                    err_type = None
                    if isinstance(err, dict):
                        err_type = err.get("code") or err.get("type")
                    msg = (err or {}).get("message") if isinstance(err, dict) else "response failed"
                    return buf, "error", Classification(
                        "transient", 502, None, None, err_type, msg or "response failed", None,
                    )
            if len(buf) > max_bytes:
                return buf, "started", None
    except httpx.HTTPError as e:
        _LOG.warning("upstream stream head read error: %s", e)
        return buf, "closed", None
    return buf, "closed", None


async def _handle_streaming(
    provider: ProviderLike,
    url: str,
    responses_body: dict[str, Any],
    requested_model: str,
    max_stream_retries: int,
    max_retries: int,
    max_wait: int,
    client: httpx.AsyncClient,
) -> Response:
    payload = json.dumps(responses_body).encode("utf-8")
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
            return _openai_error_response(503, "no usable credentials", "server_error")

        headers = _build_forward_headers(
            access_token,
            streaming=True,
            account_id=provider.account_id(),
            is_fedramp=provider.is_fedramp(),
        )

        stream_cm = client.stream("POST", url, content=payload, headers=headers)
        try:
            upstream = await stream_cm.__aenter__()
        except httpx.HTTPError as e:
            if attempt < max_retries:
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, max_wait))
                continue
            return _openai_error_response(502, f"upstream connect error: {e}", "server_error")

        if 200 <= upstream.status_code < 300:
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
                        if not isinstance(provider, MultiAccountCredentialProvider):
                            await asyncio.to_thread(provider.refresh)
                    except CredentialsError as e:
                        return _openai_error_response(401, str(e), "authentication_error")
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
                _LOG.warning("upstream closed before response.created; treating as transient")
                if isinstance(provider, MultiAccountCredentialProvider):
                    provider.mark_exhausted(60.0)
                    tried_tokens.add(access_token)
                if buffer_retries < max_stream_retries and attempt < max_retries:
                    attempt += 1
                    buffer_retries += 1
                    await asyncio.sleep(min(2 ** attempt, max_wait))
                    continue
                return _openai_error_response(502, "upstream closed before response", "server_error")

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

        cls = classify_openai_error(upstream.status_code, body_bytes, upstream.headers)
        await _apply_classification(provider, cls)
        _LOG.info(
            "upstream stream error: status=%d kind=%s retry_after=%s",
            upstream.status_code, cls.kind, cls.retry_after_s,
        )

        if cls.kind == "auth" and not auth_retried:
            auth_retried = True
            tried_tokens.add(access_token)
            try:
                if not isinstance(provider, MultiAccountCredentialProvider):
                    await asyncio.to_thread(provider.refresh)
            except CredentialsError as e:
                return _openai_error_response(401, str(e), "authentication_error")
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

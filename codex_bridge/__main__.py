from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m codex_bridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8398)
    p.add_argument("--log-level", default="info")
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify credentials, then exit.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from codex_bridge.credentials import CredentialsError, resolve_provider
    from codex_bridge.server import build_app

    provider = resolve_provider()
    try:
        provider.preflight()
    except CredentialsError as e:
        print(f"[bridge] credentials error: {e}", file=sys.stderr)
        return 2

    tp = provider.token_prefix() or "?"
    plan = provider.plan_type() or "?"
    account = provider.account_id() or "?"
    print(f"[bridge] credentials OK (token prefix: {tp}... plan={plan} account={account})")
    if args.check:
        return 0

    import uvicorn

    print(f"[bridge] listening on http://{args.host}:{args.port}")
    print("[bridge] point clients at:")
    print(f"           export ZORO_LLM_BASE_URL=http://{args.host}:{args.port}/v1")
    print("           export ZORO_LLM_API_KEY=${ZORO_CX_BRIDGE_SECRET:-zoro-cx-stub}")
    uvicorn.run(
        build_app(provider),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

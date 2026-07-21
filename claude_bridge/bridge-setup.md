# Claude Code Bridge — Setup Guide

The bridge is a small local server that looks like OpenAI but actually talks to
Claude using your Claude Code subscription. You point `run_env.py` at it
instead of a real API.

This guide covers **the bridge itself** — how to start it, how it finds
credentials, how to tune it, and how to debug it. For the credentials
workflow (which files go where, how to generate them, how the two-bridge
setup avoids intermixing) see [`../credential-setup.md`](../credential-setup.md).

---

## 1. Prerequisites

- Credentials are already in place in `creds/cc/` (see
  [`../credential-setup.md`](../credential-setup.md) sections 2–4).
- You run everything from the `zoro-harness` folder with the venv active:
  ```
  cd /Users/apple/Sources/office/zoro-main/zoro-harness
  source .venv/bin/activate
  ```

---

## 2. The two "keys" (don't mix them up)

| Thing | What it is | Where it lives |
|---|---|---|
| **Bridge secret** | A password *you make up*. Both the bridge and `run_env` must use the same one. | `ZORO_CC_BRIDGE_SECRET` env var |
| **Credentials** | Your real Claude login (an OAuth token blob — **not** email/password) | JSON files under `creds/cc/` |

Full creds setup is in [`../credential-setup.md`](../credential-setup.md).
The bridge's own knobs are below.

---

## 3. Start the bridge (terminal #1)

Set the bridge secret **before** launching. This is the password you picked
in the credential-setup step.

```
export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CC_WRITE_BACK_KEYCHAIN=1
./script/start_claude_bridge.sh start
```

- `ZORO_CC_BRIDGE_SECRET` → the made-up password.
- `ZORO_CC_WRITE_BACK_KEYCHAIN=1` → keeps refreshed tokens in sync so your
  `claude` CLI doesn't get logged out.

**It worked if** the log says
`Claude bridge: auto-discovered pool from …/creds/cc (N slots): creds1.json, …`.

Stop it later with:
```
./script/start_claude_bridge.sh stop
```

There is no `restart` subcommand — `stop` then `start`.

### Health check

```
curl -s http://127.0.0.1:8738/healthz
```

### Logs

- `logs/claude_bridge.log` — the bridge process itself
- `logs/claude_bridge_monitor.log` — the supervisor that restarts it if it dies

PID files (safe to ignore; git-ignored):
- `.claude_bridge.pid`
- `.claude_bridge_monitor.pid`

---

## 4. Bridge-specific tuning

### Custom creds directory
```
export ZORO_CC_CREDS_DIR=/some/other/path
```
The bridge will auto-discover `*.json` files there instead of `creds/cc/`.

### Explicit pool list (wins over auto-discovery)
```
export ZORO_CC_ACCOUNT_POOL="/some/other/path/a.json:/some/other/path/b.json"
```

### Custom port
```
export ZORO_CC_BRIDGE_PORT=8888
```
If you change this, `run_env.py`'s auto-routing still assumes `8738`, so
you'll also need to pass `--base_url http://127.0.0.1:8888/v1` explicitly.

---

## 5. Point `run_env.py` at it (terminal #2)

Open a **second** terminal (the bridge keeps running in the first).

```
cd /Users/apple/Sources/office/zoro-main/zoro-harness
source .venv/bin/activate

export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"   # same as step 3

python run_env.py \
  --model sonnet \
  --max_days 7 \
  --max_strategy_turns 2 \
  --max_execution_turns 3 \
  --max_input_tokens 20000
```

- No `--base_url` or `--api_key` needed. `--model opus`, `sonnet`, `haiku`,
  or any `claude-…` name auto-routes to the Claude bridge and reads
  `ZORO_CC_BRIDGE_SECRET` from the environment.
- The `--max_*` flags keep this a small, cheap test run.

**It worked if** you see real `Token usage: prompt=… completion=…` numbers each
day (not `stream_chat failed`).

For the full routing table and how bridges avoid intermixing, see
[`../credential-setup.md`](../credential-setup.md) section 6.

---

## 6. Bridge-specific troubleshooting

Credential-file mix-ups and general "did I export the secret" issues are
covered in [`../credential-setup.md`](../credential-setup.md) section 7.
The table below is for bridge-runtime issues.

| Message | Meaning | Fix |
|---|---|---|
| `no usable accounts in pool` | Every `creds/cc/*.json` is malformed or empty | Re-run the credential-setup step for at least one account |
| `all N accounts exhausted` | Every Claude account is rate-limited | Wait, or drop another `creds/cc/creds*.json` in and restart the bridge |
| `Address already in use` on start | Something else already holds port `8738` (maybe an old bridge) | `./script/start_claude_bridge.sh stop`, then `start` |
| `ModuleNotFoundError` on start | venv isn't active | `source .venv/bin/activate` |
| Bridge dies silently | Check `logs/claude_bridge.log`; the monitor should relaunch it | Look at monitor log too: `logs/claude_bridge_monitor.log` |

---

## Cheat sheet

```
# terminal 1 — bridge
export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CC_WRITE_BACK_KEYCHAIN=1
./script/start_claude_bridge.sh start

# terminal 2 — run
export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"
python run_env.py --model sonnet \
  --max_days 7 --max_strategy_turns 2 --max_execution_turns 3
```

For credentials, see [`../credential-setup.md`](../credential-setup.md).

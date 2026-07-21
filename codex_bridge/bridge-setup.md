# Codex Bridge — Setup Guide

The bridge is a small local server that looks like OpenAI but actually talks to
GPT-5 / Codex using your **ChatGPT subscription** (the same one the `codex` CLI
uses). You point `run_env.py` at it instead of a real OpenAI API key.

This guide focuses on the **multi-account** setup (a "pool"), so the bridge can
rotate between several ChatGPT logins and keep going when one hits its weekly
usage cap.

---

## 1. What you need

- You are logged into at least one ChatGPT account via the `codex` CLI, using
  the **"Sign in with ChatGPT"** option (not "Use an API key" — the bridge
  refuses API-key mode on purpose).
- You run everything from the `zoro-harness` folder with the venv active:
  ```
  cd /Users/apple/Sources/office/zoro-main/zoro-harness
  source .venv/bin/activate
  ```

---

## 2. Understand the two "keys" (don't mix them up)

| Thing | What it is | Where it lives |
|---|---|---|
| **Bridge secret** | A password *you make up*. Both the bridge and `run_env` must use the same one. | An env var you set |
| **Credentials** | Your real ChatGPT login (an OAuth token blob — **not** email/password) | A JSON file per account |

---

## 3. Create a credentials file per account

Each account is **one JSON file** holding that account's OAuth tokens. You never
type these by hand — the `codex` CLI writes them to `~/.codex/auth.json` when
you sign in. You just copy that file.

> `creds*.json` files are listed in `.gitignore`, so placing them in the
> repo root is safe — they will never be accidentally committed.

**Account #1** (your current login):
```
cp ~/.codex/auth.json creds1.json
```

**Account #2, #3, …**: sign into each other ChatGPT account with the `codex`
CLI (on this or another Mac), then copy that machine's `~/.codex/auth.json`
over. Name them `creds2.json`, `creds3.json`, etc.

Each file must look like this (this is what `codex login` already produces):
```json
{
  "auth_mode": "chatgpt",
  "tokens": {
    "id_token": "eyJhbG…",
    "access_token": "eyJhbG…",
    "refresh_token": "…",
    "account_id": "…"
  },
  "last_refresh": "2026-07-20T00:00:00Z"
}
```

> If `auth_mode` is anything other than `"chatgpt"` (e.g. `"api_key"`), the
> bridge will refuse to start. Re-run `codex logout && codex login` and pick
> **"Sign in with ChatGPT"**.

> One account = no cap-rotation benefit. The pool only helps if you have 2+
> real ChatGPT accounts.

---

## 4. Start the bridge (terminal #1)

Set the variables **before** launching — the bridge reads them once at startup.
List every credentials file in `ZORO_CX_ACCOUNT_POOL`, separated by colons `:`.

```
export ZORO_CX_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CX_ACCOUNT_POOL="/Users/apple/Sources/office/zoro-main/zoro-harness/creds1.json:/Users/apple/Sources/office/zoro-main/zoro-harness/creds2.json"
export ZORO_CX_WRITE_BACK_KEYCHAIN=1
python -m codex_bridge --host 127.0.0.1 --port 8398
```

- `ZORO_CX_BRIDGE_SECRET` → the made-up password.
- `ZORO_CX_ACCOUNT_POOL` → full paths to your creds files, joined with `:`.
- `ZORO_CX_WRITE_BACK_KEYCHAIN=1` → keeps refreshed tokens in sync so your
  `codex` CLI doesn't get logged out.

**It worked if** the log says `Using multi-account pool with N slots`.

> Use file paths in the pool (like above). Don't use `keychain:...` entries
> there — the colon separator breaks that form.

### Optional: pick a reasoning effort

The bridge sets GPT-5's thinking budget to `medium` by default. To change it
globally:

```
export ZORO_CX_DEFAULT_REASONING_EFFORT=high   # or low / medium
```

A single call can also override it per-request via `extra_body.reasoning_effort`
— nothing to configure here.

---

## 5. Run run_env against the bridge (terminal #2)

Open a **second** terminal (the bridge keeps running in the first).

```
cd /Users/apple/Sources/office/zoro-main/zoro-harness
source .venv/bin/activate

python run_env.py \
  --model gpt5-codex-cc \
  --base_url http://127.0.0.1:8398/v1 \
  --api_key pick-any-random-string \
  --max_days 7 \
  --max_strategy_turns 2 \
  --max_execution_turns 3 \
  --max_input_tokens 20000
```

- `--base_url` → the bridge (keep the `/v1`).
- `--api_key` → the **same** string as `ZORO_CX_BRIDGE_SECRET`.
- `--model` → a Codex-bridge model spec in the harness (e.g. `codex-cc`,
  `gpt5-codex-cc`). If your `run_env.py` doesn't recognize these yet, wiring
  those model specs is a separate task — see the note below.
- The `--max_*` flags keep this a small, cheap test run.

**It worked if** you see real `Token usage: prompt=… completion=…` numbers each
day (not `stream_chat failed`).

### Smoke test without run_env (any terminal)

If you just want to confirm the bridge itself works before touching
`run_env.py`:

```
curl -s http://127.0.0.1:8398/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer $ZORO_CX_BRIDGE_SECRET" \
  -d '{
    "model": "gpt-5.6-sol",
    "messages": [{"role":"user","content":"say hi in one word"}]
  }' | python3 -m json.tool
```

You should get back an OpenAI-shaped `chat.completion` JSON with `choices[0].message.content` set.

Available models on a Pro plan (as of July 2026):
`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`, `codex-auto-review`

The bridge default is `gpt-5.6-sol`. The short aliases `codex-cc` and `gpt5-codex-cc` both resolve to `gpt-5.6-sol`.

---

## 6. If something breaks

| Message | Meaning | Fix |
|---|---|---|
| `401 missing or invalid bridge secret` | `--api_key` ≠ `ZORO_CX_BRIDGE_SECRET` | Make them match |
| `Codex auth_mode is 'api_key'; this bridge only supports 'chatgpt'` | You signed in with an API key instead of ChatGPT | `codex logout && codex login`, pick "Sign in with ChatGPT" |
| `No Codex credentials found. Sign in via the codex CLI first` | The bridge can't find `~/.codex/auth.json` (or keychain) | Run `codex login` on this machine, or copy `auth.json` from a machine that has it |
| `no usable accounts in pool` | A creds file is missing/empty/bad | Re-run `cp ~/.codex/auth.json credsN.json` on the machine where that account is signed in |
| `all N accounts exhausted; soonest reset in Xs` | Every account hit its ChatGPT weekly cap | Wait `X` seconds, or add another account file |
| `OAuth refresh permanently failed (refresh_token_expired)` | That account's refresh token is dead | Re-run `codex login` for that account and refresh its creds file |

---

## Cheat sheet

```
# terminal 1 — bridge
export ZORO_CX_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CX_ACCOUNT_POOL="$PWD/creds1.json:$PWD/creds2.json"
export ZORO_CX_WRITE_BACK_KEYCHAIN=1
python -m codex_bridge --host 127.0.0.1 --port 8398

# terminal 2 — run
python run_env.py --model gpt5-codex-cc \
  --base_url http://127.0.0.1:8398/v1 --api_key pick-any-random-string \
  --max_days 7 --max_strategy_turns 2 --max_execution_turns 3
```

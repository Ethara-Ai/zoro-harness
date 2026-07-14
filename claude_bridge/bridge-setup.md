# Claude Code Bridge — Setup Guide

The bridge is a small local server that looks like OpenAI but actually talks to
Claude using your Claude Code subscription. You point `run_env.py` at it instead
of a real API.

This guide focuses on the **multi-account** setup (a "pool"), so the bridge can
rotate between several Claude logins and keep going when one hits its rate limit.

---

## 1. What you need

- You are logged into at least one Claude account via the `claude` CLI.
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
| **Credentials** | Your real Claude login (an OAuth token blob — **not** email/password) | A JSON file per account |

---

## 3. Create a credentials file per account

Each account is **one JSON file** holding that account's OAuth tokens. You never
type these by hand — you copy them out of the Mac Keychain.

**Account #1** (your current login):
```
security find-generic-password -s "Claude Code-credentials" -w > creds1.json
```

**Account #2, #3, …**: log into each other Claude account with the `claude` CLI
(on this or another Mac), then run the same command there and copy the file over.
Name them `creds2.json`, `creds3.json`, etc.

Each file must look like this (this is what the `security` command already gives you):
```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-xxxxx",
    "refreshToken": "sk-ant-ort01-xxxxx",
    "expiresAt": 1760000000000,
    "scopes": ["user:inference", "user:profile"],
    "subscriptionType": "max"
  }
}
```

> One account = no rate-limit benefit. The pool only helps if you have 2+ real
> Claude accounts.

---

## 4. Start the bridge (terminal #1)

Set the variables **before** launching — the bridge reads them once at startup.
List every credentials file in `ZORO_CC_ACCOUNT_POOL`, separated by colons `:`.

```
export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CC_ACCOUNT_POOL="/Users/apple/Sources/office/zoro-main/zoro-harness/creds1.json:/Users/apple/Sources/office/zoro-main/zoro-harness/creds2.json"
export ZORO_CC_WRITE_BACK_KEYCHAIN=1
python -m claude_bridge --host 127.0.0.1 --port 8738
```

- `ZORO_CC_BRIDGE_SECRET` → the made-up password.
- `ZORO_CC_ACCOUNT_POOL` → full paths to your creds files, joined with `:`.
- `ZORO_CC_WRITE_BACK_KEYCHAIN=1` → keeps refreshed tokens in sync so your
  `claude` CLI doesn't get logged out.

**It worked if** the log says `Using multi-account pool with N slots`.

> Use file paths in the pool (like above). Don't use `keychain:...` entries there
> — the colon separator breaks that form.

---

## 5. Run run_env against the bridge (terminal #2)

Open a **second** terminal (the bridge keeps running in the first).

```
cd /Users/apple/Sources/office/zoro-main/zoro-harness
source .venv/bin/activate

python run_env.py \
  --model sonnet \
  --base_url http://127.0.0.1:8738/v1 \
  --api_key pick-any-random-string \
  --max_days 7 \
  --max_strategy_turns 2 \
  --max_execution_turns 3 \
  --max_input_tokens 20000
```

- `--base_url` → the bridge (keep the `/v1`).
- `--api_key` → the **same** string as `ZORO_CC_BRIDGE_SECRET`.
- `--model` → only `sonnet`, `opus`, or `haiku` are understood.
- The `--max_*` flags keep this a small, cheap test run.

**It worked if** you see real `Token usage: prompt=… completion=…` numbers each
day (not `stream_chat failed`).

---

## 6. If something breaks

| Message | Meaning | Fix |
|---|---|---|
| `401 invalid bridge secret` | `--api_key` ≠ `ZORO_CC_BRIDGE_SECRET` | Make them match |
| `no usable accounts in pool` | a creds file is missing/empty/bad | Re-run the `security … > credsN.json` step |
| `all N accounts exhausted` | every account is rate-limited | Wait, or add another account file |
| `ModuleNotFoundError: sku` | unrelated import bug | Already fixed in the repo |

---

## Cheat sheet

```
# terminal 1 — bridge
export ZORO_CC_BRIDGE_SECRET="pick-any-random-string"
export ZORO_CC_ACCOUNT_POOL="$PWD/creds1.json:$PWD/creds2.json"
export ZORO_CC_WRITE_BACK_KEYCHAIN=1
python -m claude_bridge --host 127.0.0.1 --port 8738

# terminal 2 — run
python run_env.py --model sonnet \
  --base_url http://127.0.0.1:8738/v1 --api_key pick-any-random-string \
  --max_days 7 --max_strategy_turns 2 --max_execution_turns 3
```

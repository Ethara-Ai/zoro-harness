# Bridge credentials

OAuth tokens for the two local proxies (`claude_bridge` and `codex_bridge`)
that let this harness use your Claude Code / ChatGPT subscriptions instead of
paid API keys.

## Layout (strict bifurcation)

```
creds/
  cc/    Claude Code credentials     (from `security find-generic-password` or `~/.claude/.credentials.json`)
  cx/    Codex / ChatGPT credentials (from `~/.codex/auth.json`)
```

**Rules:**

1. `.json` files under `cc/` MUST be Claude Code format (top-level
   `claudeAiOauth` field). If you drop a Codex file here the Claude bridge
   will refuse to start with a loud, named error.
2. `.json` files under `cx/` MUST be Codex format (top-level `auth_mode:
   chatgpt` and `tokens` object). If you drop a Claude file here the Codex
   bridge will refuse to start with a loud, named error.
3. Files here are ignored by git (see repo `.gitignore`), so it is safe to
   store OAuth tokens under `creds/` — they will never be accidentally
   committed. This README is the only tracked content.

## How the bridges find these files

Each bridge auto-discovers `.json` files in its dedicated directory at
startup, in this order of priority:

1. `ZORO_CC_ACCOUNT_POOL` / `ZORO_CX_ACCOUNT_POOL` env var (colon-separated
   file paths) — explicit override, wins if set.
2. Auto-discovered `creds/cc/*.json` (Claude) or `creds/cx/*.json` (Codex).
3. Fallback to single-account provider (env var / `~/.codex/auth.json` /
   `~/.claude/.credentials.json` / macOS Keychain / cache).

For the normal case, just drop the files under the right subdirectory and
start the bridge — no env vars needed.

## How run_env picks the right bridge

`run_env.py` routes on the `--model` name:

| Model prefix                                             | Bridge         | Reads secret env var       |
|----------------------------------------------------------|----------------|----------------------------|
| `opus` / `sonnet` / `haiku` / `claude`                   | claude_bridge  | `ZORO_CC_BRIDGE_SECRET`    |
| `sol` / `terra` / `luna` / `gpt-5` / `gpt5` / `codex`    | codex_bridge   | `ZORO_CX_BRIDGE_SECRET`    |

So `python run_env.py --model opus` only ever touches Claude creds and
`python run_env.py --model sol` only ever touches Codex creds. Explicit
`--base_url` + `--api_key` bypass routing entirely.

See `cc/README.md` and `cx/README.md` for per-bridge setup.

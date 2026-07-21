# Codex / ChatGPT credentials — drop files here

The Codex bridge (`script/start_codex_bridge.sh`) auto-picks every `*.json`
file in this directory as one account slot at startup.

## Adding an account

Each account is one JSON file holding that account's OAuth tokens. The
`codex` CLI writes them to `~/.codex/auth.json` when you sign in with the
**"Sign in with ChatGPT"** option (NOT "Use an API key" — the bridge
refuses API-key mode on purpose).

```bash
cp ~/.codex/auth.json creds/cx/creds1.json
```

For more accounts: sign into a different ChatGPT account with the `codex`
CLI (on this or another Mac), copy that machine's `~/.codex/auth.json`, save
as `creds2.json`, `creds3.json`, etc. Filenames don't matter beyond the
`.json` extension — the bridge sorts them alphabetically and rotates
round-robin.

## File shape (do not hand-edit)

```json
{
  "auth_mode": "chatgpt",
  "tokens": {
    "id_token": "eyJhbG...",
    "access_token": "eyJhbG...",
    "refresh_token": "...",
    "account_id": "..."
  },
  "last_refresh": "2026-07-20T00:00:00Z"
}
```

If `auth_mode` is anything other than `"chatgpt"` (e.g. `"api_key"`), the
bridge will refuse to start. Re-run `codex logout && codex login` and pick
**"Sign in with ChatGPT"**.

If you drop a **Claude Code** file here by mistake, the Codex bridge will
refuse to start and name the offending file. No silent misrouting.

## Verify

```bash
./script/start_codex_bridge.sh start
# expect a log line like:
#   Codex bridge: auto-discovered pool from .../creds/cx (2 slots): creds1.json, creds2.json
```

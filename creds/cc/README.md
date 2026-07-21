# Claude Code credentials — drop files here

The Claude bridge (`script/start_claude_bridge.sh`) auto-picks every
`*.json` file in this directory as one account slot at startup.

## Adding an account

Each account is one JSON file holding that account's OAuth tokens, exported
from macOS Keychain after you've signed in with the `claude` CLI:

```bash
# from any machine with `claude` logged in:
security find-generic-password -s "Claude Code-credentials" -w > /tmp/cc.json

# then place it here:
mv /tmp/cc.json creds/cc/creds1.json
```

For more accounts: sign in with a different Claude account (on this or
another Mac), export again, save as `creds2.json`, `creds3.json`, etc.
Filenames don't matter beyond the `.json` extension — the bridge sorts them
alphabetically and rotates round-robin.

## File shape (do not hand-edit)

```json
{
  "claudeAiOauth": {
    "accessToken": "sk-ant-oat01-...",
    "refreshToken": "sk-ant-ort01-...",
    "expiresAt": 1760000000000,
    "scopes": ["user:inference", "user:profile"],
    "subscriptionType": "max"
  }
}
```

If you drop a **Codex** file here by mistake, the Claude bridge will refuse
to start and name the offending file. No silent misrouting.

## Verify

```bash
./script/start_claude_bridge.sh start
# expect a log line like:
#   Claude bridge: auto-discovered pool from .../creds/cc (2 slots): creds1.json, creds2.json
```

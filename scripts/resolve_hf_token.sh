#!/usr/bin/env bash
# resolve_hf_token.sh — SOURCE this (don't execute) to export HF_TOKEN.
#
#   source scripts/resolve_hf_token.sh
#
# Resolution order (first hit wins):
#   1. an already-set $HF_TOKEN
#   2. $LEDUC_HF_TOKEN            (injected from .claude/settings.local.json "env")
#   3. .claude/settings.local.json  → .env.LEDUC_HF_TOKEN / .env.HF_TOKEN
#   4. repo .env                  → HF_TOKEN=...
#
# Both .claude/ and .env are gitignored, so the token is never hard-coded in a
# tracked script and never committed. On a box reached only by `git pull`, put the
# token there once (any of: keep .claude/settings.local.json, `export LEDUC_HF_TOKEN`,
# or `echo HF_TOKEN=hf_... >> .env`) and this resolver picks it up.
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${HF_TOKEN:-}" ] && [ -n "${LEDUC_HF_TOKEN:-}" ]; then
  HF_TOKEN="$LEDUC_HF_TOKEN"
fi

_settings="$_REPO/.claude/settings.local.json"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$_settings" ]; then
  if command -v jq >/dev/null 2>&1; then
    HF_TOKEN="$(jq -r '.env.LEDUC_HF_TOKEN // .env.HF_TOKEN // empty' "$_settings" 2>/dev/null)"
  elif command -v python3 >/dev/null 2>&1; then
    HF_TOKEN="$(python3 -c 'import json,sys; e=json.load(open(sys.argv[1])).get("env",{}); print(e.get("LEDUC_HF_TOKEN") or e.get("HF_TOKEN") or "")' "$_settings" 2>/dev/null)"
  fi
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "$_REPO/.env" ]; then
  HF_TOKEN="$(grep -E '^[[:space:]]*HF_TOKEN=' "$_REPO/.env" | tail -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  echo "HF_TOKEN resolved (${HF_TOKEN:0:6}…, ${#HF_TOKEN} chars)"
else
  echo "WARNING: no HF_TOKEN found (checked env, .claude/settings.local.json, .env) — HF downloads run unauthenticated (rate-limited; big FP8 shards can stall)." >&2
fi

# This pod hangs forever in huggingface_hub's Xet transfer backend — the download
# threads block in file_download.py:xet_get (the Xet CAS endpoint is unreachable
# here). That is THE cause of the "Starting to load model" freeze. Force the classic
# HTTP/LFS download path instead.
export HF_HUB_DISABLE_XET=1

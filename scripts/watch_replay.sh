#!/usr/bin/env bash
# watch_replay.sh — one-shot progress snapshot (re-run to refresh; safe in a `!` cell).
#   bash scripts/watch_replay.sh [OUT]
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
OUT="${1:-data/eval/matrix}"
PIDF="$OUT/replay.pid"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  echo "STATUS: RUNNING (PID $(cat "$PIDF"))"
else
  echo "STATUS: not running (finished, or not started)"
fi

echo "--- responses written (each climbs toward the job count) ---"
wc -l "$OUT"/responses_*.jsonl 2>/dev/null || echo "(none yet)"

echo "--- last 25 log lines ---"
tail -n 25 "$OUT/replay.log" 2>/dev/null || echo "(no log yet)"

#!/usr/bin/env bash
# watch_replay.sh — one-shot progress snapshot (re-run to refresh; safe in a `!` cell).
#   bash scripts/watch_replay.sh [OUT]
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
OUT="${1:-data/eval/matrix}"
PIDF="$OUT/replay.pid"
source "$REPO_DIR/scripts/replay_common.sh"

if [ -f "$PIDF" ] && is_alive "$(cat "$PIDF" 2>/dev/null)"; then
  echo "STATUS: RUNNING (PID $(cat "$PIDF"))"
elif [ -f "$PIDF" ]; then
  echo "STATUS: DEAD — pidfile PID $(cat "$PIDF") is gone or a <defunct> zombie. Run: bash scripts/stop_replay.sh"
else
  echo "STATUS: not running (finished, or not started)"
fi

# GPU snapshot: during model load/compile the log is silent for minutes but memory
# climbs — that's WORKING, not frozen. ~0 MiB used while STATUS says RUNNING = wrong.
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

echo "--- responses written (each climbs toward the job count) ---"
wc -l "$OUT"/responses_*.jsonl 2>/dev/null || echo "(none yet)"

echo "--- last 25 log lines ---"
tail -n 25 "$OUT/replay.log" 2>/dev/null || echo "(no log yet)"

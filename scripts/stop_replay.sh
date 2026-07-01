#!/usr/bin/env bash
# stop_replay.sh — stop a detached replay run.
#   bash scripts/stop_replay.sh [OUT]
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
OUT="${1:-data/eval/matrix}"
PIDF="$OUT/replay.pid"

[ -f "$PIDF" ] || { echo "no pidfile at $PIDF — nothing to stop"; exit 0; }
PID="$(cat "$PIDF")"

if kill -0 "$PID" 2>/dev/null; then
  # kill the whole process group (bash + the running python/vLLM child); fall back to the pid
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  echo "sent SIGTERM to replay (PID/group $PID)"
else
  echo "not running (PID $PID)"
fi

#!/usr/bin/env bash
# stop_replay.sh — stop a detached replay run. SIGTERM first, then waits (up to
# STOP_TIMEOUT seconds — a mid-torch.compile process can be slow to notice the
# signal), and escalates to SIGKILL (+ a pkill -f safety net for any child that
# escaped the process group, e.g. vLLM's EngineCore) if it's still alive.
#   bash scripts/stop_replay.sh [OUT]
#   STOP_TIMEOUT=30 bash scripts/stop_replay.sh
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
OUT="${1:-data/eval/replay-v2}"
PIDF="$OUT/replay.pid"
TIMEOUT="${STOP_TIMEOUT:-20}"
source "$REPO_DIR/scripts/replay_common.sh"

[ -f "$PIDF" ] || { echo "no pidfile at $PIDF — nothing to stop"; exit 0; }
PID="$(cat "$PIDF")"

if ! is_alive "$PID"; then
  echo "not running (PID $PID is dead or a zombie) — clearing pidfile"
  rm -f "$PIDF"
  exit 0
fi

# kill the whole process group (bash + the running python/vLLM child); fall back to the pid
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
echo "sent SIGTERM to replay (PID/group $PID), waiting up to ${TIMEOUT}s..."

for ((i = 0; i < TIMEOUT; i++)); do
  is_alive "$PID" || break
  sleep 1
done

if is_alive "$PID"; then
  echo "still alive after ${TIMEOUT}s — escalating to SIGKILL"
  kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null
  pkill -9 -f build_response_matrix.py 2>/dev/null
  pkill -9 -f EngineCore 2>/dev/null
  sleep 1
fi

if is_alive "$PID"; then
  echo "WARNING: PID $PID still alive after SIGKILL — check manually (ps -o pid,state,cmd -p $PID)"
else
  echo "stopped (any <defunct> zombie left behind is harmless — it holds no GPU/RAM)."
  rm -f "$PIDF"
fi

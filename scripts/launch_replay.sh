#!/usr/bin/env bash
# launch_replay.sh — start the replay DETACHED (survives disconnect + kernel restart).
# Safe to run from a Jupyter `!` cell: it does not end in `&`, so IPython won't reject it.
#
#   bash scripts/launch_replay.sh [JOBS] [OUT] [MODEL...]
#
# Uses .venv-replay/bin/python by default; override with PYTHON=/path/to/python.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

JOBS="${1:-data/eval/replay-v2/jobs_v2.jsonl}"
OUT="${2:-data/eval/replay-v2}"
LOG="$OUT/replay.log"
PIDF="$OUT/replay.pid"
export PYTHON="${PYTHON:-$REPO_DIR/.venv-replay/bin/python}"
source "$REPO_DIR/scripts/replay_common.sh"

# Resolve HF_TOKEN now so the detached run inherits it (kills the unauthenticated
# download stall). Reads .claude/settings.local.json / env / .env — see the script.
source "$REPO_DIR/scripts/resolve_hf_token.sh"

mkdir -p "$OUT"

if [ -f "$PIDF" ] && is_alive "$(cat "$PIDF" 2>/dev/null)"; then
  echo "already running (PID $(cat "$PIDF")).  stop it first:  bash scripts/stop_replay.sh"
  exit 1
fi
rm -f "$PIDF"   # stale/zombie pidfile from a dead run — clear it so the guard is honest

EXTRA_MODELS=("${@:3}")
# setsid = new session (outlives the kernel); nohup = ignore SIGHUP; redirect to log.
setsid nohup bash "$REPO_DIR/scripts/run_replay.sh" "$JOBS" "$OUT" "${EXTRA_MODELS[@]}" > "$LOG" 2>&1 < /dev/null &
sleep 2

echo "launched (PID $(cat "$PIDF" 2>/dev/null || echo '?'))   log: $LOG"
echo "monitor:  bash scripts/watch_replay.sh"
echo "stop:     bash scripts/stop_replay.sh"

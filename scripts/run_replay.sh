#!/usr/bin/env bash
# run_replay.sh — replay every routing candidate over the extracted jobs to build
# the response matrix. Runs the models cheap→expensive, one at a time (GPU freed
# between models), and keeps going if one model fails so you still get the rest.
#
# Usage (from anywhere — the script cd's to the repo root itself):
#     bash scripts/run_replay.sh [JOBS] [OUT]
#
# Override the interpreter / knobs via env vars:
#     PYTHON=/home/leduc/ai-smart-routing/.venv/bin/python \
#     GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768 BATCH=512 \
#     bash scripts/run_replay.sh
#
# Detached run that survives disconnect (see progress in the log):
#     nohup bash scripts/run_replay.sh > data/eval/matrix/replay.log 2>&1 &
#     tail -f data/eval/matrix/replay.log
#
# Safe to re-run: replay skips prompt_ids already written per model.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

JOBS="${1:-data/eval/matrix/jobs.jsonl}"
OUT="${2:-data/eval/matrix}"
PYTHON="${PYTHON:-python}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BATCH="${BATCH:-512}"

MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "Qwen/Qwen3.5-35B-A3B-FP8"
  "openai/gpt-oss-120b"
  "Qwen/Qwen3.5-122B-A10B-FP8"
)

if [ ! -f "$JOBS" ]; then
  echo "ERROR: jobs file not found: $JOBS  (run the extract stage first)" >&2
  exit 1
fi
mkdir -p "$OUT"
echo $$ > "$OUT/replay.pid"        # record own PID so launch/watch/stop can find us
echo "repo=$REPO_DIR"
echo "jobs=$JOBS  out=$OUT  batch=$BATCH  gpu_mem_util=$GPU_MEM_UTIL  max_model_len=$MAX_MODEL_LEN"
echo "python=$("$PYTHON" -c 'import sys; print(sys.executable)')"

for M in "${MODELS[@]}"; do
  echo "===== $(date +%H:%M:%S)  $M ====="
  "$PYTHON" scripts/build_response_matrix.py replay \
      --jobs "$JOBS" --model "$M" --out "$OUT" \
      --gpu-mem-util "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --batch "$BATCH" \
      || echo "!!!!! $M FAILED (continuing) !!!!!"
done

echo "===== ALL DONE $(date +%H:%M:%S) ====="

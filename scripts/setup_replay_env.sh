#!/usr/bin/env bash
# setup_replay_env.sh — ONE-TIME env prep on the H200 box. Upgrades vLLM in
# .venv-replay so it registers the Qwen3.5 MoE architecture (needed for
# Qwen3.5-35B-A3B-FP8 and Qwen3.5-122B-A10B-FP8), then verifies. gpt-oss-120b and
# the Qwen3-30B already work on the current vLLM; run this before trusting the two
# Qwen3.5 rows.
#
#   bash scripts/setup_replay_env.sh
#
# Override the interpreter with PYTHON=/path/to/python.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-$REPO_DIR/.venv-replay/bin/python}"
source "$REPO_DIR/scripts/resolve_hf_token.sh"   # config downloads in the check below want auth too

echo "=== interpreter ==="
"$PYTHON" -c 'import sys; print(sys.executable)' || { echo "ERROR: $PYTHON not usable"; exit 1; }

echo "=== vLLM before ==="
"$PYTHON" -c 'import vllm; print("vllm", vllm.__version__)' || echo "(vLLM not importable yet)"

echo "=== upgrading vLLM ==="
"$PYTHON" -m pip install -U vllm

echo "=== installing hf_transfer (fast parallel HF downloads) ==="
"$PYTHON" -m pip install -U hf_transfer

echo "=== vLLM after ==="
"$PYTHON" -c 'import vllm; print("vllm", vllm.__version__)'

echo "=== Qwen3.5 MoE arch support ==="
"$PYTHON" - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry as R
archs = set(R.get_supported_archs())
hits = sorted(a for a in archs if "Qwen3_5" in a or "Qwen35" in a)
if hits:
    print("OK — registered:", hits)
    print("PASS: the Qwen3.5 models will load. Run: bash scripts/launch_replay.sh")
else:
    print("MISSING: no Qwen3.5 MoE arch in this vLLM build.")
    print("FAIL: pin to a newer vLLM whose changelog lists Qwen3_5Moe*, then re-run this script.")
PY

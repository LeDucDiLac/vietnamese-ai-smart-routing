#!/usr/bin/env bash
# resolve_cuda_home.sh — SOURCE to export CUDA_HOME if a CUDA toolkit is present.
#
# vLLM's FP8-MoE load path (deep_gemm / Triton / FlashInfer) JIT-compiles kernels
# and needs nvcc/ptxas. Without a toolkit it logs "CUDA_HOME is None" and then HANGS
# during "Starting to load model" (weights sit on the GPU at ~0% util forever) — it
# does not fail cleanly. This points CUDA_HOME at a toolkit so that path works.
#
# Checks (first hit wins): existing $CUDA_HOME, nvcc on PATH, /usr/local/cuda,
# the active conda prefix. If none is found it warns loudly with the install fix.

_cuda=""
for c in "${CUDA_HOME:-}" /usr/local/cuda "${CONDA_PREFIX:-}"; do
  if [ -n "$c" ] && [ -x "$c/bin/nvcc" ]; then _cuda="$c"; break; fi
done
if [ -z "$_cuda" ] && command -v nvcc >/dev/null 2>&1; then
  _cuda="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [ -n "$_cuda" ] && [ -x "$_cuda/bin/nvcc" ]; then
  export CUDA_HOME="$_cuda"
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1))"
else
  echo "WARNING: no CUDA toolkit (nvcc) found — the FP8-MoE model load will likely HANG." >&2
  echo "         Install one into the replay env, e.g.:  conda install -y -c nvidia cuda-nvcc cuda-cudart-dev" >&2
  echo "         then re-run. (This is the 'CUDA_HOME is None' warning vLLM has been printing.)" >&2
fi

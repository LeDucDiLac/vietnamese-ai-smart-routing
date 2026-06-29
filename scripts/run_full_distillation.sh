#!/usr/bin/env bash
# Full student distillation + evaluation pipeline — sequential, versioned, MLflow-tracked.
#
# For each student, in sequence (one at a time → fits the 12GB H200 partition):
#   1. distill from its v2 teacher          (AMP, logged to MLflow)
#   2. export to INT8 ONNX
#   3. evaluate immediately on BOTH:
#        - the production logs   (eval_logs.py  → data/eval/intern_data.csv)
#        - the eval set          (eval_router.py → routing_testset.jsonl, torch + INT8 ONNX)
#      also logged to MLflow.
#
# VERSIONING: every invocation gets a RUN id (timestamp by default, or set a
# semantic one). All artifacts go under runs/students/<RUN>/ and MLflow runs are
# named "<RUN>/<student>-{distill,eval}" + tagged pipeline_run=<RUN>, so iterations
# never clobber each other on disk or blur together in MLflow.
#
# Launch detached (survives logout):
#   RUN=v1-t2-a0.5 nohup bash scripts/run_full_distillation.sh &   # named iteration
#   nohup bash scripts/run_full_distillation.sh &                  # timestamp id
#   tail -f runs/students/<RUN>/pipeline.log
#   mlflow ui --backend-store-uri ./mlruns        # filter by tag pipeline_run=<RUN>
#
# Override any setting via env, e.g. BATCH=24 EPOCHS=4 PYTHON=".venv/bin/python" RUN=...

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export VI_ROUTER_REPO_ROOT="$REPO"

# ── config (override via env) ────────────────────────────────────────────────
PYTHON="${PYTHON:-uv run --extra ml python}"
# v2 dataset dir differs by machine (data/v2 on the H200, data/processed/v2 locally).
if [[ -z "${DATA_ROOT:-}" ]]; then
  for cand in data/v2 data/processed/v2; do
    if [[ -f "$cand/train.jsonl" ]]; then DATA_ROOT="$cand"; break; fi
  done
  DATA_ROOT="${DATA_ROOT:-data/v2}"
fi
TEACHERS_ROOT="${TEACHERS_ROOT:-runs/teachers}"
OUT_ROOT="${OUT_ROOT:-runs/students}"
RUN="${RUN:-$(date +%Y%m%d-%H%M%S)}"             # version id for this iteration
SCHEMA="${SCHEMA:-v2}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"                             # 32 ⇒ ~9GB peak (granite), fits 12GB
TEMP="${TEMP:-2.0}"
ALPHA="${ALPHA:-0.5}"
MAX_STEPS="${MAX_STEPS:-}"                        # cap steps/student (smoke-test the wiring)
STUDENTS="${STUDENTS:-}"                          # space-separated subset, e.g. "vi-router-fast-granite"
CSV="${CSV:-data/eval/intern_data.csv}"          # production logs
TESTSET="${TESTSET:-data/eval/routing_testset.jsonl}"
MLFLOW_EXP="${MLFLOW_EXP:-vi-smart-routing}"
MLFLOW_URI="${MLFLOW_URI:-$REPO/mlruns}"

if [[ ! -f "$DATA_ROOT/train.jsonl" ]]; then
  echo "ERROR: no train.jsonl under DATA_ROOT='$DATA_ROOT'. Set DATA_ROOT=<dir with train/val/test.jsonl>." >&2
  exit 1
fi

RUN_DIR="$OUT_ROOT/$RUN"                          # all artifacts for this run live here
mkdir -p "$RUN_DIR"
export MLFLOW_TRACKING_URI="$MLFLOW_URI"
# Newer MLflow gates the ./mlruns file store behind this opt-out; we use file store.
export MLFLOW_ALLOW_FILE_STORE=true
# Tee everything into a versioned log (works under nohup too).
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

# teacher_dir : student_name  (run in this order)
PAIRS=(
  "vi-router-quality-granite:vi-router-fast-granite"   # v2 accuracy/MAE winner
  "vi-router-quality-mmbert:vi-router-fast-mmbert"     # v2 test-F1 winner
  "vi-router-quality:vi-router-fast"                   # mDeBERTa → MiniLM baseline
)

echo "=================================================================="
echo " STUDENT DISTILLATION PIPELINE  |  RUN=$RUN  |  start $(date)"
echo " artifacts : $RUN_DIR"
echo " data=$DATA_ROOT  schema=$SCHEMA  epochs=$EPOCHS  batch=$BATCH"
echo " MLflow    : experiment=$MLFLOW_EXP  uri=$MLFLOW_URI  (tag pipeline_run=$RUN)"
echo " python    : $PYTHON"
echo "=================================================================="

for pair in "${PAIRS[@]}"; do
  teacher="${pair%%:*}"
  student="${pair##*:}"
  tdir="$TEACHERS_ROOT/$teacher"
  sdir="$RUN_DIR/$student"

  # Optional subset filter (STUDENTS="a b") — skip students not in the list.
  if [[ -n "$STUDENTS" && " $STUDENTS " != *" $student "* ]]; then continue; fi

  if [[ ! -f "$tdir/model.pt" ]]; then
    echo; echo "### SKIP $student — teacher checkpoint missing at $tdir"; continue
  fi

  echo; echo "##################################################################"
  echo "### [$(date)] DISTILL  $student   <-   $teacher"
  echo "##################################################################"
  $PYTHON -m classifier.distill \
    --teacher "$tdir" --out "$sdir" --student "$student" \
    --data "$DATA_ROOT" --schema-version "$SCHEMA" \
    --epochs "$EPOCHS" --batch-size "$BATCH" \
    --temperature "$TEMP" --alpha "$ALPHA" \
    --run-name "$RUN" ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"

  echo; echo "### [$(date)] EXPORT  $student  ->  INT8 ONNX"
  if ! $PYTHON -m classifier.export_onnx \
        --checkpoint "$sdir/model.pt" --model-name "$student" \
        --out-dir "$sdir/onnx" --schema-version "$SCHEMA"; then
    echo "### WARN: export failed for $student — eval will use the torch path only"
  fi

  echo; echo "### [$(date)] EVALUATE  $student  (production logs + routing eval set)"
  $PYTHON scripts/eval_all_students.py \
    --students-root "$RUN_DIR" --teachers-root "$TEACHERS_ROOT" \
    --students "$student" --csv "$CSV" --testset "$TESTSET" \
    --schema-version "$SCHEMA" --run-name "$RUN" --python "$PYTHON" \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"
done

echo; echo "##################################################################"
echo "### [$(date)] FINAL comparison across all students  (RUN=$RUN)"
echo "##################################################################"
# Aggregation pass: combined table over all students. --no-mlflow so it doesn't
# duplicate the per-student eval runs already logged in the loop above.
$PYTHON scripts/eval_all_students.py \
  --students-root "$RUN_DIR" --teachers-root "$TEACHERS_ROOT" \
  --csv "$CSV" --testset "$TESTSET" --schema-version "$SCHEMA" \
  --run-name "$RUN" --python "$PYTHON" --no-mlflow \
  ${STUDENTS:+--students $STUDENTS} || true

echo; echo "=== PIPELINE DONE $(date) — RUN=$RUN — artifacts in $RUN_DIR, metrics in MLflow ($MLFLOW_EXP) ==="

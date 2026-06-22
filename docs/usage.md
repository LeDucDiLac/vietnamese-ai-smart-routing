# Usage Guide

Vietnamese prompt task & complexity classifier + smart router for the AI Gateway.
This guide covers install, the offline (CPU, no-ML) paths, the GPU pipeline on
Kaggle, and serving. See `plan.md` for the design rationale.

## 1. Layout

```
configs/        label schema, complexity weights, model registry
src/
  config.py     single config loader (pure-Python, no ML)
  data/         crawl, teacher-label, synth-gen, build-dataset (M1)
  classifier/   model, train, distill, export_onnx, infer, tokenization (M2/M3)
  router/       capability, policy, match, leaderboard (M4)
  serving/      api.py — FastAPI /route shim (M4)
  sim/          routerbench_adapter, routellm_router, vi_response_cache (§11)
tests/
  unit/         pure-Python tests (no ML deps)
  eval/         simulate.py — the 4 handover metrics (M5)
kaggle_run.py   one entrypoint for the whole pipeline (Kaggle-friendly)
```

## 2. Install

Two dependency tiers. The router/sim/serving paths need only the base deps; the
classifier (train/distill/export) needs the heavy `ml` extra and runs on Kaggle.

```bash
# base — pure-Python paths (router, sim, eval, tests)
uv sync

# heavy ML stack (train/distill/export) — normally only on Kaggle
uv sync --extra ml

# serving / data crawl extras
uv sync --extra serve
uv sync --extra data
```

## 3. Offline paths (CPU, no GPU, no ML deps)

Everything except training runs locally with zero GPU.

### Build a smoke dataset

```bash
# synthetic VN prompts spanning 11 task types x 3 complexity tiers
uv run python -m data.synth_gen --out data/synthetic/synth.jsonl --per-cell 60

# assemble train/val/test (silver -> train; gold -> val+test)
uv run python -m data.build_dataset \
    --silver data/synthetic/synth.jsonl \
    --gold data/gold/gold.jsonl \
    --out data/processed
```

### Refresh the capability leaderboard (success criterion #1)

```bash
uv run python -m router.leaderboard --out runs/leaderboard
# -> runs/leaderboard/leaderboard.json + leaderboard.md
```

Re-running after editing `configs/sim_models.yaml` (new model / new scores)
regenerates the ranking — that is the "auto-updated leaderboard".

### Run the routing simulation (the 4 metrics)

```bash
# build a one-time VN response cache (synthetic mode = no API spend)
uv run python -m sim.vi_response_cache \
    --prompts data/processed/train.jsonl \
    --out data/sim/vi_cache.jsonl --mode synthetic

# replay always-best / always-cheap / our-router and emit the report
uv run python -m tests.eval.simulate \
    --data data/sim/vi_cache.jsonl \
    --user-groups premium engineering \
    --out runs/eval/report.json
```

> Pass `--user-groups` so the router can actually reach the gated models;
> otherwise it is compared against an always-best baseline that *can* reach them
> and the quality delta looks artificially bad.

## 4. GPU pipeline (Kaggle)

All GPU work runs on Kaggle, driven by `kaggle_run.py`. Upload the repo as a
Kaggle dataset, then in one notebook cell:

```python
!cp -r /kaggle/input/ai-smart-routing /kaggle/working/repo
%cd /kaggle/working/repo
!python kaggle_run.py --install --steps all --epochs 3
```

Stages (run in order, or pick a subset):

| step | needs GPU | what it does |
|---|---|---|
| `synth` | no | generate synthetic VN prompts |
| `dataset` | no | assemble train/val/test |
| `train` | yes | train `vi-router-quality` (mDeBERTa-v3) |
| `distill` | yes | distill into `vi-router-fast` (MiniLM student) |
| `export` | no | ONNX + INT8 quantize the student |
| `leaderboard` | no | refresh the capability leaderboard |
| `simulate` | no | run the eval, emit the 4 metrics |

Run a subset:

```bash
# CPU-only stages locally
python kaggle_run.py --steps synth dataset leaderboard simulate

# GPU stages on Kaggle
python kaggle_run.py --install --steps train distill export
```

Outputs land under `--out-root` (default `/kaggle/working/runs` on Kaggle,
`runs/` locally) for download.

### Real data instead of synthetic

```bash
# crawl Vietnamese corpora (needs the `data` extra)
python -m data.crawl --out data/raw/crawl.jsonl --max-per-source 2000

# teacher-label them (NVIDIA model on Kaggle GPU; LLM on any machine)
python -m data.label_teacher --mode nvidia --in data/raw/crawl.jsonl \
    --out data/raw/labeled_nvidia.jsonl
python -m data.label_teacher --mode llm --in data/raw/crawl.jsonl \
    --out data/raw/labeled_llm.jsonl

# reconcile the two teachers -> agreed silver + disagreement gold queue
python -m data.label_teacher --mode reconcile \
    --nvidia data/raw/labeled_nvidia.jsonl --llm data/raw/labeled_llm.jsonl \
    --out data/raw/labeled.jsonl --gold-out data/gold/queue.jsonl
```

`build_dataset` picks up `data/raw/labeled.jsonl` automatically if present.

## 5. Serving

```bash
# stub backend (no model needed — exercises the routing path end-to-end)
uv run --extra serve uvicorn serving.api:app --port 8000

# real INT8 ONNX backend
VI_ROUTER_ONNX=runs/onnx/model.int8.onnx \
VI_ROUTER_BACKBONE=microsoft/Multilingual-MiniLM-L12-H384 \
uv run --extra serve uvicorn serving.api:app --port 8000
```

Request:

```bash
curl -s localhost:8000/route -H 'content-type: application/json' -d '{
  "prompt": "Viết một hàm Python tính giai thừa và viết unit test.",
  "user_groups": ["engineering"]
}' | jq
```

The response carries the chosen `model_id`, the full classifier analysis
(task type + 6 complexity dims + `prompt_complexity_score`), the reason, and the
routing overhead in ms. The Gateway forwards to `model_id` — no client changes.

Endpoints: `POST /route`, `GET /leaderboard?task_type=...`, `GET /health`.

### Serving env vars

| var | meaning |
|---|---|
| `VI_ROUTER_ONNX` | path to INT8 ONNX export (preferred path) |
| `VI_ROUTER_BACKBONE` | tokenizer backbone for the ONNX model |
| `VI_ROUTER_MAX_TOKENS` | tokenizer cap (default 256) |
| `VI_ROUTER_TORCH` | torch checkpoint dir (quality path) |
| `VI_ROUTER_LATENCY_CEIL` | default latency ceiling (ms/1k) for matching |

If neither ONNX nor torch is configured, a deterministic heuristic stub serves —
useful for tests and wiring checks before a model is trained.

## 6. Swapping in real Viettel models

Replace `configs/sim_models.yaml` with the real model list (same schema: per-task
skill scores, `cost_per_1k_tokens`, `latency_ms_per_1k`, `permissions`). Nothing
else changes — the leaderboard, matcher, and eval all read that one file.

## 7. Tests & lint

```bash
uv run pytest -q          # 36 pure-Python unit tests, no ML deps
uv run ruff check src tests kaggle_run.py
```

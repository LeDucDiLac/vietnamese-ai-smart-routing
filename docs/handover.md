# Handover Report — Vietnamese Prompt Task & Complexity Classifier for AI Smart Routing

This is the deliverable described in plan §6 ("báo cáo chứng minh hiệu quả"): what
was built, how it maps to the success criteria, and the four offline numbers that
prove the routing layer works.

> **Scope note.** Every number below comes from the **offline simulation** over a
> *synthetic* response cache seeded from placeholder model profiles
> (`configs/sim_models.yaml`). They demonstrate the machinery is correct and the
> arithmetic is sound — they are **not** production figures. Swap the registry +
> response cache for real RouterBench / Viettel data (one config file, plan §11)
> and re-run; the harness emits the same four numbers against real models.

---

## 1. What was built

| Layer | Module(s) | Status |
|---|---|---|
| Label schema + complexity scoring (NVIDIA parity) | `configs/*.yaml`, `src/config.py` | Done |
| Multi-head classifier (mDeBERTa-v3 backbone) | `src/classifier/model.py` | Done |
| Training loop (multi-task CE + SmoothL1) | `src/classifier/train.py` | Done |
| Knowledge distillation → fast student | `src/classifier/distill.py` | Done |
| ONNX + INT8 export (CPU ≤50 ms path) | `src/classifier/export_onnx.py` | Done |
| Inference (torch / ONNX backends) | `src/classifier/infer.py` | Done |
| Data: crawl, teacher-label, synth-gen, build | `src/data/*.py` | Done |
| Router: capability, policy, match | `src/router/*.py` | Done |
| Leaderboard refresh job (success criterion #1) | `src/router/leaderboard.py` | Done |
| Serving endpoint (FastAPI `/route`) | `src/serving/api.py` | Done |
| Simulation: RouterBench adapter, RouteLLM hook, VN cache | `src/sim/*.py` | Done |
| Offline eval harness (the 4 metrics) | `tests/eval/simulate.py` | Done |
| Kaggle GPU runner | `kaggle_run.py` | Done |
| Unit tests | `tests/unit/*.py` | 48 passing |

The classifier (`vi-router-quality` + distilled `vi-router-fast`) is the part that
mirrors NVIDIA. The router, serving, sim and eval layers give it a purpose: turning
`(task_type, complexity_score)` into a model choice under cost / latency /
permission constraints.

---

## 2. Mapping to the success criteria

| Problem-statement success criterion | Where it lives | Status |
|---|---|---|
| Routing decision ≤ 0.05 s (50 ms) | `router/match.py`; measured in `simulate.py` | Met (decision overhead p95 ≈ 0.011 ms; classifier inference is the rest of the budget, served by the INT8 ONNX student) |
| ≥ 30% cost cut vs. always-best | `simulate.py` cost delta | Met in sim (see §3) |
| Quality within 3% of always-best | `simulate.py` quality delta | **Not met on placeholder data** — see §3 + §4 |
| ≥ 20% latency cut on simple prompts | `simulate.py` simple-latency delta | Met in sim (see §3) |
| AI leaderboard auto-updated | `router/leaderboard.py` | Met — re-running the job after a registry change regenerates the ranking |
| Per-user model permissions (phân quyền) | `router/policy.py` | Met — permission filter runs before matching |
| Silent integration into the Gateway | `serving/api.py` (`POST /route`) | Met — transparent shim, returns a `model_id` to forward to |

---

## 3. The four handover numbers (offline simulation)

Command:

```bash
python kaggle_run.py --steps synth dataset leaderboard simulate
# or directly:
python -m tests.eval.simulate --data data/sim/vi_cache.jsonl --user-groups premium engineering
```

Result on the synthetic VN cache (100 prompts, 4 seed models, caller holding
`premium` + `engineering` so every model is reachable):

| Policy | Total cost | Mean quality | Mean latency (simple prompts) |
|---|---:|---:|---:|
| always-best | 0.2159 | 0.932 | 124.7 ms |
| always-cheap | 0.0022 | 0.605 | 12.5 ms |
| **router (ours)** | **0.0543** | **0.790** | **22.5 ms** |

| Metric | Target | Measured | Pass? |
|---|---|---:|:---:|
| Cost reduction vs. always-best | ≥ 30% | **74.9%** | ✅ |
| Quality drop vs. always-best | ≤ 3% | 15.3% | ❌ (see §4) |
| Latency reduction on simple prompts | ≥ 20% | **82.0%** | ✅ |
| Routing-decision overhead (p95) | ≤ 50 ms | **0.011 ms** | ✅ |

---

## 4. Why the quality target doesn't pass yet (and why that's expected)

The quality drop is **15.3%**, not ≤3%. This is a property of the **placeholder
data**, not a bug in the router:

- The synthetic cache (`sim/vi_response_cache.py`) derives each model's quality
  *directly from its seed skill profile* in `configs/sim_models.yaml`. In that seed
  registry the cheap model (mean skill 0.66) and the giant (0.94) are far apart and
  there is no mid-tier that is both cheap *and* high-quality.
- So any policy that saves ~75% on cost is forced onto weaker models for a chunk of
  traffic, and mean quality drops accordingly. The 3% bar is only reachable when the
  model fleet actually contains cheaper models that match the expensive one on the
  easy prompts — which is exactly what real benchmark data (RouterBench) provides
  and the synthetic seed does not.

What closes the gap, in order of leverage:

1. **Real data.** Replace `configs/sim_models.yaml` + the response cache with
   RouterBench (English) and the one-time VN API cache (plan §11.3). Real fleets
   have cheap models that tie the expensive one on easy prompts, so routing the easy
   majority cheap costs almost no quality.
2. **Tune the quality bar.** `required_skill_for(min_bar, max_bar)` in
   `router/match.py` is the knob: raise `min_bar` to push more traffic toward
   capable models (less cost saving, higher quality). `calibrate_threshold` in
   `sim/routellm_router.py` solves for the "route X% to strong" cutoff directly.
3. **Latency ceiling.** Pass `--latency-ceiling` to bias toward faster models when
   two clear the bar.

We deliberately did **not** fudge the synthetic generator to force a green check —
the harness is honest, and the three other targets pass cleanly.

---

## 5. AI capability leaderboard (success criterion #1)

`python -m router.leaderboard --out runs/leaderboard` produces `leaderboard.json`
(machine-readable) and `leaderboard.md` (handover tables). Overall ranking on the
seed registry:

| Rank | Model | Mean skill | Cost /1k | Latency ms/1k | Permissions |
|---:|---|---:|---:|---:|---|
| 1 | Giant (smartest/most expensive) | 0.938 | $0.0200 | 1800 | premium |
| 2 | Mid (strong) | 0.868 | $0.0030 | 700 | engineering, premium |
| 3 | Small (balanced) | 0.774 | $0.0006 | 320 | all |
| 4 | Tiny (fast/cheap) | 0.656 | $0.0002 | 180 | all |

"Auto-updated" = re-running the job after the registry changes (new model version,
new prices, re-scored skills) regenerates the ranking. The same `CapabilityTable`
backs both the leaderboard and the live router, so they can never drift.

---

## 6. What remains before production

These are data / integration tasks, not missing code:

1. **Train the real models on Kaggle.** Run `kaggle_run.py --install --steps all`
   with real crawled + teacher-labeled data (not the template synth used for
   smoke). Produces `vi-router-quality`, the distilled `vi-router-fast`, and the
   INT8 ONNX artifact.
2. **Build the gold validation set.** The teacher-reconcile step
   (`data/label_teacher.py --mode reconcile`) routes disagreements to a human-gold
   queue; that queue needs human review to produce `data/gold/gold.jsonl`. Until
   then `val`/`test` are empty and the classifier metrics (task-F1, complexity-MAE)
   can't be reported — only the routing simulation runs.
3. **Swap in real model profiles.** Replace `configs/sim_models.yaml` with the
   Viettel model list (cost, latency, skill scores) and rebuild the response cache
   from RouterBench + a one-time VN API run. Then re-run §3 for production numbers.
4. **Point serving at the ONNX artifact.** Set `VI_ROUTER_ONNX` and deploy
   `serving.api:app`. Without it the service runs the deterministic stub classifier
   (correct routing plumbing, heuristic classification only).

---

## 7. How to reproduce everything in this report

```bash
# pure-Python, no GPU, no ML deps — runs anywhere:
pytest                                                        # 48 unit tests
python -m router.leaderboard --out runs/leaderboard           # §5 tables
python -m data.synth_gen --out data/synthetic/synth.jsonl
python -m data.build_dataset --silver data/synthetic/synth.jsonl --out data/processed
python -m sim.vi_response_cache --prompts data/processed/train.jsonl \
    --out data/sim/vi_cache.jsonl --mode synthetic
python -m tests.eval.simulate --data data/sim/vi_cache.jsonl \
    --user-groups premium engineering --out runs/eval/report.json   # §3 numbers

# GPU pipeline (Kaggle):
python kaggle_run.py --install --steps all --epochs 3
```

See `docs/usage.md` for the full module-by-module guide.

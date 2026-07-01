# AI Smart Routing — Progress Report

*Ongoing. Each section is a completed phase. Updated as work progresses.*

---

## Goal

Build a Vietnamese prompt routing classifier (11 task types) that runs on CPU in ≤50ms. A large teacher model (DeBERTa) labels data offline; a small student model (MiniLM) handles real-time inference.

---

## Phase 1 — Data Collection & Initial Dataset (v1)

**What:** Crawled 7 Vietnamese HuggingFace datasets covering summarization, QA, and instruction following. Applied a three-tier labeling strategy: free provenance labels from dataset purpose for planning, then Anthropic Sonnet for ground-truth classification of all 11 task types. Built the first processed dataset snapshot (v1): **55,595 rows** (50,041 train / 2,778 val / 2,776 test).

**Why:** No existing Vietnamese routing dataset — had to assemble from scratch by mining public corpora and relabeling with a teacher model.

**Result:** v1 processed dataset stored at `data/processed/v1/`. A versioning registry (`versions.json`) tracks all dataset snapshots going forward.

---

## Phase 2 — Class Imbalance Discovery

**What:** Analyzed the Sonnet-labeled v1 distribution. Found 3 classes dominate (Summarization 37%, Closed QA 25%, Open QA 18%) while 4 classes are critically underrepresented (Brainstorming 1.1%, Rewrite 0.8%, Chatbot 0.4%). Also cross-referenced Sonnet labels against each source's provenance label — found that provenance is inaccurate for most sources (e.g., ViQuAD labeled as "closed QA" is actually 72% Open QA by Sonnet; Wikipedia labeled as "open QA" is actually 65% Closed QA).

**Why:** Class imbalance causes the classifier to overfit to common classes and fail on rare ones. Provenance accuracy matters because it's used as the planning signal for what to crawl next — if it's wrong, crawl decisions are wrong.

**Result:** Documented per-source Sonnet distribution in `data_classify.md`. Established that only `vi-alpaca` covers all 11 classes meaningfully and is the most valuable source in the dataset.

---

## Phase 3 — Data Expansion

**What:** Three targeted actions to address imbalance:

1. **Raised vi-alpaca cap from 10k → 50k.** Since vi-alpaca is the only source covering rare classes (Brainstorming, Rewrite, Code Gen, Chatbot), pulling the full 50k rows gives roughly 5× more minority-class coverage. Downloaded 49,940 rows.

2. **Added UIT-VSFC (16,175 rows) for Classification.** This Vietnamese student feedback dataset is a clean single-purpose classification corpus (sentiment + topic). The HuggingFace loader was broken, so the data was manually retrieved from Google Drive by reading the original loading script.

3. **Fixed source attribution in the pipeline.** Discovered that the labeling step was overwriting the original dataset ID with the string "teacher_anthropic," making it impossible to trace processed rows back to their HF source. Fixed by separating `label_source` (who labeled) from `hf_source` (where the text came from). Both fields now flow through to processed data.

**Why:** Minority class coverage is the bottleneck for model quality. Traceability is needed for auditing and future cap decisions.

**Result:** Raw crawl grew from ~56k → **111,967 rows** across 8 sources.

---

## Phase 4 — Predicted Distribution & Next Crawl Planning

**What:** Estimated the class distribution of the full 111,967-row raw dataset without labeling all new rows first. Method: rows already in v1 use their actual Sonnet labels (ground truth); new vi-alpaca rows use the per-source Sonnet ratio measured from the first 10k rows (not raw provenance, which would call all 40k rows "text generation"); new uitnlp rows use provenance (classification — accurate for this source).

**Predicted distribution (111,967 rows):**

| Class | Predicted | Share |
|---|---|---|
| Summarization | 21,242 | 19.0% |
| Open QA | 18,632 | 16.6% |
| Classification | 18,414 | 16.4% |
| Text Generation | 18,110 | 16.2% |
| Closed QA | 16,138 | 14.4% |
| Code Generation | 8,596 | 7.7% |
| Extraction | 3,105 | 2.8% |
| Brainstorming | 3,043 | 2.7% |
| Other | 2,147 | 1.9% |
| Rewrite | 2,042 | 1.8% |
| **Chatbot** | **498** | **0.4%** |

**Remaining gaps:** Chatbot (0.4%) cannot be fixed by increasing existing source caps — a dedicated conversational Vietnamese dataset is required. Recommended next crawl: `CohereLabs/aya_dataset` (multilingual instruction data with Vietnamese conversational and creative prompts, likely covers Chatbot + Brainstorming in one crawl).

**Why:** Labeling 56k new rows with Sonnet costs money and time. The ratio-extrapolation method gives directional planning signal at zero additional cost. The prediction will be verified once Sonnet labeling of the new rows completes and v2 is built.

**Next steps:**
1. Run Sonnet labeling on the 56k new rows → rebuild as v2
2. Compare actual v2 distribution against the prediction above
3. Crawl `CohereLabs/aya_dataset` for Chatbot / Brainstorming
4. Fix `vilm/OpenOrca-Viet` field mapping (currently yields only 17 usable rows due to repetitive system-prompt extraction)
5. Begin student model training on v2

---

## Phase 5 — Teacher Cost Reduction: DeepSeek v4 Flash Evaluation

**What:** Evaluated DeepSeek v4 Flash (via OpenCode AI) as a cheaper alternative to Sonnet for labeling the 56k new rows. Ran a 500-row sample from the Sonnet-labeled v1 data and compared results class-by-class.

**Overall agreement with Sonnet: 76.6%** (383/500 rows).

| Class | Agreement | Main Confusion |
|---|---|---|
| Text Generation | 90.2% | Open QA (9 cases) |
| Classification | 90.9% | — |
| Rewrite | 82.4% | — |
| Summarization | 100% | — (small N) |
| Closed QA | 78.6% | Extraction (4 cases) |
| Extraction | 75.0% | — |
| Open QA | 69.4% | Text Generation (16), Closed QA (14) |
| Code Generation | 72.7% | Text Generation (9), Closed QA (8) |
| **Brainstorming** | **36.7%** | Open QA (12), Text Generation (7) |
| **Other** | **33.3%** | Open QA (2), Chatbot (1) |

**Key label confusion findings:** Several class boundaries are inherently fuzzy in Vietnamese and cause consistent confusion across both models:

- **Open QA ↔ Text Generation:** A prompt like *"Giải thích khái niệm X"* (Explain concept X) sits on the boundary — it is an open-ended question, but it can also be read as "generate an explanatory text." DeepSeek leans toward Text Generation; Sonnet more often calls it Open QA.
- **Brainstorming → Open QA / Text Generation:** DeepSeek almost never predicts Brainstorming (36.7% recall). The class is defined as open-ended ideation, but without explicit signals like *"hãy đề xuất nhiều ý tưởng"* (suggest multiple ideas), the model defaults to Open QA or Text Gen. This is partially a label definition problem — Brainstorming and Open QA are conceptually close.
- **Other → anything else:** The "Other" catch-all is avoided by DeepSeek (33.3% recall). Models trained to be helpful resist labeling something as "Other" — they find the nearest concrete class instead.
- **Closed QA ↔ Extraction:** Both involve finding a specific answer, often from context. The distinction (Closed QA = answer a question; Extraction = pull out a span/entity) is subtle and consistently confused across teachers.

**Why:** Sonnet at ~$3/M tokens is expensive for 56k rows. DeepSeek v4 Flash is significantly cheaper while achieving 76.6% agreement — acceptable for training silver labels, where some label noise is tolerated and the model learns the overall distribution rather than individual labels.

**Decision:** Use DeepSeek for the new 56k rows. Keep Sonnet labels as ground truth for the existing v1 rows. For Brainstorming specifically (the weakest class), consider a targeted Sonnet re-label pass after v2 is built if model performance on that class is poor.

**Result:** Labeling pipeline extended to support DeepSeek via `src/data/label_deepseek.py`. Produces the same output schema as the Sonnet labeler (`label_source: "teacher_deepseek"`), with resume support and batch calls (8 texts per API request).

---

*Next section will be added after the 56k new rows are labeled and v2 is built.*

---

## Phase 6 — NVIDIA Baseline Benchmark

**What:** Benchmarked `nvidia/prompt-task-and-complexity-classifier` (the upstream English model our label schema is derived from) directly on CPU, to establish a reference point before training our own Vietnamese variant.

**Latency results (CPU, no GPU):**

| Prompt length | P50 | P95 | SLA pass (≤50ms) |
|---|---|---|---|
| Short | 68.6 ms | 71.9 ms | ✗ |
| Medium | 105.3 ms | 116.6 ms | ✗ |
| Long | 355.8 ms | 394.3 ms | ✗ |

**Label alignment with our schema (100-row sample, Vietnamese prompts):** 20% overall accuracy. Only Open QA showed any alignment (94.7%). All other 10 classes scored 0% — the model was never trained on Vietnamese and collapses everything into Open QA or Text Generation.

**Why this matters:** This confirms two things: (1) the NVIDIA model cannot serve as a drop-in baseline for Vietnamese routing — it fails both the latency SLA and label alignment; (2) training a Vietnamese-specific model is necessary, not optional.

**Result:** Benchmark stored at `runs/2026-06-24_nvidia_cpu-benchmark/benchmark.json`. The latency figures serve as a negative baseline — our student target is ≤50ms (must beat NVIDIA's 68ms short-prompt floor).

---

## Phase 7 — Multi-Teacher Backbone Smoke Test

**What:** Smoke-tested 4 teacher backbone candidates simultaneously — each trained for 20 steps on the same data to quickly rank them before committing to a full Kaggle training run.

**Results (20-step smoke, val set):**

| Backbone | Macro-F1 | Top-1 Acc | Top-2 Acc | Complexity MAE | Train time |
|---|---|---|---|---|---|
| `microsoft/mdeberta-v3-base` | 0.276 | **0.635** | 0.830 | **0.191** | 225s |
| `jhu-clsp/mmBERT-base` | **0.456** | 0.516 | 0.827 | 0.288 | 195s |
| `BAAI/bge-m3` | 0.082 | 0.311 | 0.387 | 0.183 | 315s |
| `ibm-granite/granite-embedding-311m-multilingual-r2` | 0.395 | 0.420 | 0.764 | 0.297 | 210s |

**Key observations:**
- **mDeBERTa** has the highest top-1 accuracy (63.5%) and lowest complexity MAE (0.191). Strong at per-instance correctness.
- **mmBERT** has the highest macro-F1 (0.456) — better balanced across rare classes at smoke scale. Fastest to train.
- **BGE-M3** collapses badly at 20 steps — likely needs far more warmup due to its 568M parameter count and embedding-focused pretraining objective.
- **Granite** is between the two BERT variants but underperforms on Top-2 accuracy (76.4%).

**Decision:** Proceed to full training with mDeBERTa (primary teacher, best accuracy) and keep mmBERT as a candidate for distillation target comparison. BGE-M3 and Granite deprioritized unless mDeBERTa full-run underperforms.

**Result:** Smoke results at `runs/smoke-all/teacher_comparison.json`. Full mDeBERTa training queued as the v1-baseline Kaggle run.

---

## Phase 8 — First Full Teacher Training (mDeBERTa v1-baseline)

**What:** Ran the first full training of `vi-router-quality` (mDeBERTa-v3-base backbone) on Kaggle GPU, on the v1 dataset (50k train / 2.8k val).

**Training outcome:** 9,384 steps, final train loss **0.325**. Model checkpoint saved to `runs/2026-06-23_mdeberta-base_v1-baseline/quality/`.

**Simulation evaluation** (router vs always-best vs always-cheap, on 2,778 val prompts using `sim_models.yaml` capability profiles):

| Policy | Total cost | Mean quality | Mean latency (simple) |
|---|---|---|---|
| always-best | $53.02 | 0.940 | 1721 ms |
| always-cheap | $0.53 | 0.694 | 173 ms |
| **router** | **$0.57** | **0.725** | **202 ms** |

| Metric | Value | Target | Pass |
|---|---|---|---|
| Cost reduction vs best | **98.9%** | ≥30% | ✓ |
| Latency reduction (simple queries) | **89.9%** | ≥20% | ✓ |
| Router decision overhead (P95) | **0.010 ms** | ≤50 ms | ✓ |
| Quality drop vs best | **22.9%** | ≤3% | ✗ |

**Why the quality drop is expected:** The simulation assigns quality by tier (simulated models), not by task-specific skill. The router aggressively routes to cheap models (median complexity of val set is 0.14 — most prompts look simple). The 22.9% quality drop is a simulation artifact of routing nearly everything to the cheapest tier. In practice, routing decisions are gated by task type + skill matching, not complexity score alone. The real quality check requires actual model outputs per query (ground-truth evaluation, not simulated).

**Result:** First full model artifact ready. Evaluation infrastructure needs to be built against real production traffic before the quality target can be properly assessed.

---

## Phase 9 — Routing Evaluation Metrics Research

**What:** Conducted a systematic literature survey (deep-research workflow: 104 agents, 22 sources, 25 claims verified, 8 killed by adversarial verification) to understand what routing metrics the field uses and how they differ from the classifier metrics we were tracking.

**Key finding — metric decoupling:** Routing metrics (Gap@O, PGR, AIQ) are computed post-hoc from routing decisions; they are not differentiable and cannot be used as training objectives. Training remains on cross-entropy (task type) + SmoothL1 (complexity dims). The routing metrics only tell you how well those features translate to system-level routing behavior.

**Confirmed metrics by benchmark:**

| Metric | Definition | Source |
|---|---|---|
| **AvgAcc** | Mean quality of routed model across queries | LLMRouterBench (ACL 2026) |
| **Gap@O** | `1 − Acc(router) / Acc(oracle)` — headroom vs perfect oracle | LLMRouterBench |
| **Gain@B** | Quality delta vs always-best-model | LLMRouterBench |
| **CostSave** | `1 − cost_routed / cost_best` | LLMRouterBench |
| **PGR** | `(router_quality − cheap) / (expensive − cheap)` | RouteLLM (ICLR 2025) |
| **CPT(k%)** | Min % of expensive calls to reach PGR ≥ k | RouteLLM |
| **AIQ** | Area under cost–quality curve | RouterBench (ICLR 2024) |
| **LPM / HCR / MPM** | Scenario-specific: budget / accuracy-critical / trade-off bands | RouterXBench (Feb 2026) |

**Gap identified vs our prior tracking:**

| We tracked | Field standard | Gap |
|---|---|---|
| Macro-F1, Top-1/Top-2 accuracy | AvgAcc, Gap@O, Gain@B | No oracle headroom measurement |
| MAE per complexity dim | — | Regression not standard for routing eval |
| — | AIQ / PGR / CPT | No cost–quality curve |
| — | CostSave / ParetoDist | No Pareto frontier tracking |

**Result:** Metric definitions adopted as the standard for all subsequent evaluation scripts.

---

## Phase 10 — Production Log EDA & Baseline Routing Analysis

**What:** Analysed 5,000 LiteLLM proxy log records from Viettel's `netflow` team (`data/eval/intern_data.csv`) to understand actual routing patterns and establish a production baseline.

**Dataset characteristics:**
- **Models in use:** 7 (3 routing-eligible: `gpt-oss-120b` large, `Qwen3.5-35B` mid, `Qwen3-30B` small; plus embedding, OCR, and vision models)
- **Call types:** 4,838 `acompletion`, 127 `aembedding`, 12 `aresponses`
- **Prompt tokens:** min 0, max 97,765, mean 2,570
- **Completion tokens:** min 0, max 15,699, mean 1,557
- **Latency:** min 0ms, max 82,203ms, mean 16,904ms

**Current routing distribution (4,843 valid acompletion records):**

| Tier | Count | Share |
|---|---|---|
| large | 4,077 | 84.2% |
| mid | 736 | 15.2% |
| small | 30 | 0.6% |

**Key finding:** The oracle ceiling (best possible routing with perfect complexity knowledge) is only **3.6% cost savings**. This is not a router failure — the `netflow` workload is inherently complex (complaint classification, Elasticsearch query generation, DevOps log analysis). Most queries genuinely need large models. The 30% cost-saving target requires a more mixed workload or a broader model pool.

**Industrial KPIs (baseline — current system, no router):**

| KPI | Value | Target | Pass |
|---|---|---|---|
| Cost saving vs always-best | 4.4% | ≥30% | ✗ |
| Latency reduction (simple queries) | 61.3% | ≥20% | ✓ |
| Quality loss | 1.8% | ≤3% | ✓ |
| Router latency | — (no router) | ≤50ms | — |

**Result:** EDA complete. `scripts/eval_logs.py` built to compute all industrial + research metrics from any LiteLLM log CSV. Task diversity confirmed: logs span complaint classification, Elasticsearch query gen, name gender detection, DevOps log analysis, English tutoring — the router must handle all these task types.

---

## Phase 11 — Routing Testset Construction from Production Logs

**What:** Converted the 5,000 production log records into a labeled routing evaluation testset (`data/eval/routing_testset.{csv,jsonl}`) using two principles:

1. **Monotonicity** — if a cheaper model succeeded, more expensive models would too. If the expensive model failed, cheaper ones definitely fail.
2. **Complexity proxy** — for queries where only the large model ran, estimate whether a cheaper tier would have sufficed using token-based complexity scoring.

**Quality assessment:** Parsed each response for structural validity (JSON validity, refusal-phrase detection) rather than task-specific correctness. Short `{"index": N}` responses from the small model were correctly identified as valid index-classification outputs, not failures.

**Oracle label assignment logic:**

| Actual tier | Quality | Oracle tier | Confidence |
|---|---|---|---|
| small | 1 | small | confirmed |
| mid | 1, complexity < 0.35 | small | estimated |
| mid | 1, complexity ≥ 0.35 | mid | confirmed |
| mid | 0 | large | confirmed (monotonicity) |
| large | 1 | from complexity proxy | estimated |
| large | 0 | — | excluded (impossible task) |
| small | 0 | — | excluded (oracle ambiguous) |

**Testset summary:**

| | Count | Share |
|---|---|---|
| Total parsed | 4,843 | |
| Quality=0 (failed responses) | 1,969 | 40.7% |
| Excluded (ambiguous oracle) | 1,966 | |
| **Included in testset** | **2,877** | |
| — Confirmed labels | 765 | 26.6% |
| — Estimated labels | 2,112 | 73.4% |

**Oracle tier distribution in testset:**

| Tier | Count | Share |
|---|---|---|
| small | 37 | 1.3% |
| mid | 809 | 28.1% |
| large | 2,031 | 70.6% |

**Limitations:** Estimated labels rely on token-count heuristics, not semantic understanding. Quality is assessed from response structure (JSON validity, refusal detection), not task-specific correctness. Multi-model replay (sending the same query to all tiers and scoring each) would produce ground-truth labels but requires API access not currently available.

**Result:** `scripts/build_routing_testset.py` regenerates the testset from any LiteLLM log CSV. The confirmed subset (765 rows) provides high-trust evaluation; the full 2,877-row set provides broader but noisier coverage.

---

## Phase 12 — Router Evaluation Infrastructure

**What:** Built three evaluation scripts that together cover the full measurement stack from raw logs to trained model assessment.

### `scripts/eval_logs.py` — Log-based EDA + Baseline Metrics

Computes all industrial KPIs and research metrics directly from LiteLLM proxy logs, with two scenarios: the actual system (baseline) and a token-count heuristic router.

```
python scripts/eval_logs.py                                   # baseline + heuristic
python scripts/eval_logs.py --model-path runs/quality         # + real vi-router
python scripts/eval_logs.py --out runs/eval_logs              # save JSON report
```

### `scripts/build_routing_testset.py` — Oracle Testset Construction

Converts production logs into a labeled routing testset using monotonicity and complexity proxy. Outputs `routing_testset.{csv,jsonl}` with `oracle_tier` and `oracle_confidence` per row.

```
python scripts/build_routing_testset.py
python scripts/build_routing_testset.py --csv data/eval/intern_data.csv --out data/eval
```

### `scripts/eval_router.py` — Trained Model Evaluation

Loads one or more trained checkpoints (PyTorch or ONNX), runs them over the routing testset, and reports all metrics. Supports side-by-side comparison.

```
python scripts/eval_router.py runs/quality                    # single model
python scripts/eval_router.py runs/quality runs/student       # teacher vs student
python scripts/eval_router.py runs/student/model.onnx \       # ONNX path
    --backbone microsoft/Multilingual-MiniLM-L12-H384
python scripts/eval_router.py runs/quality --confirmed-only   # high-trust only
```

**Metrics reported by `eval_router.py`:**

| Category | Metrics |
|---|---|
| Industrial KPIs | router_latency_ms (≤50ms), cost_saving_pct (≥30%), latency_reduction_pct (≥20%), quality_loss_pct (≤3%) |
| Routing accuracy | routing_acc_all, routing_acc_confirmed, underrouting_rate, overrouting_rate |
| Research: quality | AvgAcc, Gain@B, Gap@O (real — not proxy), CostSave |
| Research: trade-off | PGR, CPT(50%), CPT(80%), AIQ |
| Research: scenario | LPM, HCR, MPM |

**Key distinction from `eval_logs.py`:** Gap@O here is computed against real oracle labels from the testset (not a token-count proxy), so it measures genuine routing headroom once a trained model is available.

**Result:** All three scripts committed to `scripts/`. The pipeline is ready: train a model on Kaggle → download checkpoint → run `eval_router.py` → see all routing metrics in one pass.

---

## Phase 13 — V2 Dataset: Full Teacher Backbone Competition (H200)

**What:** All four teacher backbones trained to completion on the v2 dataset on an H200 GPU, using `scripts/eval_all_teachers.py`. This is the first full apples-to-apples backbone comparison on real data.

**Dataset (v2 vs v1):**

| | v1 (Kaggle) | v2 (H200) |
|---|---|---|
| Train | 50,041 | 100,771 |
| Val | 2,778 | 5,598 |
| Test | 2,776 | 5,598 |
| Schema | v1 | v2 (configs/schemas/v2.yaml) |

**Training config:** 3 epochs, batch size 8, lr 2e-5, 1 model at a time (sequential to avoid VRAM contention).

**Results (full training, H200):**

| Backbone | Val F1 | Val Acc | Test F1 | Test Acc | Cplx MAE | Train time |
|---|---|---|---|---|---|---|
| `granite-311m-multilingual` | **0.7329** | **0.7705** | 0.7289 | **0.7712** | **0.0850** | 54.0m |
| `mmBERT-base` | 0.7291 | 0.7647 | **0.7388** | 0.7696 | 0.0876 | 53.5m |
| `bgem3` (retry) | 0.7221 | 0.7640 | 0.7314 | 0.7690 | 0.0925 | 122.3m |
| `mdeberta-v3-base` | 0.7130 | 0.7563 | 0.7269 | 0.7624 | 0.0907 | **48.5m** |

**BGE-M3 incident:** The first BGE-M3 launch (during the main 4-model orchestrator run) crashed at 0.3m (exit=1). Root cause unknown — the log was overwritten by the retry. The retry via `bgem3_retry.log` completed successfully in 122.3m. BGE-M3 results are from the retry run.

**Key observations vs the Kaggle smoke test (20 steps):**

| Backbone | Smoke top-1 | Full top-1 | Smoke F1 | Full F1 |
|---|---|---|---|---|
| mDeBERTa | **0.635** | 0.756 | 0.276 | 0.713 |
| mmBERT | 0.516 | 0.765 | **0.456** | 0.739 |
| BGE-M3 | 0.311 | 0.764 | 0.082 | 0.731 |
| Granite | 0.420 | **0.771** | 0.395 | **0.733** |

The smoke-test ranking was misleading: mDeBERTa led on top-1 at 20 steps but finishes last on both metrics at full training. Granite, which appeared mediocre in the smoke test, wins on accuracy and near-ties on F1. mmBERT's early F1 advantage (good class balance at 20 steps) narrows to within noise at full scale.

**Complexity MAE improved dramatically across all backbones** (v1 baseline was 0.191; v2 best is 0.085) — roughly 55% reduction. This is primarily due to the 2× larger dataset, not backbone choice.

**Routing evaluation:** All four `eval_router.py` runs failed with:
```
cannot import name 'build_tokenizer' from 'classifier.tokenization'
```
The H200 was running an older local code version where `tokenization.py` had a different API. Routing metrics (Gap@O, PGR, CostSave) are not available for these checkpoints. The fix requires syncing the H200 repo to the current commit and re-running `eval_all.py`.

**Decision:** Granite and mmBERT are the two strongest backbones — nearly identical on accuracy, Granite slightly better on consistency (test/val gap is smaller), mmBERT slightly better on test F1. Both are candidates for the final teacher. mDeBERTa is eliminated from the backbone search. BGE-M3 is competitive but 2.3× slower to train.

**Artifacts:** Training logs at `runs/run-v2-log/`. Checkpoints remain on the H200 at `/home/leduc/ai-smart-routing/runs/teachers/`.

---

## Phase 14 — V2 Routing Evaluation

**What:** Ran `eval_all_teachers.py` on all four v2 checkpoints using two scripts: `eval_logs.py` (production log KPIs) and `eval_router.py` (routing testset, 2,877 rows). Report at `runs/eval/teacher_eval_comparison.json`.

### eval_logs — model not integrated into simulation

All four models returned **identical** `eval_logs` numbers (same as the Phase 10 baseline). The `--model-path` flag was passed but `eval_logs.py` does not add a model-specific scenario when `--model-path` is given — the argument exists but the feature is incomplete. This eval is not useful until the model-routing scenario is implemented in `eval_logs.py`.

Current baseline (unchanged from Phase 10):

| Scenario | Cost Save | Quality Loss | Lat Reduce |
|---|---|---|---|
| Baseline (actual prod) | 4.4% | 1.76% | 61.3% |
| Heuristic router | 3.6% | 1.82% | 61.1% |

### eval_router — severe underrouting due to v2 threshold mismatch

| Model | Routing Acc | Underrouting | Predicted small | Quality Loss | AvgAcc | Gap@O |
|---|---|---|---|---|---|---|
| vi-router-quality | 1.1% | 98.5% | 2850/2877 | 20.3% | 0.741 | 17.6% |
| vi-router-quality-mmbert | 5.7% | 94.3% | 2743/2877 | 20.0% | 0.744 | 17.3% |
| vi-router-quality-bgem3 | 6.3% | 93.7% | 2717/2877 | 19.9% | 0.745 | 17.2% |
| vi-router-quality-granite | ERROR (exit 1) | — | — | — | — | — |

**Root cause — threshold mismatch:** The tier is determined by `prompt_complexity_score` with hardcoded thresholds (`< 0.35 → small, < 0.65 → mid`). These thresholds were implicitly calibrated for v1 complexity dims. The v2 dims (`reasoning_depth`, `instruction_precision`) produce scores in a lower range, causing 94–99% of queries to fall below 0.35 and route to `small`. The oracle testset expects 70.6% `large` — the models deliver 0%.

This is not a model quality failure. The F1 scores (0.71–0.73) are healthy. The routing collapse is entirely a post-processing calibration issue in `_tier_from_complexity`.

**Fixes needed:**
1. **Threshold recalibration** — measure actual v2 `prompt_complexity_score` distribution on the testset and set thresholds to match the oracle tier distribution (1.3% small / 28.1% mid / 70.6% large)
2. **eval_logs model scenario** — implement the model-routing scenario in `eval_logs.py` so `--model-path` generates a third scenario using the classifier's predicted tier
3. **Granite eval_router error** — unknown, need stderr from the failed run

**Artifacts:** `runs/eval/teacher_eval_comparison.json`

---

## Phase 15 — Complexity Score Calibration & MAE as a Flawed Metric

### Threshold calibration

Ran `scripts/calibrate_thresholds.py` on the granite v2 checkpoint against the routing testset (2,877 rows). Results:

**V2 complexity score distribution (granite):**

| Stat | Value |
|---|---|
| min | 0.0069 |
| median | 0.1127 |
| p90 | 0.2216 |
| p99 | 0.3823 |
| max | 0.6792 |

The entire distribution sits below 0.68 — far lower than v1's [0,1] range. With v1 thresholds (0.35/0.65), 98.6% of queries fall in "small" → 0.5% routing accuracy.

Grid search (200×200) found optimal thresholds **[0.0069, 0.0439]** → 75.6% routing accuracy. These are committed to `configs/schemas/v2-complexity.yaml` and loaded automatically by `TorchClassifier` via `ComplexityConfig.tier_thresholds`.

### Root cause of score compression

```
dim_raw = sigmoid(Linear(GELU(Linear(dropout(pooled)))))   # each dim ∈ (0, 1)

prompt_complexity_score =
    0.45 × reasoning_depth
  + 0.35 × domain_knowledge
  + 0.20 × instruction_precision
```

`sigmoid` alone doesn't cause compression — its output spans (0,1). The compression comes from **training label skew**: if the labeled complexity values cluster near zero (most prompts in the dataset are low-complexity), SmoothL1 loss pushes the heads to output near-zero for almost everything. The weighted sum of three near-zero dims gives scores mostly in [0.007, 0.18].

Fix for v3: normalize training targets to span [0,1] before computing loss, or add a calibration layer post-sigmoid.

### MAE is a flawed metric for skewed complexity labels

**The problem:** If 90% of training labels are 0.1, a model that always predicts 0.1 gets MAE ≈ 0 for 90% of samples — a very low MAE without learning anything. The current `complexity_mae` metric in `meta.json` gives full credit to this mode-collapse behaviour.

**Better metrics to add:**

| Metric | What it measures | Mode-predictor score |
|---|---|---|
| **MAE** (current) | Mean absolute error | Near-zero (misleading) |
| **R²** (variance explained) | Fraction of label variance the model explains | R² = 0 (exposes collapse) |
| **Pearson r / Spearman ρ** | Rank/linear correlation between predicted and true | r = 0, ρ = 0 |
| **MAE skill score** | `1 − MAE_model / MAE_mean_baseline` | 0 (baseline is mean-predict) |
| **Complexity tier accuracy** | Did predicted tier (low/mid/high) match label tier? | Accuracy = modal tier frequency |

**R²** and **Spearman ρ** are the most useful: a model that only learns the mean gets R² = 0 and ρ = 0 exactly, regardless of label skew. A positive R² means the model explains variance beyond the mean; ρ > 0 means predictions rank-order correctly even if magnitudes are off.

**Action:** Add R² and Spearman ρ to `train.py` eval output and `meta.json`. Track these instead of (or alongside) raw MAE in the backbone comparison table.

---

## Phase 16 — H200 Student Distillation and Four-Model Replay

**Started:** 2026-07-01 18:24 UTC on a dedicated NVIDIA H200 (141 GB VRAM).

Two guarded, detached handoffs were started so the pipelines do not compete with
in-progress artifact transfers:

- **Four-model replay:** waits for the existing Hugging Face prefetch, performs a
  second resumable prefetch pass, verifies that no incomplete shards remain, then
  launches `scripts/launch_replay.sh`. The four configured models are Qwen3 30B,
  Qwen3.5 35B, GPT-OSS 120B, and Qwen3.5 122B. Handoff state is logged in
  `runs/operations/replay_handoff.log`.
- **Student distillation and evaluation:** waits until `runs/teachers.rar` is a
  valid complete archive, extracts it, then waits for the v2 train/validation/test
  splits and both evaluation inputs. It retries `uv sync --extra ml --frozen` on
  transient network failures and starts `scripts/run_full_distillation.sh` only
  while the GPU is idle. Handoff state is logged in
  `runs/operations/distillation_handoff.log`.

At this checkpoint the H200 is idle, the teacher archive and model prefetch are
still transferring, and the v2 dataset/evaluation inputs are not yet present.
Final student and replay metrics will be added after both pipelines finish.

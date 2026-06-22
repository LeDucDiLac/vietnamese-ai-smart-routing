# Plan — Vietnamese Prompt Task & Complexity Classifier for AI Smart Routing

> Status: DRAFT for review. Nothing built yet. Read this, push back, then I implement.

## 0. TL;DR

We build a Vietnamese-capable clone of NVIDIA's [`prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) and wire it into the AI Gateway as the "dispatcher brain" (Người điều phối thông minh). The classifier reads each incoming prompt and emits:

- **task type** (11 classes: Open QA, Closed QA, Summarization, Text Generation, Code Generation, Chatbot, Classification, Rewrite, Brainstorming, Extraction, Other)
- **complexity dimensions** (creativity, reasoning, contextual knowledge, domain knowledge, constraints, few-shots)
- a single **prompt_complexity_score** via NVIDIA's weighted formula

The router then matches `(task_type, complexity)` against each model's **capability profile** + **cost/latency** + **user permissions** and picks the cheapest/fastest model that still clears the quality bar — in **≤ 50 ms**.

The hard part vs. NVIDIA: NVIDIA is **English-only** and their training data is **not public**. So the bulk of the work is **data** — we crawl Vietnamese corpora, generate labels via a teacher (the NVIDIA model + a strong LLM), and finetune a **multilingual encoder** on Vietnamese + mixed VN/EN/code prompts.

---

## 1. How this maps to the problem statement

| Problem statement step | This project's deliverable |
|---|---|
| "Đào tạo Người điều phối hiểu câu hỏi" (train the dispatcher to understand questions) | **The classifier** — the centerpiece of this plan |
| "Lập Hồ sơ năng lực cho từng AI" (capability profiles) | `router/capability.py` — per-model skill/cost/latency profile + leaderboard |
| "Thuật toán tự động ghép đôi" (auto-matching algorithm) | `router/match.py` — constrained selection (best quality at min cost/latency) |
| "Quản lý phức tạp / phân quyền" (per-user model permissions) | `router/policy.py` — permission filter before matching |
| "Lắp ráp vào hệ thống thực tế" (silent integration) | `serving/api.py` — drop-in routing endpoint for the Gateway |
| Success: routing ≤ 0.05s | Distilled + quantized model; 50 ms latency budget is a first-class design constraint |
| Success: ≥30% cost cut, quality within 3% | Offline evaluation harness comparing routed vs. always-best-model |
| Success: ≥20% latency cut on simple prompts | Same harness measures p50/p95 wait-time deltas |
| Success: AI leaderboard auto-updated | Capability-profile job re-scores models on new versions |

The classifier is the part that resembles NVIDIA. The rest is the routing layer that gives the classifier a purpose.

---

## 2. The classifier — architecture

We mirror NVIDIA's design so we can reuse their head structure and (optionally) distill from them.

**Backbone:** `microsoft/mdeberta-v3-base` (multilingual DeBERTa-v3).
- Same architecture family as NVIDIA's `deberta-v3-base`, so the multi-head pattern transfers 1:1.
- Covers Vietnamese **and** English **and** code tokens — important because real Viettel prompts are mixed (Vietnamese instructions with English code/technical terms).
- Rejected alternatives:
  - **PhoBERT** (`vinai/phobert-base-v2`): best pure-Vietnamese encoder, but needs word-segmentation preprocessing, caps at 256 tokens, and handles code/English poorly. Good as a secondary specialist, not the primary.
  - **XLM-RoBERTa-base**: solid multilingual fallback; we keep it as plan B if mDeBERTa-v3 finetuning is unstable.

**Heads:** one shared mean-pooled representation of the encoder's last hidden state → multiple heads (single forward pass, same as NVIDIA):
- `task_type`: 11-way classification head (softmax), report top-2 + probability.
- `creativity_scope`, `reasoning`, `contextual_knowledge`, `domain_knowledge`: regression heads in [0,1].
- `constraint_ct`, `number_of_few_shots`: regression heads, normalized by a `divisor_map` (mirror NVIDIA).
- Overall score uses NVIDIA's exact weights so outputs are comparable:
  ```
  score = 0.35*creativity + 0.25*reasoning + 0.15*constraints
        + 0.15*domain_knowledge + 0.05*contextual_knowledge + 0.05*few_shots
  ```

**Two model sizes (deliberate):**
- **`vi-router-quality`** — mDeBERTa-v3-base. Used for offline labeling/eval and for traffic where 50 ms isn't required.
- **`vi-router-fast`** — distilled student (`microsoft/Multilingual-MiniLM-L12-H384` or a 6-layer mDeBERTa), INT8-quantized, ONNX Runtime. This is what serves the 50 ms online path.

> **Decided (your feedback):** traffic is **mixed, ~70% Vietnamese / 30% EN+code** → mDeBERTa-v3 is the right call, PhoBERT is out as primary. Serving is **CPU-only if possible** → the distilled `vi-router-fast` is *mandatory*, not optional; the 50 ms budget on CPU is the binding constraint and drives an aggressive student (≤6 layers, H384, INT8). Task types / complexity dims stay at **NVIDIA parity** (no extra classes).

---

## 3. Data strategy (the real work)

NVIDIA's labels don't exist for Vietnamese, so we manufacture them. Three sources, combined:

### 3.1 Crawl / download Vietnamese corpora (raw prompts)
Candidate HuggingFace datasets (IDs to be verified at crawl time — `huggingface_hub` + `datasets`):
- **Instruction / chat (gives natural "prompt" text):**
  - `bkai-foundation-models/vi-alpaca`, `vilm/OpenOrca-Viet`, `5CD-AI/Vietnamese-*` instruction sets, ShareGPT-vi translations.
- **QA (Open/Closed QA, Extraction):** UIT-ViQuAD, UIT-ViNewsQA, `5CD-AI` QA sets.
- **Summarization:** `Yuhthe/vietnews`, VNDS, VietNews.
- **Classification / topic:** VNTC, UIT topic datasets.
- **General text (domain coverage):** `vietgpt/wikipedia_vi`, OSCAR-vi / CC100-vi (sampled), Binhvq news corpus.
- **Code prompts (Code Generation, mixed VN/EN):** translate a slice of an English code-instruction set (e.g. an Evol-Instruct / CodeAlpaca subset) into Vietnamese, or mine VN dev-forum questions.

Crawler lives in `src/data/crawl.py`: pulls, dedupes, language-filters (keep vi + mixed vi/en, drop pure-en/zh noise), strips to prompt text, caps length, writes to `data/raw/`.

### 3.2 Teacher labeling (turn raw prompts into NVIDIA-style labels)
Two teachers, cross-checked:
1. **NVIDIA model on machine-translated prompts** — translate VN→EN, run the original NVIDIA classifier, keep its task_type + complexity outputs as silver labels. Cheap, gives us NVIDIA-aligned distributions for free.
2. **Strong LLM labeler** (Qwen2.5/GPT-4-class) prompted with NVIDIA's exact rubric, labeling the **Vietnamese** text directly — catches what translation distorts (idioms, code, register).

Where the two teachers disagree beyond a threshold, route to a small **human-reviewed gold set** (a few thousand examples) used for validation/test, never training. `src/data/label_teacher.py`.

### 3.3 Synthetic generation (fill gaps + balance)
LLM-generate Vietnamese prompts deliberately spanning all 11 task types × low/med/high complexity, plus few-shot-heavy and constraint-heavy variants, so the training set isn't skewed toward whatever the crawl happened to contain. `src/data/synth_gen.py`.

**Final dataset:** `data/processed/` — train (silver, ~100k+), validation + test (human-gold, balanced across task types). Documented label schema and provenance per row.

> Note: the problem statement implies real Gateway logs exist. Once available, those become the highest-value source and replace synthetic data over time. The pipeline is built to ingest them.

---

## 4. Training pipeline

- `src/classifier/model.py` — multi-head `CustomModel`, config-driven (`target_sizes`, `task_type_map`, `weights_map`, `divisor_map`) so it's a clean parallel to NVIDIA's.
- Multi-task loss: cross-entropy (task_type) + MSE/SmoothL1 (complexity heads), weighted; class-balanced sampling for rare task types.
- Train `vi-router-quality` first → use it (+ teachers) to distill `vi-router-fast` (KD on logits + regression outputs).
- Export: `src/classifier/export_onnx.py` → ONNX + INT8 dynamic quantization; verify parity vs. PyTorch within tolerance.
- Tracking: metrics + configs logged; checkpoints versioned.

**Classifier metrics:** task_type macro-F1 + top-2 accuracy; complexity dims MAE/Spearman vs. gold; overall-score correlation vs. NVIDIA on a parallel EN set (sanity that we didn't drift from their rubric).

---

## 5. Router integration

- **Capability profiles** (`router/capability.py`): per model, a vector of skill scores by task_type, plus measured cost/1k-tokens and latency. Seeded from public benchmarks + a small internal eval suite; refreshed by a job when a model version changes → produces the **leaderboard** (success criterion #1).
- **Permission filter** (`router/policy.py`): drop models the requesting user/department can't access (success criterion: phân quyền).
- **Matching** (`router/match.py`): map `prompt_complexity_score` → minimum required capability tier for that task_type; among permitted models that clear the tier, pick **min cost** subject to a latency ceiling (tie-break on latency). Cheap simple prompts fall to small fast models → drives the ≥20% latency and ≥30% cost wins.
- **Serving** (`serving/api.py`): FastAPI endpoint `POST /route` → returns chosen model + the classifier's analysis + reason. Designed as a transparent shim so the Gateway needs no client changes.

**Latency budget (≤50 ms):** classifier inference (fast model, ONNX+INT8) + matching (pure in-memory lookup, microseconds). We measure end-to-end and tune (truncate token length, batch, optionally GPU) until p95 < 50 ms.

---

## 6. Evaluation against success criteria

Offline harness (`tests/eval/`) replays a held-out prompt set through three policies — (a) always-best-model, (b) always-cheap-model, (c) our router — and reports:
- **Cost delta** vs. always-best (target ≥30% reduction).
- **Quality delta** vs. always-best (target ≤3% drop) — judged by an LLM-judge / reference answers per task type.
- **Latency delta** on simple prompts (target ≥20% faster).
- **Routing overhead** p50/p95 (target ≤50 ms).

These four numbers are the report we hand over (success criterion: "báo cáo chứng minh hiệu quả").

---

## 7. Repo structure

```
ai-smart-routing/
  plan.md                  ← this file
  pyproject.toml
  data/{raw,processed,synthetic,gold}/
  src/
    data/{crawl,label_teacher,synth_gen,build_dataset}.py
    classifier/{model,tokenization,train,distill,infer,export_onnx}.py
    router/{capability,policy,match}.py
    serving/api.py
  configs/                 ← model + label-schema + weights/divisor maps
  tests/{unit,eval}/
  docs/                    ← usage guide + handover report
```

---

## 8. Milestones

1. **M1 — Data backbone:** crawler + teacher labeling + synthetic gen; first `data/processed` snapshot + gold set. *(highest risk, do first)*
2. **M2 — Quality classifier:** train `vi-router-quality`, hit task-F1 + complexity-MAE targets, sanity-check vs. NVIDIA on parallel EN.
3. **M3 — Fast classifier:** distill + ONNX + INT8; prove ≤50 ms p95.
4. **M4 — Router:** capability profiles + leaderboard + matching + permissions; FastAPI endpoint.
5. **M5 — Evaluation + handover:** run the cost/quality/latency harness, write the report and usage docs.

---

## 9. Key risks & calls I'm making for you to veto

- **Backbone = mDeBERTa-v3-base, not PhoBERT.** ✅ Confirmed by your "70% Vietnamese, mixed" answer — mixed VN/EN/code coverage is exactly what mDeBERTa handles and PhoBERT struggles with. Settled.
- **Silver labels from teachers, not human annotation at scale.** Fast and cheap; quality hinges on teacher agreement. Human gold only for val/test. Veto if you need certified labels.
- **CPU-only serving makes the distilled tiny model mandatory, not optional.** ✅ You chose CPU. On CPU, a base-size encoder cannot hit 50 ms, so `vi-router-fast` (6-layer distilled + INT8 ONNX) becomes *the* serving model, and we accept a small accuracy drop vs. the quality model. If 50 ms-on-CPU still misses after distillation, fallbacks are: shorter token cap (256), or relaxing the budget for the rare high-complexity prompts only.
- **External dataset IDs are candidates, verified at crawl time.** Some may be gated/renamed; the crawler is written to tolerate substitutions.
- **NVIDIA-as-teacher needs a VN→EN translation step**, which injects noise; the second LLM teacher labeling native Vietnamese is the hedge.

---

## 10. Feedback — answered

1. **Traffic mix?** → **Mixed, ~70% Vietnamese / 30% EN+code.** ✅ Decided: mDeBERTa-v3 backbone (not PhoBERT).
2. **CPU or GPU serving?** → **CPU if possible.** ✅ Decided: distilled+quantized `vi-router-fast` is mandatory; 50 ms-on-CPU is the binding constraint.
3. **Real Gateway logs?** → *You weren't sure what I meant.* **Clarification:** the problem statement says we integrate into an existing Viettel AI Gateway that's already serving traffic. If that Gateway keeps a history of past user prompts (the actual questions people typed), those real prompts are far better training data than crawled/synthetic ones, because they match what the model will see in production. **My assumption since none are available yet:** build entirely from crawled + synthetic data (Section 3), and design the pipeline so real logs can be dropped in later when/if you get access. No blocker.
4. **Candidate models + cost/latency?** → **Not sure yet.** **My assumption:** seed capability profiles from a public benchmark dataset (RouterBench — see §11) which already has 11 real models with measured cost + quality. We simulate against those now; swap in the real Viettel model list later via one config file. No blocker.
5. **Task types / complexity dims beyond NVIDIA?** → **Parity is fine.** ✅ We keep NVIDIA's 11 task types + 6 complexity dims exactly.

> Net: all five are resolved or safely assumed. Nothing blocks starting M1.

---

## 11. Simulation strategy — how we route without owning the big models

Your question: we don't have the actual large models sitting around, so how do we prove the router works? Answer: **the router never runs a model to make a decision — it only reads each model's profile (skill scores + cost + latency).** So we replace live models with two things: a static profile registry, and a table of *pre-recorded* answers. We don't hand-roll either — two existing open-source projects already provide exactly this, and we build on them.

### 11.1 Tool 1 — RouterBench (the cached-response dataset + harness)
[RouterBench](https://github.com/withmartian/routerbench) is a public benchmark of **~30,000 prompts with pre-computed responses from 11 real LLMs** (drawn from MMLU, GSM-8K, MBPP, MT-Bench, etc.). Every record carries: the prompt, each model's actual answer, that answer's **estimated cost**, and a **correctness/quality score**.

This solves the "we don't have the models" problem outright:
- **Cost & latency** of any routing decision = arithmetic over RouterBench's cost column. Zero models needed.
- **Quality** of a decision = look up the chosen model's pre-scored answer for that prompt. Also zero live calls.
- The 11 models with measured cost+quality become our **seed capability profiles** (answers feedback question #4 — we have real numbers today, swap in the Viettel model list later).

We replay our router over RouterBench and read off the four success metrics directly. This is the spine of the eval harness in §6.

### 11.2 Tool 2 — RouteLLM (the router framework + cost/quality calibration)
[RouteLLM](https://github.com/lm-sys/RouteLLM) is a pip-installable framework (`pip install "routellm[serve,eval]"`) for serving and evaluating routers that direct between a strong/expensive and weak/cheap model. It gives us, off the shelf:
- A **threshold-calibration** mechanism: "route X% of traffic to the strong model" → solves for the threshold. This is exactly the knob behind our ≥30%-cost / ≤3%-quality tradeoff — we tune one number and read the resulting cost/quality curve.
- **Cached benchmark judgements** (MT-Bench, MMLU, GSM8K precomputed) — same offline philosophy as RouterBench, no live calls.
- An **extensible router interface**: we register our Vietnamese classifier as a custom RouteLLM router (implement its scoring hook) instead of writing serving/calibration/eval plumbing ourselves.

### 11.3 How they fit our design
- Our **classifier** (§2) produces `(task_type, complexity_score)`. We register it as a **custom RouteLLM router** — its complexity score is the routing signal RouteLLM calibrates a threshold against.
- **RouterBench** supplies the model registry (cost/quality/latency) and the prompt set we replay over. Its 11 models seed `router/capability.py`; the real Viettel models drop into the same schema later.
- **RouteLLM** supplies calibration + the offline eval loop; our `tests/eval/` harness (§6) wraps it to emit the four handover numbers.
- Caveat: RouterBench is **English**. For Vietnamese cost/quality we extend it cheaply — take a few hundred VN prompts, call public model APIs **once** (a few dollars, one time), cache every answer + cost + an LLM-judge score to disk in RouterBench's schema, and treat that file as a permanent fixture. From then on the VN simulation is also 100% offline.

### 11.4 What this means concretely
- **Cost win (≥30%) and latency win (≥20%) are pure offline arithmetic** over cached cost/latency columns — provable today, no models, no GPU.
- **Quality bound (≤3% drop)** uses RouterBench's pre-scored answers (EN) + the small one-time VN cache (VN). No live inference in the loop.
- We write **no** routing/calibration/eval framework from scratch — we configure RouteLLM and feed it RouterBench-format data. New code is just the classifier, the VN data pipeline, and a thin adapter.

### 11.5 New deliverables this adds
```
src/sim/
  routerbench_adapter.py   ← load RouterBench → our capability-profile + prompt-replay format
  routellm_router.py       ← register vi-router classifier as a custom RouteLLM router
  vi_response_cache.py     ← one-time: call public APIs on VN prompts, cache answers+cost+score
tests/eval/simulate.py     ← run RouteLLM calibration over the data, emit the 4 success metrics
configs/sim_models.yaml    ← model registry (RouterBench seed now; Viettel models later)
```

> Bottom line: we don't simulate the models — we reuse a dataset where 11 real models' answers, costs, and scores are already recorded, and a framework that already does cost/quality calibration. The only fresh spend is a one-time few-dollar API run to add Vietnamese coverage.

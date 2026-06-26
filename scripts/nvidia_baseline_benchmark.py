#!/usr/bin/env python
"""Benchmark nvidia/prompt-task-and-complexity-classifier on CPU vs our teacher.

Reconstructs the NVIDIA model from its safetensors weights + config.json
(no NeMo Curator dependency needed). Runs the same 3 Vietnamese prompts
(short/medium/long) with 10 warmup + 30 timed iterations each, then compares:
  - p50/p95/mean latency per category vs our teacher (mDeBERTa, prior run)
  - task_type prediction alignment with our val set ground-truth labels

Usage (from repo root):
    PYTHONPATH=src python scripts/nvidia_baseline_benchmark.py
    PYTHONPATH=src python scripts/nvidia_baseline_benchmark.py --val-samples 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompts (same 3 categories as teacher benchmark)
# ---------------------------------------------------------------------------

PROMPTS = {
    "short":  "Việt Nam ở đâu?",
    "medium": (
        "Hãy giải thích sự khác biệt giữa học máy và học sâu. "
        "Cho ví dụ cụ thể về ứng dụng của từng loại trong thực tế."
    ),
    "long": (
        "Bối cảnh: Công ty chúng tôi đang phát triển một hệ thống AI tư vấn tài chính "
        "cho thị trường Việt Nam. Hệ thống cần phân tích danh mục đầu tư của khách hàng, "
        "đưa ra khuyến nghị dựa trên mục tiêu tài chính cá nhân, mức độ chấp nhận rủi ro, "
        "và điều kiện thị trường hiện tại. Khách hàng mục tiêu là những người có thu nhập "
        "trung bình đến cao, độ tuổi 30-55, muốn tối ưu hóa danh mục đầu tư dài hạn.\n\n"
        "Nhiệm vụ: Hãy thiết kế kiến trúc tổng thể cho hệ thống AI này, bao gồm: "
        "(1) Các module chính và chức năng của từng module, "
        "(2) Luồng dữ liệu từ đầu vào đến đầu ra, "
        "(3) Các thách thức kỹ thuật đặc thù cho thị trường tài chính Việt Nam, "
        "(4) Phương pháp đánh giá và kiểm thử hệ thống, "
        "(5) Các vấn đề tuân thủ pháp lý cần lưu ý theo quy định của UBCKNN và NHNN."
    ),
}

WARMUP = 10
TIMED  = 30

NVIDIA_MODEL = "nvidia/prompt-task-and-complexity-classifier"

# Teacher CPU benchmark results from prior run (mDeBERTa-v3-base, fp32)
TEACHER_REF = {
    "short":  {"p50_ms": 73.4,  "p95_ms": 84.2},
    "medium": {"p50_ms": 101.2, "p95_ms": 105.3},
    "long":   {"p50_ms": 349.1, "p95_ms": 430.4},
}


# ---------------------------------------------------------------------------
# NVIDIA model reconstruction (no NeMo Curator needed)
# ---------------------------------------------------------------------------

def build_nvidia_model(cfg: dict, state_dict: dict):
    """Reconstruct the NVIDIA multi-head DeBERTa classifier from weights."""
    import torch
    import torch.nn as nn
    from transformers import DebertaV2Model

    class _Head(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.fc = nn.Linear(in_f, out_f)

        def forward(self, x):
            return self.fc(x)

    class NvidiaClassifier(nn.Module):
        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg
            print(f"  Loading DeBERTa backbone ({cfg['base_model']}) ...", flush=True)
            self.backbone = DebertaV2Model.from_pretrained(
                cfg["base_model"], torch_dtype=torch.float32
            )
            hidden = self.backbone.config.hidden_size  # 768
            self._target_sizes = cfg["target_sizes"]   # ordered dict
            self.heads = nn.ModuleList([
                _Head(hidden, sz)
                for sz in self._target_sizes.values()
            ])
            self._target_names = list(self._target_sizes.keys())

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token pooling
            cls = out.last_hidden_state[:, 0, :].float()
            return {name: head(cls) for name, head in zip(self._target_names, self.heads)}

    # Remap state dict keys: head_N.fc.* -> heads.N.fc.*
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("head_") and not k.startswith("head_s"):
            parts = k.split(".", 1)
            head_num = parts[0].replace("head_", "")
            remapped[f"heads.{head_num}.{parts[1]}"] = v.float()
        else:
            remapped[k] = v.float()

    model = NvidiaClassifier(cfg)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys: {missing[:3]} ...")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys: {unexpected[:3]} ...")
    model.float().eval()
    return model


def _argmax_score(logits_tensor) -> tuple[int, float]:
    """Softmax argmax → (class_idx, probability)."""
    import torch
    probs = torch.softmax(logits_tensor, dim=-1)
    idx = int(probs.argmax().item())
    return idx, float(probs[idx].item())


def decode_output(logits: dict, cfg: dict) -> dict:
    """Convert raw logits to NVIDIA-style output dict."""
    import torch

    task_map   = cfg["task_type_map"]          # "0"→"Brainstorming" etc.
    weights_m  = cfg["weights_map"]
    divisor_m  = cfg["divisor_map"]

    # task_type (softmax)
    tt_logits = logits["task_type"]
    tt_probs  = torch.softmax(tt_logits, dim=-1)
    tt_order  = tt_probs.argsort(descending=True)
    idx1 = int(tt_order[0].item())
    idx2 = int(tt_order[1].item()) if len(tt_order) > 1 else idx1
    task_type_1 = task_map[str(idx1)]
    task_type_2 = task_map[str(idx2)]
    task_type_prob = float(tt_probs[idx1].item())

    def dim_score(name: str) -> float:
        d_logits = logits[name]
        idx = int(d_logits.argmax().item())
        raw = weights_m[name][idx]
        return raw / divisor_m[name]

    creativity_scope      = dim_score("creativity_scope")
    reasoning             = dim_score("reasoning")
    contextual_knowledge  = dim_score("contextual_knowledge")
    number_of_few_shots   = dim_score("number_of_few_shots")
    domain_knowledge      = dim_score("domain_knowledge")
    constraint_ct         = dim_score("constraint_ct")

    # Complexity score per README formula
    prompt_complexity_score = (
        0.35 * creativity_scope
        + 0.25 * reasoning
        + 0.15 * constraint_ct
        + 0.15 * domain_knowledge
        + 0.05 * contextual_knowledge
        + 0.05 * number_of_few_shots
    )

    return {
        "task_type_1":            task_type_1,
        "task_type_2":            task_type_2,
        "task_type_prob":         task_type_prob,
        "creativity_scope":       creativity_scope,
        "reasoning":              reasoning,
        "contextual_knowledge":   contextual_knowledge,
        "number_of_few_shots":    number_of_few_shots,
        "domain_knowledge":       domain_knowledge,
        "constraint_ct":          constraint_ct,
        "prompt_complexity_score": prompt_complexity_score,
    }


def load_nvidia():
    """Download weights + config, build and return (tokenizer, model, cfg)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    print(f"[nvidia] Fetching {NVIDIA_MODEL} ...", flush=True)
    t0 = time.time()
    cfg_path    = hf_hub_download(NVIDIA_MODEL, "config.json")
    model_path  = hf_hub_download(NVIDIA_MODEL, "model.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)
    state_dict = load_file(model_path)
    print(f"  weights fetched in {time.time()-t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(NVIDIA_MODEL)
    model     = build_nvidia_model(cfg, state_dict)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, eval mode", flush=True)
    return tokenizer, model, cfg


# ---------------------------------------------------------------------------
# Single-prompt inference
# ---------------------------------------------------------------------------

def infer(tokenizer, model, cfg, text: str) -> dict:
    import torch
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        raw_logits = model(enc["input_ids"], enc["attention_mask"])
    # squeeze batch dim
    squeezed = {k: v.squeeze(0) for k, v in raw_logits.items()}
    return decode_output(squeezed, cfg)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(tokenizer, model, cfg, label: str, text: str) -> dict:
    print(f"\n[bench] {label} ({len(text)} chars) — {WARMUP} warmup + {TIMED} timed ...", flush=True)
    for _ in range(WARMUP):
        infer(tokenizer, model, cfg, text)

    times_ms: list[float] = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        infer(tokenizer, model, cfg, text)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    n  = len(times_ms)
    p50  = times_ms[n // 2]
    p95  = times_ms[int(n * 0.95)]
    mean = statistics.mean(times_ms)
    print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  mean={mean:.1f}ms", flush=True)
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean, "sla_pass": p95 <= 50.0}


# ---------------------------------------------------------------------------
# Val-set alignment
# ---------------------------------------------------------------------------

def val_alignment(tokenizer, model, cfg, val_path: Path, n: int = 100) -> dict:
    examples = []
    with val_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append((row["text"], row["task_type"]))
            if len(examples) >= n:
                break

    print(f"\n[align] Running NVIDIA on {len(examples)} val examples ...", flush=True)
    matches = 0
    results = []
    for i, (text, gold) in enumerate(examples):
        pred_dict = infer(tokenizer, model, cfg, text)
        pred = pred_dict["task_type_1"]
        hit  = (pred == gold)
        if hit:
            matches += 1
        results.append({"gold": gold, "pred": pred, "prob": pred_dict["task_type_prob"], "match": hit})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(examples)} done ...", flush=True)

    acc = matches / len(examples) if examples else 0.0
    print(f"  accuracy={acc:.1%}  ({matches}/{len(examples)} exact matches)", flush=True)

    from collections import defaultdict
    per_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        per_class[r["gold"]]["total"] += 1
        if r["match"]:
            per_class[r["gold"]]["correct"] += 1

    class_acc = {}
    print("  Per-class accuracy:")
    for cls, vals in sorted(per_class.items()):
        a = vals["correct"] / vals["total"] if vals["total"] else 0.0
        class_acc[cls] = a
        print(f"    {cls:<22} {a:.1%}  ({vals['correct']}/{vals['total']})")

    return {"accuracy": acc, "n": len(examples), "per_class": class_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NVIDIA baseline benchmark")
    ap.add_argument("--val-samples", type=int, default=100)
    ap.add_argument("--out", default="runs/nvidia_baseline.json")
    args = ap.parse_args()

    try:
        tokenizer, model, cfg = load_nvidia()
    except Exception as e:
        print(f"[ERROR] Could not load NVIDIA model: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # --- Latency ---
    latency = {}
    for label, text in PROMPTS.items():
        latency[label] = benchmark_latency(tokenizer, model, cfg, label, text)

    # --- Sample predictions ---
    print("\n[sample] NVIDIA predictions for benchmark prompts:")
    for label, text in PROMPTS.items():
        result = infer(tokenizer, model, cfg, text)
        print(f"  {label:6s}: task_type={result['task_type_1']!r:20s} "
              f"(p={result['task_type_prob']:.3f})  "
              f"complexity={result['prompt_complexity_score']:.3f}")

    # --- Val alignment ---
    repo_root = Path(__file__).resolve().parent.parent
    val_path  = repo_root / "data" / "processed" / "val.jsonl"
    alignment = {}
    if val_path.exists():
        alignment = val_alignment(tokenizer, model, cfg, val_path, n=args.val_samples)
    else:
        print(f"\n[align] val.jsonl not found at {val_path}, skipping")

    # --- Summary table ---
    print()
    print("=" * 80)
    print("LATENCY COMPARISON: NVIDIA vs Our Teacher (microsoft/mdeberta-v3-base, CPU)")
    print("=" * 80)
    print(f"{'Prompt':<8}  {'NVIDIA p50':>10} {'NVIDIA p95':>10} {'Teacher p50':>12} {'Teacher p95':>12}  {'NVIDIA SLA'}")
    print("-" * 80)
    for cat in ["short", "medium", "long"]:
        nv = latency[cat]
        tr = TEACHER_REF[cat]
        sla = "PASS ✓" if nv["sla_pass"] else "FAIL ✗"
        print(f"  {cat:<6}  {nv['p50_ms']:>9.1f}ms {nv['p95_ms']:>9.1f}ms "
              f"{tr['p50_ms']:>11.1f}ms {tr['p95_ms']:>11.1f}ms   {sla}")
    print("=" * 80)

    if alignment:
        print(f"\nVal alignment: NVIDIA task_type_1 vs gold labels → "
              f"{alignment['accuracy']:.1%} accuracy over {alignment['n']} examples")
        print("(Our training labels came from this model — high alignment = consistent data)")

    # Write JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": NVIDIA_MODEL,
        "latency": latency,
        "alignment": alignment,
        "teacher_reference_cpu": TEACHER_REF,
    }, indent=2, ensure_ascii=False))
    print(f"\n[done] Results → {out}")


if __name__ == "__main__":
    main()

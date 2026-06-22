"""Knowledge distillation: vi-router-quality (teacher) -> vi-router-fast (student).

The student is a small multilingual encoder (MiniLM-L12-H384). It learns from
both the gold labels and the teacher's soft outputs (plan §4):

  - task_type: KL divergence between student/teacher softened logits (temperature T)
  - complexity dims: MSE between student/teacher scalar outputs

This is what makes the CPU <=50ms target reachable (plan §2, §9): the student is
~6x smaller and gets INT8-quantized at ONNX export.

    uv run --extra ml python -m classifier.distill \
        --teacher runs/quality --out runs/fast --data data/processed
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classifier.model import CustomModel, ModelSpec
from classifier.tokenization import PromptTokenizer
from classifier.train import PromptDataset, make_collate
from config import load_complexity, load_label_schema, load_model_configs


def _load_teacher(teacher_dir: Path, device: str) -> tuple[CustomModel, dict[str, Any]]:
    meta = json.loads((teacher_dir / "meta.json").read_text())
    schema = load_label_schema()
    complexity = load_complexity()
    cfg = load_model_configs()[meta["model_name"]]
    spec = ModelSpec.from_config(cfg)
    model = CustomModel(spec, schema, complexity, pretrained=False)
    model.load_state_dict(torch.load(teacher_dir / "model.pt", map_location=device))
    model.to(device).eval()
    return model, meta


def distill(
    teacher_dir: str | Path,
    out_dir: str | Path,
    data_dir: str | Path,
    *,
    student_name: str = "vi-router-fast",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    temperature: float = 2.0,
    alpha: float = 0.5,  # weight on distillation vs. hard-label loss
    device: str | None = None,
    pretrained: bool = True,
    max_steps: int | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher_dir = Path(teacher_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_label_schema()
    complexity = load_complexity()

    teacher, _ = _load_teacher(teacher_dir, device)
    teacher_tok = PromptTokenizer(teacher.spec.backbone, teacher.spec.max_tokens)

    student_cfg = load_model_configs()[student_name]
    student_spec = ModelSpec.from_config(student_cfg)
    student = CustomModel(student_spec, schema, complexity, pretrained=pretrained).to(device)
    student_tok = PromptTokenizer(student_spec.backbone, student_spec.max_tokens)

    ds = PromptDataset(Path(data_dir) / "train.jsonl", schema)
    # collate per-tokenizer: we need both teacher and student encodings of the same text
    student_collate = make_collate(student_tok, schema)

    def dual_collate(batch):
        out = student_collate(batch)
        t_enc = teacher_tok([e.text for e in batch])
        out["t_input_ids"] = t_enc["input_ids"]
        out["t_attention_mask"] = t_enc["attention_mask"]
        return out

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=dual_collate)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    hard_ce = nn.CrossEntropyLoss()
    reg = nn.SmoothL1Loss()

    history: list[dict[str, Any]] = []
    step = 0
    student.train()
    for epoch in range(epochs):
        for batch in dl:
            s_ids = batch["input_ids"].to(device)
            s_mask = batch["attention_mask"].to(device)
            t_ids = batch["t_input_ids"].to(device)
            t_mask = batch["t_attention_mask"].to(device)
            task_idx = batch["task_idx"].to(device)
            dim_targets = {k: v.to(device) for k, v in batch["dim_targets"].items()}

            with torch.no_grad():
                t_out = teacher(t_ids, t_mask)

            s_out = student(s_ids, s_mask)

            # --- task_type: KD (KL on softened logits) + hard CE ---
            t_soft = F.log_softmax(t_out["task_type"] / temperature, dim=-1)
            s_soft = F.log_softmax(s_out["task_type"] / temperature, dim=-1)
            kd_task = F.kl_div(
                s_soft, t_soft, reduction="batchmean", log_target=True
            ) * (temperature**2)
            hard_task = hard_ce(s_out["task_type"], task_idx)

            # --- complexity dims: match teacher outputs + gold ---
            kd_reg = s_out["task_type"].new_zeros(())
            hard_reg = s_out["task_type"].new_zeros(())
            for dim in schema.complexity_dimensions:
                kd_reg = kd_reg + reg(s_out[dim], t_out[dim])
                hard_reg = hard_reg + reg(s_out[dim], dim_targets[dim])

            loss = (
                alpha * (kd_task + kd_reg) + (1 - alpha) * (hard_task + hard_reg)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            if step % 20 == 0:
                history.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "kd_task": float(kd_task.item()),
                        "hard_task": float(hard_task.item()),
                    }
                )
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    torch.save(student.state_dict(), out_dir / "model.pt")
    student_tok.save(str(out_dir / "tokenizer"))
    meta = {
        "model_name": student_name,
        "backbone": student_spec.backbone,
        "max_tokens": student_spec.max_tokens,
        "distilled_from": str(teacher_dir),
        "temperature": temperature,
        "alpha": alpha,
        "steps": step,
        "final_loss": history[-1]["loss"] if history else math.nan,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill quality model into fast student")
    ap.add_argument("--teacher", default="runs/quality")
    ap.add_argument("--out", default="runs/fast")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--student", default="vi-router-fast")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    meta = distill(
        args.teacher,
        args.out,
        args.data,
        student_name=args.student,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        max_steps=args.max_steps,
        pretrained=not args.no_pretrained,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

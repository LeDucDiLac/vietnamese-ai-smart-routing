"""Knowledge distillation: vi-router-quality (teacher) -> vi-router-fast (student).

The student is a small multilingual encoder (MiniLM-L12-H384). It learns from
both the gold labels and the teacher's soft outputs (plan §4):

  - task_type: KL divergence between student/teacher softened logits (temperature T)
  - complexity dims: MSE between student/teacher scalar outputs

This is what makes the CPU <=50ms target reachable (plan §2, §9): the student is
~6x smaller and gets INT8-quantized at ONNX export.

The teacher may be any pre-trained checkpoint (e.g. a v2 winner trained on the
H200); ``--teacher`` just points at its run dir (model.pt + meta.json + tokenizer).

    uv run --extra ml python -m classifier.distill \
        --teacher runs/teachers/vi-router-quality-granite \
        --out runs/students/vi-router-fast-granite \
        --student vi-router-fast-granite --data data/processed
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classifier.model import CustomModel, ModelSpec
from classifier.tokenization import PromptTokenizer
from classifier.train import PromptDataset, evaluate, make_collate
from config import LabelSchema, load_complexity, load_label_schema, load_model_configs


def _load_schema(schema_version: str | None):
    """Load the (schema, complexity) pair for a given version (v2 → configs/schemas/v2*).

    Teacher, student, dataset and eval must all use the SAME schema — the v2
    teachers have a 6-class head + 3 complexity dims, incompatible with the v1
    default (11 classes + 6 dims).
    """
    from config import CONFIGS_DIR

    schema = load_label_schema(version=schema_version)
    complexity_path = (
        str(CONFIGS_DIR / "schemas" / f"{schema_version}-complexity.yaml")
        if schema_version
        else None
    )
    complexity = load_complexity(path=complexity_path)
    return schema, complexity


def _load_teacher(
    teacher_dir: Path, device: str, schema: LabelSchema, complexity
) -> tuple[CustomModel, dict[str, Any]]:
    meta = json.loads((teacher_dir / "meta.json").read_text())
    cfg = load_model_configs()[meta["model_name"]]
    spec = ModelSpec.from_config(cfg)
    model = CustomModel(spec, schema, complexity, pretrained=False)
    model.load_state_dict(torch.load(teacher_dir / "model.pt", map_location=device))
    model.float().to(device).eval()
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
    schema_version: str | None = None,
    mlflow_experiment: str = "vi-smart-routing",
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher_dir = Path(teacher_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _MLFLOW:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        _mlflow_run = mlflow.start_run(run_name=f"{student_name}-distill")
    else:
        _mlflow_run = None

    schema, complexity = _load_schema(schema_version)

    teacher, teacher_meta = _load_teacher(teacher_dir, device, schema, complexity)
    teacher_tok = PromptTokenizer(teacher.spec.backbone, teacher.spec.max_tokens)

    student_cfg = load_model_configs()[student_name]
    student_spec = ModelSpec.from_config(student_cfg)
    # Cast student to fp32 before AMP training (GradScaler fails on fp16 grads).
    student = CustomModel(student_spec, schema, complexity, pretrained=pretrained).float().to(device)
    student_tok = PromptTokenizer(student_spec.backbone, student_spec.max_tokens)

    data_dir = Path(data_dir)
    ds = PromptDataset(data_dir / "train.jsonl", schema)
    # collate per-tokenizer: we need both teacher and student encodings of the same text
    student_collate = make_collate(student_tok, schema)

    def dual_collate(batch):
        out = student_collate(batch)
        t_enc = teacher_tok([e.text for e in batch])
        out["t_input_ids"] = t_enc["input_ids"]
        out["t_attention_mask"] = t_enc["attention_mask"]
        return out

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=dual_collate)

    # Student-tokenized val/test loaders for parity metrics (reuse train.evaluate).
    val_path = data_dir / "val.jsonl"
    val_dl: DataLoader | None = None
    if val_path.exists():
        val_ds = PromptDataset(val_path, schema)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=student_collate)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    hard_ce = nn.CrossEntropyLoss()
    reg = nn.SmoothL1Loss()

    use_amp = torch.cuda.is_available() and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if _MLFLOW and _mlflow_run:
        mlflow.log_params({
            "student_name": student_name,
            "student_backbone": student_spec.backbone,
            "teacher_dir": str(teacher_dir),
            "teacher_backbone": teacher.spec.backbone,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": temperature,
            "alpha": alpha,
            "schema_version": schema_version or "default",
            "device": device,
            "train_size": len(ds),
        })

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

            with torch.amp.autocast("cuda", enabled=use_amp):
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
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

        # Per-epoch student parity metrics on val (macro-F1, top-1/2, MAE, R², ρ).
        if val_dl is not None:
            ep_metrics = evaluate(student, val_dl, schema, device)
            student.train()
            print(
                f"[distill] epoch {epoch + 1}/{epochs} | val_f1={ep_metrics['task_macro_f1']:.4f} "
                f"val_acc={ep_metrics['task_top1_acc']:.4f} mae={ep_metrics['complexity_mae']:.4f}",
                flush=True,
            )
            if _MLFLOW and _mlflow_run:
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in ep_metrics.items()}, step=epoch + 1
                )

        if max_steps is not None and step >= max_steps:
            break

    # Final parity metrics: val + test (vs the teacher's own meta for comparison).
    val_metrics: dict[str, Any] = {}
    if val_dl is not None:
        val_metrics = evaluate(student, val_dl, schema, device)

    test_metrics: dict[str, Any] = {}
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = PromptDataset(test_path, schema)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=student_collate)
        test_metrics = {f"test_{k}": v for k, v in evaluate(student, test_dl, schema, device).items()}

    torch.save(student.state_dict(), out_dir / "model.pt")
    student_tok.save(str(out_dir / "tokenizer"))
    meta = {
        "model_name": student_name,
        "backbone": student_spec.backbone,
        "max_tokens": student_spec.max_tokens,
        "schema_version": schema_version or "default",
        "distilled_from": str(teacher_dir),
        "teacher_backbone": teacher.spec.backbone,
        "temperature": temperature,
        "alpha": alpha,
        "steps": step,
        "final_loss": history[-1]["loss"] if history else math.nan,
        **val_metrics,
        **test_metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history), encoding="utf-8"
    )

    if _MLFLOW and _mlflow_run:
        final = {
            "final_val_f1": val_metrics.get("task_macro_f1", math.nan),
            "final_val_acc": val_metrics.get("task_top1_acc", math.nan),
            "final_val_top2_acc": val_metrics.get("task_top2_acc", math.nan),
            "final_val_complexity_mae": val_metrics.get("complexity_mae", math.nan),
            "test_f1": test_metrics.get("test_task_macro_f1", math.nan),
            "test_acc": test_metrics.get("test_task_top1_acc", math.nan),
            "test_complexity_mae": test_metrics.get("test_complexity_mae", math.nan),
            "final_train_loss": meta["final_loss"],
        }
        mlflow.log_metrics({k: v for k, v in final.items() if not math.isnan(v)})
        mlflow.log_artifact(str(out_dir / "meta.json"))
        mlflow.end_run()

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
    ap.add_argument("--schema-version", default=None,
                    help="label schema version, e.g. 'v2' → configs/schemas/v2.yaml. "
                         "Must match the teacher's schema (v2 teachers have 6-class heads).")
    ap.add_argument("--mlflow-experiment", default="vi-smart-routing")
    ap.add_argument("--mlflow-tracking-uri", default=None)
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
        schema_version=args.schema_version,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

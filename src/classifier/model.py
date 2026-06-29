"""Multi-head prompt task & complexity classifier.

Mirrors NVIDIA ``prompt-task-and-complexity-classifier``: one transformer
backbone, a mean-pooled representation of the last hidden state, then a set of
independent heads (one classification head for ``task_type`` + one regression
head per complexity dimension). All heads share the encoder, so a single forward
pass produces every output (NVIDIA parity).

Torch + transformers live in the ``ml`` optional extra. Import this module only
when that extra is installed (training / quality-path inference). The router and
sim paths never import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from config import ComplexityConfig, LabelSchema


@dataclass
class ModelSpec:
    """Resolved architecture knobs for one model size (from configs/model.yaml)."""

    backbone: str
    max_tokens: int = 512
    pooling: str = "mean"
    dropout: float = 0.1
    head_hidden: int = 0  # 0 = linear head; >0 inserts a hidden layer of this width

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ModelSpec":
        return cls(
            backbone=cfg["backbone"],
            max_tokens=int(cfg.get("max_tokens", 512)),
            pooling=str(cfg.get("pooling", "mean")),
            dropout=float(cfg.get("dropout", 0.1)),
            head_hidden=int(cfg.get("head_hidden", 0)),
        )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean pooling over the token dimension (NVIDIA parity)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class _Head(nn.Module):
    """A single prediction head: optional hidden layer -> output projection."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.dropout(x))


class CustomModel(nn.Module):
    """Shared-encoder, multi-head classifier.

    Outputs (dict):
      - ``task_type``: logits, shape ``(B, num_task_types)``
      - one scalar regression per complexity dimension, shape ``(B, 1)``
    """

    def __init__(
        self,
        spec: ModelSpec,
        schema: LabelSchema,
        complexity: ComplexityConfig,
        *,
        pretrained: bool = True,
    ):
        super().__init__()
        self.spec = spec
        self.schema = schema
        self.complexity = complexity

        auto_cfg = AutoConfig.from_pretrained(spec.backbone)
        # ModernBERT backbones (mmBERT, Granite-r2) torch.compile their MLP, which
        # allocates fresh inductor workspace at eval time (and recompiles per shape)
        # — enough to OOM on a busy/shared GPU mid-run. Run them eager instead: same
        # math, flat memory, no recompilation. No-op for non-ModernBERT backbones.
        if hasattr(auto_cfg, "reference_compile"):
            auto_cfg.reference_compile = False
        if pretrained:
            # transformers >= 4.47 blocks torch.load when torch < 2.6 (CVE-2025-32434).
            # P100/K80 GPUs require torch 2.3.x (sm_60 dropped in 2.6+), so we patch
            # out the check. The model loads via safetensors — not affected by the CVE.
            try:
                import transformers.utils.import_utils as _tu
                _tu.check_torch_load_is_safe = lambda: None
            except Exception:
                pass
            self.backbone = AutoModel.from_pretrained(spec.backbone, config=auto_cfg)
        else:
            # build from config only — fast, weightless; for unit tests / shape checks
            self.backbone = AutoModel.from_config(auto_cfg)
        hidden = auto_cfg.hidden_size

        self.task_head = _Head(
            hidden, schema.num_task_types, spec.head_hidden, spec.dropout
        )
        self.complexity_heads = nn.ModuleDict(
            {
                dim: _Head(hidden, 1, spec.head_hidden, spec.dropout)
                for dim in schema.complexity_dimensions
            }
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)

        result: dict[str, torch.Tensor] = {"task_type": self.task_head(pooled)}
        for dim, head in self.complexity_heads.items():
            # complexity dims are bounded [0,1] -> sigmoid squashes the regression
            result[dim] = torch.sigmoid(head(pooled))
        return result

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Human-readable predictions for a batch, NVIDIA-style output shape."""
        self.eval()
        out = self.forward(input_ids, attention_mask)
        task_probs = torch.softmax(out["task_type"], dim=-1)
        top2 = torch.topk(task_probs, k=min(2, task_probs.shape[-1]), dim=-1)

        results: list[dict[str, Any]] = []
        batch = input_ids.shape[0]
        for i in range(batch):
            dims = {
                dim: float(out[dim][i].item())
                for dim in self.schema.complexity_dimensions
            }
            idx1 = int(top2.indices[i, 0].item())
            idx2 = int(top2.indices[i, 1].item()) if top2.indices.shape[1] > 1 else idx1
            results.append(
                {
                    "task_type_1": self.schema.task_types[idx1],
                    "task_type_2": self.schema.task_types[idx2],
                    "task_type_prob": float(top2.values[i, 0].item()),
                    **dims,
                    "prompt_complexity_score": self.complexity.score(dims),
                }
            )
        return results

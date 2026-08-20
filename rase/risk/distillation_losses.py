"""Distillation losses for LightRiskStudent training.

Three types of teacher targets:
1. probability_distill: teacher soft success probability (KL/BCE)
2. evidence_distill: teacher future-embedding alignment (cosine/MSE)
3. uncertainty_distill: teacher OOD/uncertainty alignment
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def probability_distill_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    hard_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft KL-divergence distillation with optional hard-label mixing.

    Args:
        student_logits: (N,) raw logits.
        teacher_probs: (N,) teacher soft probabilities in [0,1].
        temperature: distillation temperature.
        alpha: weight of hard BCE (0=soft only, 1=hard only).
        hard_labels: (N,) binary labels.  If None, alpha=0.
    """
    # Soft target: KL-divergence
    teacher_logits = torch.logit(
        teacher_probs.clamp(1e-7, 1.0 - 1e-7)
    )
    soft_loss = F.kl_div(
        F.logsigmoid(student_logits / temperature),
        torch.sigmoid(teacher_logits / temperature),
        reduction="batchmean",
    ) * (temperature ** 2)

    if hard_labels is not None and alpha > 0:
        hard_loss = F.binary_cross_entropy_with_logits(
            student_logits, hard_labels,
            pos_weight=torch.tensor([(hard_labels.numel() - hard_labels.sum()) / hard_labels.sum().clamp_min(1)])
        )
        return (1.0 - alpha) * soft_loss + alpha * hard_loss
    return soft_loss


def evidence_distill_loss(
    student_embed: torch.Tensor,
    teacher_embed: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity + SmoothL1 loss for teacher evidence alignment.

    Args:
        student_embed: (N, D) student DistillProjection output.
        teacher_embed: (N, D) teacher pooled evidence vector.
    """
    cosine = 1.0 - F.cosine_similarity(student_embed, teacher_embed, dim=-1)
    l1 = F.smooth_l1_loss(student_embed, teacher_embed)
    return cosine.mean() + l1


def quantile_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Pinball / quantile regression loss.

    Args:
        pred_quantiles: (N, Q) predicted quantile values.
        target: (N,) true scalar value.
        quantiles: (Q,) quantile levels in [0,1].
    """
    Q = pred_quantiles.shape[-1]
    target = target.unsqueeze(-1).expand(-1, Q)
    errors = target - pred_quantiles
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()

"""Lightweight multi-head model for probabilistic RASE handback control."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rase.risk.canonical_action import CanonicalActionChunk, summary_from_chunk


class Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ProbabilisticHandbackStudent(nn.Module):
    """Shared state/action encoder with probability and cost heads."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        action_dim: int = 20,
        history_dim: int = 64,
        fused_dim: int = 128,
        head_hidden: int = 128,
        n_cost_quantiles: int = 3,
        dropout: float = 0.1,
        minimum_concentration: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.minimum_concentration = float(minimum_concentration)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.history_gru = nn.GRU(6, history_dim, batch_first=True)
        base_dim = encoder.output_dim + 64 * 2 + history_dim
        self.fusion = nn.Sequential(
            nn.Linear(base_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
        )
        self.handback_head = Head(fused_dim, head_hidden, 2, dropout)
        self.persistent_head = Head(fused_dim, head_hidden, 1, dropout)
        self.source_risk_head = Head(fused_dim, head_hidden, 1, dropout)
        self.cost_head = Head(fused_dim, head_hidden, n_cost_quantiles, dropout)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        student_action: CanonicalActionChunk | torch.Tensor,
        oft_action: CanonicalActionChunk | torch.Tensor,
        history: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        state = self.encoder(image, proprio, text_embed)
        student = self.action_mlp(_action_summary(student_action, batch_size))
        oft = self.action_mlp(_action_summary(oft_action, batch_size))
        history_out, _ = self.history_gru(history)
        fused = self.fusion(torch.cat([state, student, oft, history_out[:, -1]], dim=-1))
        concentration_raw = self.handback_head(fused)
        alpha = F.softplus(concentration_raw[:, 0]) + self.minimum_concentration
        beta = F.softplus(concentration_raw[:, 1]) + self.minimum_concentration
        cost_increments = F.softplus(self.cost_head(fused))
        cost_quantiles = torch.cumsum(cost_increments, dim=-1)
        return {
            "handback_alpha_raw": concentration_raw[:, 0],
            "handback_beta_raw": concentration_raw[:, 1],
            "handback_alpha": alpha,
            "handback_beta": beta,
            "handback_mean": alpha / (alpha + beta),
            "persistent_logit": self.persistent_head(fused).squeeze(-1),
            "source_risk_logit": self.source_risk_head(fused).squeeze(-1),
            "remaining_cost_quantiles": cost_quantiles,
        }


def _action_summary(action: CanonicalActionChunk | torch.Tensor, batch_size: int) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        if action.ndim == 1:
            return action.unsqueeze(0).expand(batch_size, -1)
        return action
    return summary_from_chunk(action).unsqueeze(0).expand(batch_size, -1)

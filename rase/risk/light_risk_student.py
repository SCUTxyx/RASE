"""LightRiskStudent: deployable risk ensemble distilled from V-JEPA 2-AC teacher.

Architecture:
  - TinyUniversalStateEncoder (shared, ~5-15M)
  - CanonicalActionChunk processing (small MLP)
  - Causal history (optional TinyGRU, ~1M)
  - Fusion MLP + 4 heads (student_success, remaining_cost_q, unsafe_ood, progress)
  - Each head: 1-3M params
  - Full package: ~10-25M total

This module does NOT import V-JEPA, world model, or teacher packages.
It only accepts canonical_state (from encoder) + canonical actions + history.

``CandidateArmStudent`` extends the design to the multi-arm selector plan:
  - shared encoder (same TinyUniversalStateEncoder);
  - source-risk head  -> P(source final success) and P(fail within 8/16/32);
  - per-arm success heads -> P(candidate arm rescues | enter now);
  - per-arm cost heads -> quantiles of teacher/action steps;
  - optional unsafe head (de-duplicated from 1 - source_success).
It is the deployable model whose training target is the candidate-arm dataset
built by ``scripts/build_candidate_arm_dataset.py``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from rase.risk.canonical_action import CanonicalActionChunk, summary_from_chunk


class LightRiskHead(nn.Module):
    """Single risk head: fused state + action difference → scores."""

    def __init__(self, fused_dim: int = 128, hidden_dim: int = 128,
                 n_outputs: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_outputs),
        )
        self._n_outputs = n_outputs

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        out = self.mlp(fused)
        if self._n_outputs == 1:
            out = out.squeeze(-1)
        return out


class LightRiskStudent(nn.Module):
    """Deployable risk ensemble.  Each member has independent heads."""

    def __init__(
        self,
        encoder: nn.Module,
        proprio_dim: int = 8,
        action_dim: int = 16,   # summary dim from CanonicalActionChunk
        history_dim: int = 64,
        fused_dim: int = 128,
        head_hidden: int = 128,
        n_members: int = 3,
        n_cost_quantiles: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # shared TinyUniversalStateEncoder
        self.n_members = n_members

        # Action intent: student and OFT action summaries
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Optional history encoder (simple GRU)
        self.history_gru = nn.GRU(
            input_size=6, hidden_size=history_dim,
            num_layers=1, batch_first=True, dropout=0.0,
        )

        # Fusion MLPs (one per member for ensemble diversity)
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder.output_dim + 64 * 2 + history_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, fused_dim),
                nn.GELU(),
            )
            for _ in range(n_members)
        ])

        # Heads per member
        self.success_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, 1, dropout)
            for _ in range(n_members)
        ])
        self.cost_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, n_cost_quantiles, dropout)
            for _ in range(n_members)
        ])
        self.ood_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, 1, dropout)
            for _ in range(n_members)
        ])

        # Training-only projection for evidence distillation.
        # Input dim = encoder output + 2 * action_hidden + history_hidden
        self.distill_proj = nn.Linear(encoder.output_dim + 64 * 2 + history_dim, 64)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        student_action: CanonicalActionChunk | torch.Tensor,
        oft_action: CanonicalActionChunk | torch.Tensor,
        history: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        export_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = image.shape[0]

        # 1. Shared encoder
        state = self.encoder(image, proprio, text_embed)  # (B, D)

        # 2. Action summaries. Accept either a single CanonicalActionChunk
        #    (expanded to batch) or a precomputed per-row summary tensor.
        student_sum = self.action_mlp(_action_summary(student_action, B))
        oft_sum = self.action_mlp(_action_summary(oft_action, B))

        # 3. History
        hist_out, _ = self.history_gru(history)
        hist = hist_out[:, -1]  # last hidden

        # 4. Fused representation per member
        base = torch.cat([state, student_sum, oft_sum, hist], dim=-1)

        all_success = []
        all_cost = []
        all_ood = []
        for m in range(self.n_members):
            fused = self.fusion_mlps[m](base)
            all_success.append(torch.sigmoid(self.success_heads[m](fused)))
            all_cost.append(self.cost_heads[m](fused))
            all_ood.append(torch.sigmoid(self.ood_heads[m](fused)))

        result = {
            "student_success": torch.stack(all_success, dim=0),  # (M, B)
            "remaining_cost": torch.stack(all_cost, dim=0),       # (M, B, Q)
            "unsafe_ood": torch.stack(all_ood, dim=0),            # (M, B)
        }

        if not export_mode:
            result["distill_embedding"] = self.distill_proj(base)

        return result

    def forward_export(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        student_action,
        oft_action,
        history: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward for TorchScript/ONNX export: no DistillProjection."""
        return self.forward(
            image, proprio, student_action, oft_action, history,
            text_embed=text_embed, export_mode=True,
        )


class SourceRiskStudent(nn.Module):
    """Dedicated lightweight source-failure model for R7.

    This module intentionally has one scientific target: the probability that
    the current source VLA fails if allowed to continue.  Fallback success,
    teacher cost and handback targets cannot backpropagate through this model.
    Optional policy-native and world-model streams are additive ablations; they
    never replace the deployable image/proprio/action representation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        action_dim: int = 20,
        fused_dim: int = 128,
        head_hidden: int = 128,
        n_members: int = 1,
        n_policies: int = 0,
        policy_emb_dim: int = 16,
        policy_descriptor_dim: int = 0,
        native_feature_dim: int = 0,
        wm_feature_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_members = n_members
        self.n_policies = n_policies
        self.policy_embedding = (
            nn.Embedding(n_policies, policy_emb_dim) if n_policies > 0 else None
        )
        self.policy_descriptor_dim = policy_descriptor_dim
        self.native_feature_dim = native_feature_dim
        self.wm_feature_dim = wm_feature_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        base_dim = encoder.output_dim + 64
        if self.policy_embedding is not None:
            base_dim += policy_emb_dim
        self.policy_descriptor_proj = None
        if policy_descriptor_dim:
            self.policy_descriptor_proj = nn.Sequential(
                nn.Linear(policy_descriptor_dim, 32), nn.LayerNorm(32), nn.GELU(),
            )
            base_dim += 32
        self.native_proj = None
        if native_feature_dim:
            self.native_proj = nn.Sequential(
                nn.Linear(native_feature_dim, 64), nn.LayerNorm(64), nn.GELU(),
            )
            base_dim += 64
        self.wm_proj = None
        if wm_feature_dim:
            self.wm_proj = nn.Sequential(
                nn.Linear(wm_feature_dim, 32), nn.LayerNorm(32), nn.GELU(),
            )
            base_dim += 32
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(fused_dim, fused_dim), nn.GELU(),
            ) for _ in range(n_members)
        ])
        self.failure_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, 1, dropout)
            for _ in range(n_members)
        ])

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action_summary: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        *,
        policy_index: Optional[torch.Tensor] = None,
        policy_descriptor: Optional[torch.Tensor] = None,
        native_features: Optional[torch.Tensor] = None,
        wm_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # R7 stores agent and wrist cameras as (B, 2, C, H, W).  The generic
        # encoder treats a 5-D input as a temporal sequence, so encode views
        # separately and pool them before fusing the per-state action stream.
        if image.dim() == 5 and image.shape[1] == 2:
            view_a = self.encoder(image[:, 0], proprio, text_embed)
            view_b = self.encoder(image[:, 1], proprio, text_embed)
            state = (view_a + view_b) / 2.0
        elif image.dim() == 4:
            state = self.encoder(image, proprio, text_embed)
        else:
            raise ValueError(f"unexpected image shape {tuple(image.shape)}")
        pieces = [state, self.action_mlp(action_summary)]
        if self.policy_embedding is not None:
            if policy_index is None:
                raise ValueError("policy_index is required when n_policies > 0")
            pieces.append(self.policy_embedding(policy_index))
        elif policy_index is not None:
            raise ValueError("policy_index was provided but n_policies is disabled")
        for name, projection, value in (
            ("policy_descriptor", self.policy_descriptor_proj, policy_descriptor),
            ("native_features", self.native_proj, native_features),
            ("wm_features", self.wm_proj, wm_features),
        ):
            if projection is not None:
                if value is None:
                    raise ValueError(f"{name} is required by this SourceRiskStudent")
                pieces.append(projection(value))
            elif value is not None:
                raise ValueError(f"{name} was provided but its dimension is disabled")
        base = torch.cat(pieces, dim=-1)
        embeddings = []
        logits = []
        for fusion, head in zip(self.fusion_mlps, self.failure_heads, strict=True):
            embedding = fusion(base)
            embeddings.append(embedding)
            logits.append(head(embedding))
        stacked_logits = torch.stack(logits, dim=0)
        return {
            "source_failure_logit": stacked_logits,
            "source_failure": torch.sigmoid(stacked_logits),
            "risk_embedding": torch.stack(embeddings, dim=0),
        }


class RecoverabilityHazardStudent(nn.Module):
    """Lightweight local fallback-recoverability transition model for R8.

    The model is deliberately action-conditioned and causal.  At a boundary it
    predicts whether persistent fallback is currently recoverable, whether it
    remains recoverable after one more eight-step source window, and the
    conditional hazard of losing recoverability during that window.  It never
    consumes OFT actions, teacher cost, future frames, or outcome labels as
    inputs.  An optional seen-policy embedding is a small calibration path, not
    a policy-specific backbone.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        action_dim: int = 20,
        history_dim: int = 28,
        fused_dim: int = 128,
        head_hidden: int = 128,
        n_policies: int = 0,
        policy_emb_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_policies = n_policies
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.history_mlp = nn.Sequential(
            nn.Linear(history_dim, 64), nn.LayerNorm(64), nn.GELU(),
        )
        self.elapsed_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU(),
        )
        self.policy_embedding = (
            nn.Embedding(n_policies, policy_emb_dim) if n_policies > 0 else None
        )
        base_dim = encoder.output_dim + 64 + 64 + 16
        if self.policy_embedding is not None:
            base_dim += policy_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(base_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fused_dim, fused_dim), nn.GELU(),
        )
        self.current_recoverable_head = LightRiskHead(
            fused_dim, head_hidden, 1, dropout
        )
        self.next_recoverable_head = LightRiskHead(
            fused_dim, head_hidden, 1, dropout
        )
        self.loss_hazard_head = LightRiskHead(fused_dim, head_hidden, 1, dropout)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action_summary: torch.Tensor,
        history: torch.Tensor,
        elapsed_context: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        *,
        policy_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if image.dim() == 5 and image.shape[1] == 2:
            first = self.encoder(image[:, 0], proprio, text_embed)
            second = self.encoder(image[:, 1], proprio, text_embed)
            state = (first + second) / 2.0
        elif image.dim() == 4:
            state = self.encoder(image, proprio, text_embed)
        else:
            raise ValueError(f"unexpected image shape {tuple(image.shape)}")
        pieces = [
            state,
            self.action_mlp(action_summary),
            self.history_mlp(history),
            self.elapsed_mlp(elapsed_context),
        ]
        if self.policy_embedding is not None:
            if policy_index is None:
                raise ValueError("policy_index is required when n_policies > 0")
            pieces.append(self.policy_embedding(policy_index))
        elif policy_index is not None:
            raise ValueError("policy_index was provided but n_policies is disabled")
        fused = self.fusion(torch.cat(pieces, dim=-1))
        current = self.current_recoverable_head(fused)
        next_value = self.next_recoverable_head(fused)
        hazard = self.loss_hazard_head(fused)
        return {
            "current_recoverable_logit": current,
            "next_recoverable_logit": next_value,
            "loss_hazard_logit": hazard,
            "current_recoverable": torch.sigmoid(current),
            "next_recoverable": torch.sigmoid(next_value),
            "loss_hazard": torch.sigmoid(hazard),
            "risk_embedding": fused,
        }


def _action_summary(
    action: CanonicalActionChunk | torch.Tensor, B: int
) -> torch.Tensor:
    """Normalize an action input to a per-row summary tensor (B, action_dim)."""
    if isinstance(action, torch.Tensor):
        if action.dim() == 1:
            return action.unsqueeze(0).expand(B, -1)
        return action
    return summary_from_chunk(action).unsqueeze(0).expand(B, -1)


class CandidateArmStudent(nn.Module):
    """Deployable multi-arm risk-return student for the conservative selector.

    Shared encoder + source adapter is the "universal risk backbone".  Each
    candidate arm gets its own success and cost heads.  Heads are:

      - ``source_success``       P(source final success), (M, B);
      - ``source_within_k``      P(source succeeds within 8/16/32), (M, B, 3);
      - ``arm_success``          P(arm rescues | enter now), (M, B, n_arms);
      - ``arm_concentration``    beta-binomial concentration, (M, B, n_arms);
      - ``arm_cost``             teacher/action step quantiles, (M, B, n_arms, Q);
      - ``unsafe``               optional irreversible-transition head, (M, B).

    ``n_arms`` is fixed at construction and every arm shares the same backbone.
    Cost heads emit ordered, non-negative log1p(step) quantiles.  The training
    script fits success mean and concentration jointly with a beta-binomial
    likelihood, so repeated counterfactual rollouts express aleatoric label
    uncertainty without becoming independent OOF examples.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_arms: int = 2,
        *,
        proprio_dim: int = 8,
        action_dim: int = 20,
        history_dim: int = 64,
        fused_dim: int = 128,
        head_hidden: int = 128,
        n_members: int = 3,
        n_cost_quantiles: int = 3,
        use_unsafe_head: bool = False,
        wm_dim: int = 0,
        n_policies: int = 0,
        policy_emb_dim: int = 16,
        policy_descriptor_dim: int = 0,
        use_calibration_adapter: bool = False,
        use_advantage_head: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # shared TinyUniversalStateEncoder
        self.n_arms = n_arms
        self.n_members = n_members
        self.use_unsafe_head = use_unsafe_head
        self.wm_dim = wm_dim
        self.n_policies = n_policies
        self.policy_emb_dim = policy_emb_dim
        self.policy_descriptor_dim = policy_descriptor_dim
        self.use_calibration_adapter = use_calibration_adapter
        self.use_advantage_head = use_advantage_head

        # Policy conditioning (R6-C.1C): VLA identity embedding for seen VLAs
        # plus an optional deployable behavior descriptor stream for new VLAs.
        # The descriptor is what makes the core adaptable without retraining.
        self.policy_embedding = None
        if n_policies > 0:
            self.policy_embedding = nn.Embedding(n_policies, policy_emb_dim)
        self.descriptor_mlp = None
        if policy_descriptor_dim > 0:
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(policy_descriptor_dim, policy_emb_dim),
                nn.LayerNorm(policy_emb_dim),
                nn.GELU(),
            )
        # Descriptor-conditioned FiLM calibration.  Parameters are shared;
        # adaptation to a VLA occurs through its behavior descriptor, not a
        # hidden per-policy parameter table.
        self.policy_film = None
        if use_calibration_adapter:
            self.policy_film = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(policy_emb_dim, 2 * fused_dim),
                    nn.GELU(),
                    nn.Linear(2 * fused_dim, 2 * fused_dim),
                ) for _ in range(n_members)
            ])

        # Source policy adapter: canonical action summary (fixed, arm 0 is the
        # source continuation so no separate adapter is needed for it).
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.history_mlp = nn.Sequential(
            nn.Linear(history_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        base_dim = encoder.output_dim + 64 + 64  # state + source action + history
        if n_policies > 0 or policy_descriptor_dim > 0:
            base_dim += policy_emb_dim
        if wm_dim > 0:
            # Pre-registered world-model auxiliary features enter as an
            # additional input stream (never a replacement).
            self.wm_proj = nn.Sequential(
                nn.Linear(wm_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
            )
            base_dim += 32
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, fused_dim), nn.LayerNorm(fused_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(fused_dim, fused_dim), nn.GELU(),
            )
            for _ in range(n_members)
        ])
        self.source_success_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, 1, dropout) for _ in range(n_members)
        ])
        self.source_within_heads = nn.ModuleList([
            LightRiskHead(fused_dim, head_hidden, 3, dropout) for _ in range(n_members)
        ])
        # Per-arm heads: one success + one cost-quantile head per arm and member.
        self.arm_success_heads = nn.ModuleList([
            nn.ModuleList([
                LightRiskHead(fused_dim, head_hidden, 1, dropout)
                for _ in range(n_arms)
            ]) for _ in range(n_members)
        ])
        self.arm_concentration_heads = nn.ModuleList([
            nn.ModuleList([
                LightRiskHead(fused_dim, head_hidden, 1, dropout)
                for _ in range(n_arms)
            ]) for _ in range(n_members)
        ])
        self.arm_cost_heads = nn.ModuleList([
            nn.ModuleList([
                LightRiskHead(fused_dim, head_hidden, n_cost_quantiles, dropout)
                for _ in range(n_arms)
            ]) for _ in range(n_members)
        ])
        # Direct success-advantage head: P(success | enter OFT) minus
        # P(success | continue source).  Teacher cost is modeled separately by
        # arm-cost quantiles and cost-constrained threshold selection.
        self.advantage_heads = (
            nn.ModuleList([LightRiskHead(fused_dim, head_hidden, 1, dropout)
                           for _ in range(n_members)]) if use_advantage_head else None
        )
        self.unsafe_heads = (
            nn.ModuleList([LightRiskHead(fused_dim, head_hidden, 1, dropout)
                           for _ in range(n_members)]) if use_unsafe_head else None
        )

    def forward(
        self,
        image: torch.Tensor,            # (B, 2, 3, H, W) two RGB views
        proprio: torch.Tensor,          # (B, 8)
        source_action: torch.Tensor,    # (B, action_dim) source canonical summary
        history: torch.Tensor,          # (B, history_dim) causal action history
        text_embed: Optional[torch.Tensor] = None,  # (B, E) optional
        wm_features: Optional[torch.Tensor] = None,  # (B, wm_dim) optional
        policy_index: Optional[torch.Tensor] = None,  # (B,) seen-VLA identity
        policy_descriptor: Optional[torch.Tensor] = None,  # (B, policy_descriptor_dim)
    ) -> dict[str, torch.Tensor]:
        B = image.shape[0]
        # The B1.2 npz stores the two views already stacked as (B, 2, C, H, W);
        # feed each view through the shared encoder and average the embeddings.
        if image.dim() == 5 and image.shape[1] == 2:
            view_a = self.encoder(image[:, 0], proprio, text_embed)
            view_b = self.encoder(image[:, 1], proprio, text_embed)
            state = (view_a + view_b) / 2.0
        elif image.dim() == 4:
            state = self.encoder(image, proprio, text_embed)
        else:
            raise ValueError(f"unexpected image shape {tuple(image.shape)}")

        src = self.action_mlp(source_action)
        hist = self.history_mlp(history)
        pieces = [state, src, hist]

        # Policy conditioning: descriptor (new VLA) takes precedence over the
        # identity embedding (seen VLA); the descriptor is the deployable path.
        if self.policy_descriptor_dim > 0:
            if policy_descriptor is None:
                raise ValueError("policy_descriptor_dim > 0 but policy_descriptor not provided")
            policy = self.descriptor_mlp(policy_descriptor)
        elif self.policy_embedding is not None:
            if policy_index is None:
                raise ValueError("policy embedding enabled but policy_index not provided")
            policy = self.policy_embedding(policy_index)
        else:
            policy = None
        if policy is not None:
            pieces.append(policy)

        if self.wm_dim > 0:
            if wm_features is None:
                raise ValueError("wm_dim > 0 but wm_features not provided")
            pieces.append(self.wm_proj(wm_features))
        base = torch.cat(pieces, dim=-1)

        all_source = []
        all_within = []
        all_arm_success = []
        all_arm_concentration = []
        all_arm_cost = []
        all_advantage = []
        all_unsafe = []
        for member in range(self.n_members):
            fused = self.fusion_mlps[member](base)
            if self.policy_film is not None:
                scale, bias = self.policy_film[member](policy).chunk(2, dim=-1)
                fused = fused * (1.0 + scale) + bias
            all_source.append(torch.sigmoid(self.source_success_heads[member](fused)))
            all_within.append(torch.sigmoid(self.source_within_heads[member](fused)))
            success_arms = torch.stack([
                torch.sigmoid(self.arm_success_heads[member][arm](fused))
                for arm in range(self.n_arms)
            ], dim=-1)
            concentration_arms = torch.stack([
                2.0 + torch.nn.functional.softplus(
                    self.arm_concentration_heads[member][arm](fused)
                )
                for arm in range(self.n_arms)
            ], dim=-1)
            raw_cost_arms = torch.stack([
                self.arm_cost_heads[member][arm](fused)
                for arm in range(self.n_arms)
            ], dim=-2)
            # Monotone parameterization prevents negative teacher steps and
            # crossed q10/q50/q90 predictions at deployment.
            q10 = torch.nn.functional.softplus(raw_cost_arms[..., 0])
            q50 = q10 + torch.nn.functional.softplus(raw_cost_arms[..., 1])
            q90 = q50 + torch.nn.functional.softplus(raw_cost_arms[..., 2])
            cost_arms = torch.stack([q10, q50, q90], dim=-1)
            all_arm_success.append(success_arms)
            all_arm_concentration.append(concentration_arms)
            all_arm_cost.append(cost_arms)
            if self.advantage_heads is not None:
                all_advantage.append(self.advantage_heads[member](fused))
            if self.unsafe_heads is not None:
                all_unsafe.append(torch.sigmoid(self.unsafe_heads[member](fused)))

        result = {
            "source_success": torch.stack(all_source, dim=0),
            "source_within": torch.stack(all_within, dim=0),
            "arm_success": torch.stack(all_arm_success, dim=0),
            "arm_concentration": torch.stack(all_arm_concentration, dim=0),
            "arm_cost": torch.stack(all_arm_cost, dim=0),
        }
        if all_advantage:
            result["advantage"] = torch.stack(all_advantage, dim=0)
        if all_unsafe:
            result["unsafe"] = torch.stack(all_unsafe, dim=0)
        return result

#!/usr/bin/env python3
"""Recovery Selector for Route C.

A lightweight binary classifier that decides whether the current step
needs Plugin intervention. Shares the same input encoder architecture
as ResidualRecoveryPlugin but outputs a single logit instead of a 7-dim
delta vector.

Inputs (identical to Plugin, plus optional delta):
  - proprio history (W=8 timesteps of proprio, student action, progress)
  - current observation features (from SmolVLA preprocessor, 144-D)
  - current SmolVLA action aS (7-D)
  - plugin_delta (7-D, optional) — Plugin's predicted correction vector
  - delta_norm (1-D float, optional) — L2 norm of plugin_delta

The plugin_delta + delta_norm provide the Selector a direct signal of
how large the Plugin's proposed correction is. A near-zero delta is a
strong indicator that takeover is unnecessary.

Output:
  - logit = selector(history, obs_features, student_action, [plugin_delta, delta_norm])
  - sigmoid(logit) > 0.5 -> activate Plugin
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RecoverySelector(nn.Module):
    def __init__(
        self,
        proprio_dim: int = 8,
        action_dim: int = 7,
        history_window: int = 8,
        obs_feature_dim: int = 144,
        hidden_dim: int = 64,
        num_layers: int = 1,
        use_delta_features: bool = True,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.history_window = history_window
        self.obs_feature_dim = obs_feature_dim
        self._use_delta_features = use_delta_features

        history_input_dim = history_window * (proprio_dim + action_dim + 1 + action_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # Optional delta feature encoder: 7-D delta + 1-D norm -> 16-D
        if use_delta_features:
            self.delta_encoder = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
            )
            delta_head_dim = 16
        else:
            self.delta_encoder = None
            delta_head_dim = 0

        layers = []
        in_dim = hidden_dim * 2 + action_dim + delta_head_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @property
    def use_delta_features(self) -> bool:
        return self._use_delta_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        history: torch.Tensor,
        obs_features: torch.Tensor,
        student_action: torch.Tensor,
        plugin_delta: torch.Tensor | None = None,
        delta_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_encoded = self.history_encoder(history.view(history.size(0), -1))
        o_encoded = self.obs_encoder(obs_features.view(obs_features.size(0), -1))

        parts = [h_encoded, o_encoded, student_action]

        # Encode delta features if available and model expects them
        if self._use_delta_features and self.delta_encoder is not None:
            if plugin_delta is not None and delta_norm is not None:
                # Ensure both are 2D [B, D]
                p_d = plugin_delta.view(history.size(0), -1)
                n_d = delta_norm.view(history.size(0), -1)
                delta_input = torch.cat([p_d, n_d], dim=-1)
            elif plugin_delta is not None:
                norm = torch.norm(plugin_delta.view(history.size(0), -1), dim=-1, keepdim=True)
                delta_input = torch.cat([plugin_delta.view(history.size(0), -1), norm], dim=-1)
            else:
                delta_input = torch.zeros(
                    history.size(0), 8,
                    dtype=history.dtype, device=history.device)
            d_encoded = self.delta_encoder(delta_input)
            parts.append(d_encoded)

        combined = torch.cat(parts, dim=-1)
        logit = self.head(combined)
        return logit.squeeze(-1)

    @torch.no_grad()
    def predict(
        self,
        history: np.ndarray,
        obs_features: np.ndarray,
        student_action: np.ndarray,
        plugin_delta: np.ndarray | None = None,
        delta_norm: float | None = None,
    ) -> float:
        """Return probability that Plugin should intervene (0-1)."""
        h_t = torch.from_numpy(history).float().unsqueeze(0)
        o_t = torch.from_numpy(obs_features).float().unsqueeze(0)
        a_t = torch.from_numpy(student_action).float().reshape(1, -1)

        d_t = None
        n_t = None
        if plugin_delta is not None:
            d_t = torch.from_numpy(
                np.asarray(plugin_delta, dtype=np.float32).flatten()[:7]
            ).float().unsqueeze(0)
            if delta_norm is not None:
                n_t = torch.tensor([[float(delta_norm)]], dtype=torch.float32)

        logit = self.forward(h_t, o_t, a_t, plugin_delta=d_t, delta_norm=n_t)
        prob = torch.sigmoid(logit).item()
        return prob

    @torch.no_grad()
    def should_intervene(
        self,
        history: np.ndarray,
        obs_features: np.ndarray,
        student_action: np.ndarray,
        plugin_delta: np.ndarray | None = None,
        delta_norm: float | None = None,
        threshold: float = 0.5,
    ) -> bool:
        """Boolean decision: should Plugin take over?"""
        return self.predict(
            history, obs_features, student_action,
            plugin_delta=plugin_delta,
            delta_norm=delta_norm,
        ) >= threshold


def make_selector(
    proprio_dim: int = 8,
    action_dim: int = 7,
    history_window: int = 8,
    obs_feature_dim: int = 144,
    hidden_dim: int = 64,
    num_layers: int = 1,
    use_delta_features: bool = True,
) -> RecoverySelector:
    return RecoverySelector(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        history_window=history_window,
        obs_feature_dim=obs_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_delta_features=use_delta_features,
    )


def save_selector(selector: RecoverySelector, path: str):
    torch.save(
        {
            "model_state_dict": selector.state_dict(),
            "config": {
                "proprio_dim": selector.proprio_dim,
                "action_dim": selector.action_dim,
                "history_window": selector.history_window,
                "obs_feature_dim": selector.obs_feature_dim,
                "hidden_dim": selector.history_encoder[0].out_features,
                "num_layers": (len(selector.head) - 1) // 2,
                "use_delta_features": selector.use_delta_features,
            },
        },
        path,
    )


def load_selector(path: str) -> RecoverySelector:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    # Backward compat: old checkpoints may not have use_delta_features
    if "use_delta_features" not in cfg:
        cfg["use_delta_features"] = False
    selector = RecoverySelector(**cfg)
    selector.load_state_dict(ckpt["model_state_dict"], strict=False)
    return selector

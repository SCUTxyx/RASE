#!/usr/bin/env python3
"""Standalone residual recovery plugin for Route C.

A lightweight feedforward network that takes:
  - proprio history (W=8 timesteps of proprio, student action, progress)
  - current observation features (from SmolVLA preprocessor)
  - current SmolVLA action aS
and outputs:
  - delta_a = clip(plugin(proprio, obs_feats, aS), -delta_clip, +delta_clip)

Designed to be frozen SmolVLA-compatible: no LoRA, no hidden state access.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualRecoveryPlugin(nn.Module):
    def __init__(
        self,
        proprio_dim: int = 8,
        action_dim: int = 7,
        history_window: int = 8,
        obs_feature_dim: int = 144,      # F2 feature dim (128 latent + 7 proprio + 7 action + 2 stats)
        hidden_dim: int = 128,
        num_layers: int = 2,
        delta_clip: float = 0.5,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.history_window = history_window
        self.obs_feature_dim = obs_feature_dim
        self.delta_clip = delta_clip

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

        layers = []
        in_dim = hidden_dim * 2 + action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.head = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, history: torch.Tensor, obs_features: torch.Tensor,
                student_action: torch.Tensor) -> torch.Tensor:
        h_encoded = self.history_encoder(history.view(history.size(0), -1))
        o_encoded = self.obs_encoder(obs_features.view(obs_features.size(0), -1))
        combined = torch.cat([h_encoded, o_encoded, student_action], dim=-1)
        delta_raw = self.head(combined)
        delta_clipped = torch.clamp(delta_raw, -self.delta_clip, self.delta_clip)
        return delta_clipped

    @torch.no_grad()
    def predict_delta(self, history: np.ndarray, obs_features: np.ndarray,
                      student_action: np.ndarray) -> np.ndarray:
        h_t = torch.from_numpy(history).float().unsqueeze(0)
        o_t = torch.from_numpy(obs_features).float().unsqueeze(0)
        a_t = torch.from_numpy(student_action).float().reshape(1, -1)
        delta = self.forward(h_t, o_t, a_t)
        return delta.squeeze(0).cpu().numpy()


def make_recovery_plugin(proprio_dim: int = 8, action_dim: int = 7,
                         history_window: int = 8, obs_feature_dim: int = 128,
                         hidden_dim: int = 128, num_layers: int = 2,
                         delta_clip: float = 0.5) -> ResidualRecoveryPlugin:
    return ResidualRecoveryPlugin(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        history_window=history_window,
        obs_feature_dim=obs_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        delta_clip=delta_clip,
    )


def save_plugin(plugin: ResidualRecoveryPlugin, path: str):
    torch.save({"model_state_dict": plugin.state_dict(),
                "config": {"proprio_dim": plugin.proprio_dim,
                           "action_dim": plugin.action_dim,
                           "history_window": plugin.history_window,
                           "obs_feature_dim": plugin.obs_feature_dim,
                           "hidden_dim": plugin.history_encoder[0].out_features,
                           "num_layers": (len(plugin.head) - 1) // 2,
                           "delta_clip": plugin.delta_clip}}, path)


def load_plugin(path: str) -> ResidualRecoveryPlugin:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    plugin = ResidualRecoveryPlugin(**cfg)
    plugin.load_state_dict(ckpt["model_state_dict"])
    return plugin

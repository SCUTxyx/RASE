"""Tiny Universal State Encoder for multi-VLA risk prediction.

A lightweight (5-15M parameter) CNN/MLP encoder that produces a canonical
state embedding from RGB, proprioception, and frozen task-language context.
It is shared by all risk heads and trained via teacher distillation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyUniversalStateEncoder(nn.Module):
    """Small shared encoder: RGB → features + proprio fusion + text context.

    Target: 5-15M parameters.  Runs once per handback boundary.
    Does NOT import V-JEPA, VLA internals, or read hidden states.
    """

    def __init__(
        self,
        image_size: int = 128,
        proprio_dim: int = 8,
        text_embed_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        input_mode: str = "image",
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_mode = input_mode
        # Lightweight CNN: ~2M params
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )  # -> 128
        # Latent-input fallback (used when no RGB frames are available yet):
        # maps a frozen latent vector to the same 128-dim visual embedding.
        self.latent_proj: Optional[nn.Linear] = None
        if input_mode == "latent":
            self.latent_proj = nn.Linear(latent_dim, 128)

        # Proprio fusion
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

        # Task context: frozen text embedding
        self.text_proj: Optional[nn.Linear] = None
        if text_embed_dim > 0:
            self.text_proj = nn.Linear(text_embed_dim, 64)

        # Fusion MLP: CNN + proprio + text → state embedding
        fusion_in = 128 + 64 + (64 if self.text_proj is not None else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        image: torch.Tensor,          # (B, 3, H, W) or (B, T, 3, H, W)
        proprio: torch.Tensor,         # (B, D) or (B, T, D)
        text_embed: Optional[torch.Tensor] = None,  # (B, E) or (B, T, E)
    ) -> torch.Tensor:
        # Handle temporal dimension if present
        if image.dim() == 5:
            B, T, C, H, W = image.shape
            image = image.reshape(B * T, C, H, W)
            proprio = proprio.reshape(B * T, -1)
            if text_embed is not None:
                text_embed = text_embed.reshape(B * T, -1)
            flat = True
        else:
            flat = False

        vis = (
            self.cnn(image)
            if self.input_mode == "image"
            else self.latent_proj(image)
        )
        prop = self.proprio_mlp(proprio)

        parts = [vis, prop]
        if text_embed is not None and self.text_proj is not None:
            parts.append(self.text_proj(text_embed))

        fused = torch.cat(parts, dim=-1)
        out = self.fusion(fused)

        if flat:
            out = out.reshape(B, T, -1)
        return out

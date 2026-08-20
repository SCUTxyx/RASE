"""K-step latent rollout from cached V-JEPA 2-AC features.

Offline teacher utility: given a window of cached encoder tokens and a canonical
action chunk, roll out the frozen AC predictor autoregressively and return the
latent trajectory + per-step deltas.  Used by the teacher evidence cache
(Milestone 3a) when the V-JEPA teacher is enabled; the deployment risk model
never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def rollout_from_tokens(
    tokens: torch.Tensor,
    action_chunk: np.ndarray,
    predictor: Any,
    *,
    grid_height: int,
    grid_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    tubelet_size: int = 2,
) -> dict[str, np.ndarray]:
    """Autoregressively apply the AC predictor over a cached token window.

    Args:
        tokens: (N, D) or (1, N, D) cached encoder tokens.
        action_chunk: (K, A) canonical-action tensor (already in teacher space).
        predictor: V-JEPA 2-AC predictor module.
        grid_height/grid_width: spatial token grid (img_size // patch_size).
        tubelet_size: 2 for V-JEPA 2-AC.

    Returns dict with 'latents' (K+1, D) pooled latents and 'deltas' (K, D).
    """
    assert action_chunk.ndim == 2, "action_chunk must be (K, A)"
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    tokens = tokens.to(device).to(dtype)
    B, N, D = tokens.shape
    T = N // (grid_height * grid_width)
    pool = tokens.mean(dim=1)  # (B, D)

    latents: list[np.ndarray] = [pool.squeeze(0).float().cpu().numpy()]
    deltas: list[np.ndarray] = []

    current = tokens
    with torch.no_grad():
        for step in range(len(action_chunk)):
            action_t = torch.as_tensor(
                action_chunk[step], device=device, dtype=dtype
            ).reshape(1, -1).unsqueeze(1).expand(B, T, -1)
            state_t = torch.zeros_like(action_t)
            next_tokens = predictor(current, action_t, state_t)
            next_pool = next_tokens.mean(dim=1)
            latents.append(next_pool.squeeze(0).float().cpu().numpy())
            deltas.append((next_pool - pool).squeeze(0).float().cpu().numpy())
            current = next_tokens
            pool = next_pool

    return {
        "latents": np.stack(latents, axis=0),
        "deltas": np.stack(deltas, axis=0),
    }


def cached_token_paths(cache_dir: str | Path, frame_ids: list[str]) -> list[Path]:
    cache_dir = Path(cache_dir)
    paths = [cache_dir / f"{fid}.pt" for fid in frame_ids]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached tokens: {[p.name for p in missing]}")
    return paths


def load_cached_tokens(paths: list[Path]) -> torch.Tensor:
    token_list = [torch.load(p, map_location="cpu") for p in paths]
    return torch.cat(token_list, dim=0)  # (N, D)

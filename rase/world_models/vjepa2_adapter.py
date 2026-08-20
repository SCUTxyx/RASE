"""V-JEPA 2-AC offline visual encoder adapter for LIBERO agent-view images.

This module provides frozen encoder inference, token caching, and pooled latent
extraction for the V-JEPA 2 action-conditioned predictor.  It is used ONLY during
offline teacher training; the LightRiskStudent deployment does not import or load
any V-JEPA modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


VJEPA_IMAGE_SIZE = 224
VJEPA_MEAN = (0.485, 0.456, 0.406)
VJEPA_STD = (0.229, 0.224, 0.225)


def preprocess_image(
    image: np.ndarray,
    target_size: int = VJEPA_IMAGE_SIZE,
) -> torch.Tensor:
    """Convert a LIBERO agent-view RGB image to V-JEPA input tensor."""
    import torch.nn.functional as F

    img = torch.from_numpy(np.asarray(image, dtype=np.float32))
    if img.ndim == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(target_size, target_size), mode="bilinear",
                        align_corners=False)
    img = img / 255.0
    mean = torch.tensor(VJEPA_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(VJEPA_STD).view(1, 3, 1, 1)
    return (img - mean) / std


def preprocess_frame_stack(
    frames: list[np.ndarray],
    target_size: int = VJEPA_IMAGE_SIZE,
) -> torch.Tensor:
    tensors = [preprocess_image(f, target_size).squeeze(0) for f in frames]
    return torch.stack(tensors, dim=0)


def pad_to_num_frames(
    frames: list[np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:
    """Pad a short frame list to exactly `num_frames` by repeating the last frame."""
    frames = list(frames)
    if len(frames) >= num_frames:
        return frames[:num_frames]
    if len(frames) == 0:
        raise ValueError("cannot pad an empty frame list")
    last = frames[-1]
    while len(frames) < num_frames:
        frames.append(last)
    return frames


class VJEPA2ACEncoder:
    """Frozen V-JEPA 2-AC encoder for offline teacher inference."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 64,
        img_size: int = 256,
        patch_size: int = 16,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_height = img_size // patch_size
        self.grid_width = img_size // patch_size
        self._encoder: Any = None
        self._predictor: Any = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        # Load the frozen AC teacher directly from the local (URL-patched)
        # facebookresearch/vjepa2 repo.  torch.hub is NOT used because it
        # re-fetches the upstream code and undoes the CDN URL patch.
        # `checkpoint_dir` points at the repo root; `src.hub.backbones` is
        # importable from that root (src is a namespace package there).
        repo = self.checkpoint_dir
        for name in ("", "src", "evals"):
            pkg = str(repo / name) if name else str(repo)
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
        from src.hub.backbones import vjepa2_ac_vit_giant

        encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
        encoder.to(self.device).to(self.dtype).eval()
        predictor.to(self.device).to(self.dtype).eval()
        self._encoder = encoder
        self._predictor = predictor
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def encode_single(self, image: np.ndarray) -> torch.Tensor:
        self._ensure_loaded()
        # Video input: (B, C, T, H, W) with T padded to num_frames
        padded = pad_to_num_frames([image], self.num_frames)
        x = preprocess_frame_stack(padded, self.img_size)  # (T, C, H, W)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)             # (B, C, T, H, W)
        x = x.to(self.device).to(self.dtype)
        return self._encoder(x)

    @torch.no_grad()
    def encode_stack(self, frames: list[np.ndarray]) -> torch.Tensor:
        self._ensure_loaded()
        padded = pad_to_num_frames(frames, self.num_frames)
        x = preprocess_frame_stack(padded, self.img_size)  # (T, C, H, W)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)             # (B, C, T, H, W)
        x = x.to(self.device).to(self.dtype)
        return self._encoder(x)

    @torch.no_grad()
    def pooled_latent(self, frames: list[np.ndarray]) -> np.ndarray:
        latent = self.encode_stack(frames)
        # latent: (B, N_tokens, D). Pool over token dim -> (B, D)
        pooled = latent.mean(dim=1)
        return pooled.squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def predict_step(
        self,
        latent: torch.Tensor,
        action: np.ndarray,
        state: np.ndarray | None = None,
    ) -> torch.Tensor:
        """One action-conditioned latent prediction.

        latent: encoder tokens (B, N_tokens, D).
        action: (7,) or (1, 7) LIBERO-space action.
        state:  (7,) proprio-like state, defaults to zeros.
        """
        self._ensure_loaded()
        B, N_ctxt, D = latent.shape
        T = N_ctxt // (self.grid_height * self.grid_width)
        action_t = torch.as_tensor(
            np.asarray(action, dtype=np.float32), device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        state_t = torch.zeros_like(action_t) if state is None else torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        # actions/states expected as (B, T, action_dim)
        action_t = action_t.unsqueeze(1).expand(B, T, -1)
        state_t = state_t.unsqueeze(1).expand(B, T, -1)
        return self._predictor(latent, action_t, state_t)

    @torch.no_grad()
    def predict_k_step(
        self,
        start_frames: list[np.ndarray],
        action_chunk: np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Iterative k-step AC prediction with autoregressive latent update.

        Returns per-step pooled delta vectors (k, D).
        """
        self._ensure_loaded()
        current_latent = self.encode_stack(start_frames)  # (1, N, D)
        return self.rollout_from_latent(current_latent, action_chunk, k=k)

    @torch.no_grad()
    def rollout_from_latent(
        self,
        latent: torch.Tensor,
        action_chunk: np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Autoregressive AC rollout starting from an already-encoded latent.

        Returns per-step pooled delta vectors (k, D).  Reuses the provided
        latent so callers can cache encoder outputs across windows.
        """
        self._ensure_loaded()
        current_latent = latent.to(self.device).to(self.dtype)
        deltas = []
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        for step in range(min(k, len(actions))):
            next_latent = self.predict_step(current_latent, actions[step])
            delta = next_latent.mean(dim=1) - current_latent.mean(dim=1)
            deltas.append(delta.squeeze(0).float().cpu().numpy())
            current_latent = next_latent
        return np.stack(deltas, axis=0) if deltas else np.zeros((0,), dtype=np.float32)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def cache_encoder_tokens(
        self,
        frames: list[np.ndarray],
        cache_dir: str | Path,
        frame_ids: list[str],
    ) -> dict[str, Path]:
        self._ensure_loaded()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = {}
        for i, (frame, fid) in enumerate(zip(frames, frame_ids)):
            path = cache_dir / f"{fid}.pt"
            if not path.exists():
                latent = self.encode_single(frame).cpu()
                torch.save(latent, path)
            cached[fid] = path
        return cached

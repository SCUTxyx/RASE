"""Per-VLA action adapters: raw VLA action tensors → CanonicalActionChunk.

Each Student VLA implements a thin adapter plugin.  The shared risk core only
sees canonical actions, never VLA-specific conventions.

Currently supported (identity/normalization only — LIBERO 7-DoF convention):
  - SmolVLA (smolvla)
  - OFT teacher (oft)

New VLA: add a plugin class + registry entry; risk core weights are untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from rase.risk.canonical_action import CanonicalActionChunk


class VLAActionAdapter(ABC):
    """Convert a VLA's raw action chunk to CanonicalActionChunk."""

    vla_name: str = ""

    @abstractmethod
    def to_canonical(self, raw_action_chunk: np.ndarray) -> CanonicalActionChunk:
        ...

    @abstractmethod
    def config_dict(self) -> dict:
        ...


class SmolVLAActionAdapter(VLAActionAdapter):
    """SmolVLA adapter.

    LIBERO 7-DoF: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    Identity mapping to canonical; rotation converted to 6D via axis-angle
    approximation (identity within a step: droll/dpitch/dyaw are treated as
    incremental angular deltas).  For chunk-level mapping we keep 7-dim and
    construct rotation_6d as [cos(theta)-1, sin(theta)] per axis (SO(3) exp
    near identity).  Gripper normalized to [-1, 1] open/close.
    """

    vla_name = "smolvla"

    def to_canonical(self, raw_action_chunk: np.ndarray) -> CanonicalActionChunk:
        chunk = np.asarray(raw_action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        H, D = chunk.shape
        assert D == 7, f"SmolVLA expects 7-DoF, got {D}"

        delta_position = chunk[:, :3]                      # (H, 3)
        # Convert incremental Euler deltas to 6D rotation approximation
        rot = chunk[:, 3:6]                                # (H, 3) angular deltas
        # 6D rotation: for small angles ~ [axis_normed * (cos-1), axis_normed * sin]
        ang = np.linalg.norm(rot, axis=-1, keepdims=True).clip(1e-6)
        axis = rot / ang
        rotation_6d = np.concatenate([
            axis * (np.cos(ang) - 1.0),
            axis * np.sin(ang),
        ], axis=-1)                                        # (H, 6)
        gripper = np.clip(chunk[:, 6:7], -1.0, 1.0)        # (H, 1)
        delta_time = np.full((H, 1), 0.1, dtype=np.float32)
        valid_mask = np.ones(H, dtype=np.float32)

        return CanonicalActionChunk.from_numpy(
            delta_position, rotation_6d, gripper,
            delta_time=delta_time, valid_mask=valid_mask,
        )

    def config_dict(self) -> dict:
        return {"vla_name": self.vla_name, "convention": "libero_7dof"}


class OFTActionAdapter(VLAActionAdapter):
    """OFT (teacher) adapter — same LIBERO 7-DoF convention as SmolVLA."""

    vla_name = "oft"

    def to_canonical(self, raw_action_chunk: np.ndarray) -> CanonicalActionChunk:
        chunk = np.asarray(raw_action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        H, D = chunk.shape
        assert D == 7, f"OFT expects 7-DoF, got {D}"
        delta_position = chunk[:, :3]
        rot = chunk[:, 3:6]
        ang = np.linalg.norm(rot, axis=-1, keepdims=True).clip(1e-6)
        axis = rot / ang
        rotation_6d = np.concatenate([
            axis * (np.cos(ang) - 1.0),
            axis * np.sin(ang),
        ], axis=-1)
        gripper = np.clip(chunk[:, 6:7], -1.0, 1.0)
        delta_time = np.full((H, 1), 0.1, dtype=np.float32)
        valid_mask = np.ones(H, dtype=np.float32)
        return CanonicalActionChunk.from_numpy(
            delta_position, rotation_6d, gripper,
            delta_time=delta_time, valid_mask=valid_mask,
        )

    def config_dict(self) -> dict:
        return {"vla_name": self.vla_name, "convention": "libero_7dof"}


VLA_ADAPTER_REGISTRY: dict[str, type[VLAActionAdapter]] = {
    "smolvla": SmolVLAActionAdapter,
    "smolvla_libero": SmolVLAActionAdapter,
    "pi0fast": SmolVLAActionAdapter,
    "pi0fast_libero": SmolVLAActionAdapter,
    "pi05": SmolVLAActionAdapter,
    "pi05_libero": SmolVLAActionAdapter,
    "oft": OFTActionAdapter,
}


def create_vla_adapter(vla_name: str) -> VLAActionAdapter:
    """Create a VLA action adapter by name (case-insensitive)."""
    key = vla_name.lower().strip()
    if key not in VLA_ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown VLA adapter: {vla_name}. Available: {list(VLA_ADAPTER_REGISTRY)}"
        )
    return VLA_ADAPTER_REGISTRY[key]()

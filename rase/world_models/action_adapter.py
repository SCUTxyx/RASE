"""Action adapter: LIBERO 7-DoF actions → V-JEPA 2-AC canonical action space.

LIBERO/SmolVLA uses actions of the form:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

V-JEPA 2-AC was post-trained on DROID data with a different coordinate convention.
This adapter performs explicit coordinate transform, value clipping, gripper-alignment,
and frame-rate resampling before actions are fed into the frozen AC predictor.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


class LiberoToVJEPAActionAdapter:
    """Map LIBERO 7-DoF action conventions to V-JEPA 2-AC expected input.

    The adapter applies a learned/configured linear transform per dimension group
    (position, rotation, gripper), then clips to the ranges seen in the V-JEPA 2-AC
    DROID training distribution.  All normalization parameters are versioned so that
    every teacher-evidence cache record can include the adapter hash.
    """

    def __init__(
        self,
        position_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
        position_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
        position_clip: tuple[float, float] = (-0.15, 0.15),
        rotation_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
        rotation_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
        rotation_clip: tuple[float, float] = (-0.30, 0.30),
        gripper_scale: float = 1.0,
        gripper_offset: float = 0.0,
        gripper_clip: tuple[float, float] = (-1.0, 1.0),
        source_frequency: float = 10.0,
        target_frequency: float = 15.0,
        version: str = "v1",
    ) -> None:
        self.position_scale = np.asarray(position_scale, dtype=np.float32)
        self.position_offset = np.asarray(position_offset, dtype=np.float32)
        self.position_clip = position_clip
        self.rotation_scale = np.asarray(rotation_scale, dtype=np.float32)
        self.rotation_offset = np.asarray(rotation_offset, dtype=np.float32)
        self.rotation_clip = rotation_clip
        self.gripper_scale = float(gripper_scale)
        self.gripper_offset = float(gripper_offset)
        self.gripper_clip = gripper_clip
        self.source_frequency = float(source_frequency)
        self.target_frequency = float(target_frequency)
        self.version = str(version)
        self._hash = self._compute_hash()

    def transform(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        squeeze = action.ndim == 1
        if squeeze:
            action = action.reshape(1, -1)
        result = np.zeros_like(action)
        result[:, :3] = np.clip(
            action[:, :3] * self.position_scale + self.position_offset,
            self.position_clip[0], self.position_clip[1],
        )
        result[:, 3:6] = np.clip(
            action[:, 3:6] * self.rotation_scale + self.rotation_offset,
            self.rotation_clip[0], self.rotation_clip[1],
        )
        result[:, 6:7] = np.clip(
            action[:, 6:7] * self.gripper_scale + self.gripper_offset,
            self.gripper_clip[0], self.gripper_clip[1],
        )
        return result.reshape(-1) if squeeze else result

    def transform_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        converted = self.transform(chunk)
        if self.source_frequency != self.target_frequency:
            return self._resample(converted)
        return converted

    def config_dict(self) -> dict[str, Any]:
        base = {
            "type": "LiberoToVJEPAActionAdapter",
            "version": self.version,
            "position_scale": self.position_scale.tolist(),
            "position_offset": self.position_offset.tolist(),
            "position_clip": list(self.position_clip),
            "rotation_scale": self.rotation_scale.tolist(),
            "rotation_offset": self.rotation_offset.tolist(),
            "rotation_clip": list(self.rotation_clip),
            "gripper_scale": self.gripper_scale,
            "gripper_offset": self.gripper_offset,
            "gripper_clip": list(self.gripper_clip),
            "source_frequency": self.source_frequency,
            "target_frequency": self.target_frequency,
        }
        return {**base, "adapter_hash": getattr(self, "_hash", "")}

    def save(self, path: str) -> None:
        import pathlib as _pl
        _pl.Path(path).write_text(
            json.dumps(self.config_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "LiberoToVJEPAActionAdapter":
        import pathlib as _pl
        cfg = json.loads(_pl.Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in cfg.items()
                       if k not in ("type", "adapter_hash")})

    @property
    def adapter_hash(self) -> str:
        return self._hash

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        t = np.arange(chunk.shape[0]) / self.source_frequency
        t_new = np.arange(0, t[-1] + 1e-12, 1.0 / self.target_frequency)
        out = np.zeros((len(t_new), chunk.shape[1]), dtype=np.float32)
        for d in range(chunk.shape[1]):
            out[:, d] = np.interp(t_new, t, chunk[:, d])
        return out

    def _compute_hash(self) -> str:
        cfg = {k: v for k, v in self.config_dict().items() if k != "adapter_hash"}
        raw = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def create_default_libero_adapter() -> LiberoToVJEPAActionAdapter:
    return LiberoToVJEPAActionAdapter(version="v1-default")

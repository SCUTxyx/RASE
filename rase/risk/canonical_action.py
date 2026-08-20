"""Canonical action representation for multi-VLA risk prediction.

Every VLA adapter must produce a CanonicalActionChunk that the shared risk core
consumes.  This isolates the risk model from VLA-specific conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CanonicalActionChunk:
    delta_position: torch.Tensor   # (H, 3), robot/base frame frozen convention
    rotation_6d: torch.Tensor      # (H, 6), 6D rotation representation
    gripper: torch.Tensor          # (H, 1), normalized open/close
    delta_time: torch.Tensor       # (H, 1), time delta per step
    valid_mask: torch.Tensor       # (H,), 1.0 = valid step
    source_uncertainty: torch.Tensor | None = None  # (H,), optional

    def to(self, device: torch.device) -> "CanonicalActionChunk":
        return CanonicalActionChunk(
            delta_position=self.delta_position.to(device),
            rotation_6d=self.rotation_6d.to(device),
            gripper=self.gripper.to(device),
            delta_time=self.delta_time.to(device),
            valid_mask=self.valid_mask.to(device),
            source_uncertainty=(
                self.source_uncertainty.to(device)
                if self.source_uncertainty is not None else None
            ),
        )

    def flatten(self) -> torch.Tensor:
        """Concatenate into a single (H, D) feature tensor."""
        parts = [self.delta_position, self.rotation_6d, self.gripper, self.delta_time]
        return torch.cat(parts, dim=-1)

    @property
    def horizon(self) -> int:
        return self.delta_position.shape[0]

    @classmethod
    def from_numpy(
        cls,
        position: np.ndarray,
        rotation_6d: np.ndarray,
        gripper: np.ndarray,
        delta_time: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> "CanonicalActionChunk":
        H = len(position)
        dt = delta_time if delta_time is not None else np.full((H, 1), 0.1)
        vm = valid_mask if valid_mask is not None else np.ones(H)
        return cls(
            delta_position=torch.from_numpy(np.asarray(position, dtype=np.float32)),
            rotation_6d=torch.from_numpy(np.asarray(rotation_6d, dtype=np.float32)),
            gripper=torch.from_numpy(np.asarray(gripper, dtype=np.float32).reshape(-1, 1)),
            delta_time=torch.from_numpy(np.asarray(dt, dtype=np.float32).reshape(-1, 1)),
            valid_mask=torch.from_numpy(np.asarray(vm, dtype=np.float32)),
        )


def summary_from_chunk(chunk: CanonicalActionChunk) -> torch.Tensor:
    """Produce a fixed-size summary vector from a variable-length chunk."""
    pos = chunk.delta_position * chunk.valid_mask.unsqueeze(-1)
    rot = chunk.rotation_6d * chunk.valid_mask.unsqueeze(-1)
    g = chunk.gripper.squeeze(-1) * chunk.valid_mask
    # torch.std defaults to the unbiased estimator, which is undefined for a
    # one-action OFT chunk.  The previous nan_to_num produced the right value
    # (zero) but emitted thousands of warnings during nested CV.
    pos_std = torch.zeros_like(pos[0]) if pos.shape[0] <= 1 else pos.std(0)
    rot_std = torch.zeros_like(rot[0]) if rot.shape[0] <= 1 else rot.std(0)
    g_std = torch.zeros((), dtype=g.dtype, device=g.device) if g.shape[0] <= 1 else g.std()
    return torch.cat([
        pos.mean(0),
        pos_std,
        rot.mean(0),
        rot_std,
        g.mean().unsqueeze(0),
        g_std.unsqueeze(0),
    ], dim=0)

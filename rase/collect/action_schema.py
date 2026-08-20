"""Action-schema validation for Route C pipeline.

Ensures action dimensionality and component ordering (translation / rotation /
gripper) are consistent across all scripts that produce or consume actions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


# Canonical 7-dim action layout used throughout Route C:
#   [dx, dy, dz, droll, dpitch, dyaw, gripper]
ACTION_DIM = 7
TRANSLATION_DIMS = slice(0, 3)   # dx, dy, dz
ROTATION_DIMS = slice(3, 6)      # droll, dpitch, dyaw
GRIPPER_DIM = 6                  # gripper open/close


def validate_action_shape(action: np.ndarray, *, dim: int = ACTION_DIM) -> np.ndarray:
    """Ensure action has the expected dimensionality.

    Accepts (dim,) or (1, dim).  Returns a flat (dim,) array.
    """
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape == (dim,):
        return arr
    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr.flatten()
    raise ValueError(
        f"expected action shape ({dim},) or (1, {dim}), got {arr.shape}"
    )


def compute_action_delta(
    teacher_action: np.ndarray,
    student_action: np.ndarray,
    *,
    delta_clip: float = 0.5,
) -> np.ndarray:
    """Compute clipped action delta: clip(aT - aS, -delta_clip, +delta_clip)."""
    aT = validate_action_shape(teacher_action)
    aS = validate_action_shape(student_action)
    raw = aT - aS
    return np.clip(raw, -delta_clip, delta_clip)


def compute_action_norm(actions: np.ndarray, *, dim: int = ACTION_DIM) -> np.ndarray:
    """L2 norm per action (last dim)."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        return float(np.linalg.norm(arr))
    return np.linalg.norm(arr, axis=-1)


def split_action_components(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 7-dim action into translation (3), rotation (3), gripper (1)."""
    a = validate_action_shape(action)
    return {
        "translation": a[TRANSLATION_DIMS],
        "rotation": a[ROTATION_DIMS],
        "gripper": a[GRIPPER_DIM],
    }


def verify_dim_order(action: np.ndarray, *, dim: int = ACTION_DIM) -> bool:
    """Verify action has the correct number of components (no order check)."""
    a = np.asarray(action, dtype=np.float32)
    return a.shape[-1] == dim


def action_schema_hash(*, dim: int = ACTION_DIM,
                       translation_range: str = "0:3",
                       rotation_range: str = "3:6",
                       gripper_index: str = "6") -> str:
    """Return a deterministic hash of the action schema for metadata tagging."""
    import hashlib
    payload = json.dumps({
        "dim": dim,
        "translation": translation_range,
        "rotation": rotation_range,
        "gripper": gripper_index,
        "dtype": "float32",
        "normalized": True,
        "range": "[-1, 1]",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

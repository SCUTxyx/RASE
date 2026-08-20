"""Outcome-free behavior descriptors for adapting a shared risk core.

Descriptors use only quantities available while a source VLA proposes actions:
canonical action summaries, proprioception, and RGB channel moments.  They do
not use success/failure, fallback outcomes, task IDs, teacher cost, or future
frames.  Statistics must be fit on outer-training tasks, or on a separately
declared unlabeled calibration set for a held-out VLA.
"""

from __future__ import annotations

import numpy as np


DESCRIPTOR_VERSION = "rase-multi-vla-outcome-free-behavior/v1"


def behavior_descriptor(image: np.ndarray, proprio: np.ndarray,
                        action_summary: np.ndarray) -> np.ndarray:
    """Return a fixed 80-D outcome-free policy behavior descriptor.

    Components are mean/std of 20-D canonical actions (40), mean/std of 8-D
    proprioception (16), and mean/std RGB intensity for two views (12+12).
    Images may be uint8 or normalized floats and are normalized to [0,1].
    """
    image = np.asarray(image)
    proprio = np.asarray(proprio, dtype=np.float32)
    action = np.asarray(action_summary, dtype=np.float32)
    if image.ndim != 5 or image.shape[1:3] != (2, 3):
        raise ValueError(f"expected image [N,2,3,H,W], got {image.shape}")
    if proprio.ndim != 2 or proprio.shape[1] != 8:
        raise ValueError(f"expected proprio [N,8], got {proprio.shape}")
    if action.ndim != 2 or action.shape[1] != 20:
        raise ValueError(f"expected action_summary [N,20], got {action.shape}")
    if not (len(image) == len(proprio) == len(action)) or len(image) == 0:
        raise ValueError("descriptor inputs must have the same non-zero row count")
    pixels = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer) or float(pixels.max()) > 1.5:
        pixels /= 255.0
    channel_mean_per_row = pixels.mean(axis=(-1, -2)).reshape(len(image), -1)
    channel_std_per_row = pixels.std(axis=(-1, -2)).reshape(len(image), -1)
    pieces = [
        action.mean(0), action.std(0),
        proprio.mean(0), proprio.std(0),
        channel_mean_per_row.mean(0), channel_mean_per_row.std(0),
        channel_std_per_row.mean(0), channel_std_per_row.std(0),
    ]
    value = np.concatenate(pieces).astype(np.float32)
    if value.shape != (80,) or not np.isfinite(value).all():
        raise ValueError("invalid behavior descriptor")
    return value


def descriptors_by_policy(data: dict[str, np.ndarray], indices: np.ndarray,
                          *, minimum_rows: int = 8) -> dict[str, np.ndarray]:
    """Fit one outcome-free descriptor per policy from allowed rows only."""
    indices = np.asarray(indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for policy in sorted(set(data["policy_id"][indices].tolist())):
        selected = indices[data["policy_id"][indices] == policy]
        if len(selected) < minimum_rows:
            raise ValueError(f"policy {policy} has only {len(selected)} descriptor rows")
        result[str(policy)] = behavior_descriptor(
            data["image"][selected], data["proprio"][selected],
            data["action_summary"][selected],
        )
    return result

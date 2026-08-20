"""Canonical observation hashing for parity audits.

Provides component-level hashing that is robust to dict key ordering,
float padding, and non-contiguous memory layouts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def canonical_obs_hash(obs: dict[str, Any]) -> dict[str, Any]:
    """Return a component-level canonical hash of a LIBERO observation dict.

    Components:
      - image: SHA256 of contiguous uint8 image bytes → "exact" / "missing"
      - proprio: list of float32 values → numpy array
      - sim_state: SHA256 of MuJoCo state bytes if available
      - language: SHA256 of token_ids bytes if available
      - step_index: int

    The output dict is JSON-serializable (strings for hashes, lists for arrays).
    """
    result: dict[str, Any] = {}

    # --- image ---
    image = obs.get("agentview_image") or obs.get("image")
    if image is not None:
        img_arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        result["image_hash"] = _bytes_hash(img_arr.tobytes())
        result["image_shape"] = list(img_arr.shape)
    else:
        result["image_hash"] = "missing"

    # Also hash robot_0_eye_in_hand if present
    wrist = obs.get("robot0_eye_in_hand_image") or obs.get("wrist_image")
    if wrist is not None:
        wrist_arr = np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8))
        result["wrist_image_hash"] = _bytes_hash(wrist_arr.tobytes())

    # --- proprio ---
    proprio_keys = [
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
        "robot0_joint_pos", "robot0_joint_vel",
    ]
    proprio_vals: list[float] = []
    for key in proprio_keys:
        val = obs.get(key)
        if val is not None:
            val_f = np.asarray(val, dtype=np.float32).flatten()
            proprio_vals.extend(val_f.tolist())
    result["proprio"] = [float(v) for v in proprio_vals]
    result["proprio_dim"] = len(proprio_vals)

    # --- sim state ---
    sim_state = obs.get("_sim_state_bytes") or obs.get("sim_state")
    if sim_state is not None:
        if isinstance(sim_state, bytes):
            result["sim_state_hash"] = _bytes_hash(sim_state)
        else:
            result["sim_state_hash"] = _bytes_hash(np.ascontiguousarray(
                np.asarray(sim_state, dtype=np.uint8)).tobytes())
    else:
        result["sim_state_hash"] = "missing"

    # --- language ---
    token_ids = obs.get("token_ids") or obs.get("language_tokens")
    if token_ids is not None:
        tok = np.asarray(token_ids, dtype=np.int64)
        result["language_hash"] = _bytes_hash(tok.tobytes())
        result["language_len"] = int(tok.size)

    # --- step ---
    result["step_index"] = int(obs.get("step_index", -1))

    return result


def obs_hash_equal(h1: dict[str, Any], h2: dict[str, Any]) -> bool:
    """Return True if all hash components are equal."""
    diff = obs_hash_diff(h1, h2)
    return len(diff.get("differences", [])) == 0


def obs_hash_diff(h1: dict[str, Any], h2: dict[str, Any]) -> dict[str, Any]:
    """Return per-component comparison of two canonical observation hashes."""
    differences: list[dict[str, Any]] = []
    all_keys = sorted(set(h1.keys()) | set(h2.keys()))

    for key in all_keys:
        v1 = h1.get(key)
        v2 = h2.get(key)
        if key.endswith("_hash"):
            if v1 != v2:
                differences.append({"component": key, "type": "hash_mismatch",
                                    "h1": v1, "h2": v2})
        elif key == "proprio" and isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                differences.append({"component": key, "type": "size_mismatch",
                                    "len1": len(v1), "len2": len(v2)})
            elif len(v1) > 0:
                arr1 = np.array(v1, dtype=np.float32)
                arr2 = np.array(v2, dtype=np.float32)
                max_abs = float(np.max(np.abs(arr1 - arr2)))
                differences.append({"component": key, "type": "max_abs_diff",
                                    "value": max_abs,
                                    "exact": bool(max_abs == 0.0)})
        elif key in ("proprio_dim", "step_index", "language_len"):
            if v1 != v2:
                differences.append({"component": key, "type": "value_mismatch",
                                    "v1": v1, "v2": v2})
        elif key in ("image_shape",):
            if v1 != v2:
                differences.append({"component": key, "type": "shape_mismatch",
                                    "s1": v1, "s2": v2})
        else:
            if v1 != v2:
                differences.append({"component": key, "type": "unknown_mismatch"})

    return {"differences": differences, "equal": len(differences) == 0}


def action_max_abs_diff(a1: np.ndarray, a2: np.ndarray) -> float:
    """Maximum absolute difference between two action arrays."""
    return float(np.max(np.abs(np.asarray(a1, dtype=np.float32) -
                                np.asarray(a2, dtype=np.float32))))

"""Versioned observation/action schema for the OFT oracle adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PREPROCESS_REVISION = "libero_eval_v1"
WIRE_SCHEMA_VERSION = 1


class WireSchemaError(ValueError):
    pass


def validate_predict_inputs(
    arrays: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
) -> tuple[int, list[str], str]:
    """Validate RPC arrays/payload; return ``(batch, instructions, proprio_format)``."""
    required = ("agentview", "wrist", "proprio")
    for name in required:
        if name not in arrays:
            raise WireSchemaError(f"missing array {name!r}")
    agentview = np.asarray(arrays["agentview"])
    wrist = np.asarray(arrays["wrist"])
    proprio = np.asarray(arrays["proprio"])
    if agentview.ndim != 4 or agentview.shape[-1] != 3:
        raise WireSchemaError(f"agentview must be [B,H,W,3], got {agentview.shape}")
    if wrist.shape != agentview.shape:
        raise WireSchemaError(
            f"wrist shape {wrist.shape} must match agentview {agentview.shape}"
        )
    if agentview.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise WireSchemaError("agentview/wrist must be uint8")
    batch = int(agentview.shape[0])
    proprio_format = str(payload.get("proprio_format", "policy_state"))
    if proprio_format == "policy_state":
        if proprio.shape != (batch, 8):
            raise WireSchemaError(
                f"policy_state proprio must be [B,8], got {proprio.shape}"
            )
    elif proprio_format == "raw_quat":
        if proprio.shape != (batch, 9):
            raise WireSchemaError(
                f"raw_quat proprio must be [B,9] "
                f"(pos3+quat4+grip2), got {proprio.shape}"
            )
    else:
        raise WireSchemaError(
            "proprio_format must be 'policy_state' or 'raw_quat'"
        )
    instructions = payload.get("instructions")
    if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
        raise WireSchemaError("payload.instructions must be a list of strings")
    if len(instructions) != batch:
        raise WireSchemaError(
            f"instructions length {len(instructions)} != batch {batch}"
        )
    if not all(isinstance(item, str) and item for item in instructions):
        raise WireSchemaError("each instruction must be a non-empty string")
    max_batch = int(payload.get("max_batch", 8))
    if batch > max_batch:
        raise WireSchemaError(f"batch {batch} exceeds max_batch {max_batch}")
    return batch, list(instructions), proprio_format


def _quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite-compatible quaternion (x,y,z,w) → axis-angle (no OFT import)."""
    import math

    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def proprio_to_policy_state(
    proprio: np.ndarray, *, proprio_format: str = "policy_state"
) -> np.ndarray:
    """Return OFT policy state ``[pos3, axisangle3, grip2]``."""
    value = np.asarray(proprio, dtype=np.float64).reshape(-1)
    if proprio_format == "policy_state":
        if value.shape[0] != 8:
            raise WireSchemaError(f"expected proprio length 8, got {value.shape[0]}")
        return value.astype(np.float32)
    if proprio_format != "raw_quat":
        raise WireSchemaError(f"unsupported proprio_format {proprio_format!r}")
    if value.shape[0] != 9:
        raise WireSchemaError(f"expected raw proprio length 9, got {value.shape[0]}")
    pos = value[:3]
    quat = value[3:7]
    grip = value[7:9]
    axis = _quat_xyzw_to_axisangle(quat)
    return np.concatenate(
        [pos.astype(np.float32), axis, grip.astype(np.float32)]
    )

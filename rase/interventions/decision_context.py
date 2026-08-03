"""Decision-context v2 helpers for strict same-state intervention rollouts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DECISION_CONTEXT_SCHEMA_VERSION = "rase-decision-context/v2"
ACTION_SPACE = "libero-env-action/v1"


def action_suffix_sha256(actions: np.ndarray) -> str:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"active action suffix must be [T, 7], got {array.shape}")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_decision_context(
    *,
    source_policy: str,
    snapshot_env_step: int,
    action_chunk_size: int,
    action_chunk_offset: int,
    active_action_suffix: np.ndarray,
    public_action_history: Sequence[Sequence[float]] = (),
    public_proprio_history: Sequence[Sequence[float]] = (),
    public_observation_history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    suffix = np.asarray(active_action_suffix, dtype=np.float32)
    if not source_policy:
        raise ValueError("source_policy must be non-empty")
    if snapshot_env_step < 0:
        raise ValueError("snapshot_env_step must be non-negative")
    if action_chunk_size < 2:
        raise ValueError("action_chunk_size must be at least two")
    if not 0 < action_chunk_offset < action_chunk_size:
        raise ValueError("action_chunk_offset must be strictly inside the active chunk")
    expected = action_chunk_size - action_chunk_offset
    if suffix.shape != (expected, 7):
        raise ValueError(
            "active suffix length does not match chunk_size - offset: "
            f"expected {(expected, 7)}, got {suffix.shape}"
        )
    return {
        "schema_version": DECISION_CONTEXT_SCHEMA_VERSION,
        "source_policy": source_policy,
        "snapshot_env_step": int(snapshot_env_step),
        "action_chunk_size": int(action_chunk_size),
        "action_chunk_offset": int(action_chunk_offset),
        "active_action_suffix": suffix.copy(),
        "active_action_suffix_space": ACTION_SPACE,
        "active_action_suffix_sha256": action_suffix_sha256(suffix),
        "source_rollout_suffix_parity": {
            "status": "pending",
            "observed_steps": 0,
            "max_abs_error": None,
            "atol": 1e-6,
        },
        "public_action_history": np.asarray(
            public_action_history, dtype=np.float32
        ),
        "public_proprio_history": np.asarray(
            public_proprio_history, dtype=np.float32
        ),
        "public_observation_history": [dict(item) for item in public_observation_history],
    }


def finalize_source_suffix_parity(
    value: dict[str, Any], observed_actions: np.ndarray, *, atol: float = 1e-6
) -> None:
    expected = np.asarray(value["active_action_suffix"], dtype=np.float32)
    observed = np.asarray(observed_actions, dtype=np.float32)
    if observed.shape != expected.shape:
        raise ValueError(
            f"source suffix parity shape mismatch: {observed.shape} != {expected.shape}"
        )
    max_abs_error = float(np.max(np.abs(observed - expected))) if expected.size else 0.0
    passed = bool(max_abs_error <= atol)
    value["source_rollout_suffix_parity"] = {
        "status": "passed" if passed else "failed",
        "observed_steps": len(observed),
        "max_abs_error": max_abs_error,
        "atol": float(atol),
    }
    if not passed:
        raise ValueError(
            f"source suffix parity failed: max_abs_error={max_abs_error:.3e}"
        )


def validate_decision_context(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != DECISION_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported decision-context schema {value.get('schema_version')!r}"
        )
    if value.get("active_action_suffix_space") != ACTION_SPACE:
        raise ValueError("strict CONTINUE requires env-space LIBERO actions")
    chunk_size = int(value["action_chunk_size"])
    offset = int(value["action_chunk_offset"])
    suffix = np.asarray(value["active_action_suffix"], dtype=np.float32)
    if not 0 < offset < chunk_size or suffix.shape != (chunk_size - offset, 7):
        raise ValueError("decision context contains an inconsistent active suffix")
    expected = str(value["active_action_suffix_sha256"])
    if action_suffix_sha256(suffix) != expected:
        raise ValueError("active action suffix checksum mismatch")
    if int(value["snapshot_env_step"]) % chunk_size != offset:
        raise ValueError("snapshot_env_step is inconsistent with chunk offset")
    parity = value.get("source_rollout_suffix_parity")
    if not isinstance(parity, Mapping) or parity.get("status") != "passed":
        raise ValueError("active suffix lacks passed source-rollout parity")
    if int(parity.get("observed_steps", -1)) != len(suffix):
        raise ValueError("source-rollout parity did not cover the complete suffix")
    if float(parity.get("max_abs_error", float("inf"))) > float(
        parity.get("atol", 0.0)
    ):
        raise ValueError("source-rollout parity error exceeds tolerance")


def strict_continue_suffix(controller_state: Mapping[str, Any]) -> np.ndarray:
    context = controller_state.get("decision_context")
    if not isinstance(context, Mapping):
        raise ValueError("state has no decision_context; strict CONTINUE is infeasible")
    validate_decision_context(context)
    return np.asarray(context["active_action_suffix"], dtype=np.float32).copy()

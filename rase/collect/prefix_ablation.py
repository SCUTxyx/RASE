"""Named prefix interventions for attributing continuation-policy recovery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrefixArm:
    """One explicit action-prefix intervention before a continuation policy."""

    label: str
    kind: str
    actions: np.ndarray
    candidate_index: int | None = None


def action_prefix_sha256(actions: Any) -> str:
    array = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"action prefix must have shape [T,7], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("action prefix must be finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_decision_suffix_arms(active_action_suffix: Any) -> tuple[PrefixArm, ...]:
    """Return immediate and suffix-preserving OFT switch profiles."""
    suffix = np.asarray(active_action_suffix, dtype=np.float32)
    action_prefix_sha256(suffix)
    if len(suffix) < 1:
        raise ValueError("active action suffix must be non-empty")
    return (
        PrefixArm("direct_oft", "direct", np.empty((0, 7), dtype=np.float32)),
        PrefixArm("decision_suffix_oft", "decision_suffix", suffix.copy()),
    )


def build_decision_suffix_prefix_arms(
    active_action_suffix: Any,
) -> tuple[PrefixArm, ...]:
    """Return every causal prefix length of one frozen active action suffix."""
    suffix = np.asarray(active_action_suffix, dtype=np.float32)
    action_prefix_sha256(suffix)
    if len(suffix) < 1:
        raise ValueError("active action suffix must be non-empty")
    return tuple(
        PrefixArm(
            f"suffix_prefix_{steps}",
            "decision_suffix_prefix",
            suffix[:steps].copy(),
            steps,
        )
        for steps in range(len(suffix) + 1)
    )


def summarize_decision_suffix_state(
    state_key: str,
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_label = {str(record["arm_label"]): record for record in records}
    if len(by_label) != len(records) or set(by_label) != {
        "direct_oft",
        "decision_suffix_oft",
    }:
        raise ValueError(f"incomplete decision-suffix arms for {state_key}")
    direct = bool(by_label["direct_oft"]["success"])
    deferred = bool(by_label["decision_suffix_oft"]["success"])
    classification = {
        (False, False): "neither",
        (True, False): "direct_only",
        (False, True): "deferred_only",
        (True, True): "both",
    }[(direct, deferred)]
    return {
        "state_key": state_key,
        **dict(metadata or {}),
        "classification": classification,
        "direct_oft_success": direct,
        "decision_suffix_oft_success": deferred,
        "arms": records,
    }


def summarize_decision_suffix_prefix_state(
    state_key: str,
    records: list[dict[str, Any]],
    *,
    expected_suffix_steps: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and summarize a complete k=0..T active-suffix prefix grid."""
    if expected_suffix_steps < 1:
        raise ValueError("expected_suffix_steps must be positive")
    by_steps: dict[int, dict[str, Any]] = {}
    for record in records:
        steps = int(record["prefix_steps"])
        expected_label = f"suffix_prefix_{steps}"
        if record.get("arm_label") != expected_label:
            raise ValueError(
                f"prefix label/length mismatch for {state_key}: "
                f"{record.get('arm_label')} != {expected_label}"
            )
        if steps in by_steps:
            raise ValueError(f"duplicate prefix length {steps} for {state_key}")
        if not bool(record.get("prefix_completed")):
            raise ValueError(f"incomplete prefix length {steps} for {state_key}")
        by_steps[steps] = record
    expected = set(range(expected_suffix_steps + 1))
    if set(by_steps) != expected:
        raise ValueError(
            f"incomplete suffix-prefix grid for {state_key}: "
            f"expected={sorted(expected)} observed={sorted(by_steps)}"
        )
    successes = [bool(by_steps[steps]["success"]) for steps in sorted(expected)]
    flips = [
        steps
        for steps in range(1, expected_suffix_steps + 1)
        if successes[steps] != successes[steps - 1]
    ]
    return {
        "state_key": state_key,
        **dict(metadata or {}),
        "success_pattern": "".join("1" if value else "0" for value in successes),
        "success_by_prefix_steps": {
            str(steps): successes[steps] for steps in sorted(expected)
        },
        "n_success_flips": len(flips),
        "success_flip_steps": flips,
        "single_transition": len(flips) == 1,
        "arms": [by_steps[steps] for steps in sorted(expected)],
    }


def build_prefix_arms(candidate_actions: Any) -> tuple[PrefixArm, ...]:
    """Return direct, time-matched zero, and every frozen candidate prefix."""
    array = np.asarray(candidate_actions, dtype=np.float32)
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] < 1
        or array.shape[2] != 7
    ):
        raise ValueError(f"candidate actions must have shape [K,T,7], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate actions must be finite")
    arms = [
        PrefixArm("direct_oft", "direct", np.empty((0, 7), dtype=np.float32)),
        PrefixArm(
            f"zero_{array.shape[1]}",
            "zero",
            np.zeros((array.shape[1], 7), dtype=np.float32),
        ),
    ]
    arms.extend(
        PrefixArm(f"candidate_{index}", "candidate", chunk.copy(), index)
        for index, chunk in enumerate(array)
    )
    return tuple(arms)


def classify_prefix_ablation(arm_success: dict[str, bool]) -> str:
    """Assign a conservative mechanism label from named deterministic arms."""
    required = {"direct_oft"}
    zero_labels = [label for label in arm_success if label.startswith("zero_")]
    candidate_labels = [label for label in arm_success if label.startswith("candidate_")]
    if not required.issubset(arm_success) or len(zero_labels) != 1 or not candidate_labels:
        raise ValueError("ablation requires direct_oft, one zero arm, and candidate arms")
    direct = bool(arm_success["direct_oft"])
    zero = bool(arm_success[zero_labels[0]])
    candidate_hits = sum(bool(arm_success[label]) for label in candidate_labels)
    if direct:
        if candidate_hits == len(candidate_labels):
            return "continuation_sufficient_candidate_invariant"
        return "continuation_sufficient_candidate_harm_possible"
    if zero:
        return "passive_prefix_sufficient"
    if candidate_hits:
        return "candidate_specific_rescue"
    return "unrecovered"


def summarize_prefix_state(
    state_key: str,
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one complete intervention grid and summarize its mechanism."""
    if not records:
        raise ValueError("records must be non-empty")
    labels = [str(record["arm_label"]) for record in records]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate arm labels for {state_key}")
    arm_success = {
        label: bool(record["success"])
        for label, record in zip(labels, records)
    }
    candidate_hits = sum(
        value for label, value in arm_success.items() if label.startswith("candidate_")
    )
    return {
        "state_key": state_key,
        **dict(metadata or {}),
        "classification": classify_prefix_ablation(arm_success),
        "direct_oft_success": arm_success["direct_oft"],
        "zero_prefix_success": next(
            value for label, value in arm_success.items() if label.startswith("zero_")
        ),
        "candidate_hits": int(candidate_hits),
        "candidate_trials": sum(label.startswith("candidate_") for label in labels),
        "arms": records,
    }


def aggregate_prefix_summaries(
    state_keys: list[str], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Combine suite-specific summaries and require an exact state-key union."""
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.get("schema_version") != "rase-oft-prefix-ablation/v1":
            raise ValueError("unexpected prefix-ablation schema")
        if summary.get("status") != "complete":
            raise ValueError("prefix-ablation summary is incomplete")
        rows.extend(dict(row) for row in summary.get("per_state") or [])
    observed = [str(row.get("state_key")) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate states across prefix-ablation summaries")
    if set(observed) != set(state_keys):
        raise ValueError(
            f"prefix-ablation state union mismatch: expected={sorted(state_keys)} "
            f"observed={sorted(observed)}"
        )
    labels = sorted({str(row["classification"]) for row in rows})
    return {
        "schema_version": "rase-oft-prefix-ablation-matrix/v1",
        "status": "complete",
        "n_states": len(rows),
        "classification_counts": {
            label: sum(row["classification"] == label for row in rows)
            for label in labels
        },
        "candidate_specific_rescue_states": sorted(
            str(row["state_key"])
            for row in rows
            if row["classification"] == "candidate_specific_rescue"
        ),
        "per_state": sorted(rows, key=lambda row: str(row["state_key"])),
    }

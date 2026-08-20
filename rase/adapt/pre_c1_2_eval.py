"""Shared PRE-C1.2 eval helpers (failure cohort, proprio, action spaces)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_pre_c0_failure_keys(rollout_dir: Path | str) -> list[dict[str, Any]]:
    """Load PRE-C0 current_suffix failure rows (locked 9-cohort source)."""

    root = Path(rollout_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        if bool(payload.get("family_success", {}).get("current_suffix")):
            continue
        rows.append(payload)
    return rows


def mae_per_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.abs(a[:n] - b[:n])


def sign_agreement_per_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return (np.sign(a[:n]) == np.sign(b[:n])).astype(np.float64)


def clipping_rate_per_dim(action: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    return ((action <= low + 1e-6) | (action >= high - 1e-6)).astype(np.float64)


def extract_proprio_vector(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Best-effort proprio extraction for successor distance."""

    out: dict[str, np.ndarray] = {}
    robot_state = observation.get("robot_state")
    if isinstance(robot_state, dict):
        for key in ("ee_pos", "ee_ori", "ee_quat", "joint_positions", "gripper", "gripper_qpos"):
            # Nested search.
            found = _find_leaf(robot_state, key)
            if found is None or isinstance(found, Mapping):
                continue
            # If leaf is nested container, flatten numeric children instead.
            if isinstance(found, (list, tuple)) and found and isinstance(found[0], Mapping):
                continue
            try:
                arr = np.asarray(found, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                flat_leaf = _flatten_numeric(found) if isinstance(found, Mapping) else np.asarray([])
                if flat_leaf.size == 0:
                    continue
                arr = flat_leaf
            if arr.size and np.issubdtype(arr.dtype, np.number):
                out[key] = arr
        # Flatten all numeric leaves as fallback joint-ish vector.
        flat = _flatten_numeric(robot_state)
        if flat.size:
            out["robot_state_flat"] = flat
    agent_pos = observation.get("agent_pos")
    if agent_pos is not None and not isinstance(agent_pos, Mapping):
        try:
            out["agent_pos"] = np.asarray(agent_pos, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            pass
    return out


def successor_distance(
    obs_a: Mapping[str, Any],
    obs_b: Mapping[str, Any],
) -> dict[str, float]:
    pa = extract_proprio_vector(obs_a)
    pb = extract_proprio_vector(obs_b)
    metrics: dict[str, float] = {}
    for key in sorted(set(pa) & set(pb)):
        metrics[f"{key}_l2"] = float(np.linalg.norm(pa[key] - pb[key]))
    # Preferred aggregate used for interface decision.
    if "ee_pos_l2" in metrics:
        metrics["aggregate_l2"] = metrics["ee_pos_l2"]
    elif "agent_pos_l2" in metrics:
        metrics["aggregate_l2"] = metrics["agent_pos_l2"]
    elif "robot_state_flat_l2" in metrics:
        metrics["aggregate_l2"] = metrics["robot_state_flat_l2"]
    else:
        metrics["aggregate_l2"] = float("nan")
    return metrics


def action_space_report(
    *,
    student_normalized: np.ndarray | None,
    teacher_normalized: np.ndarray | None,
    student_denormalized: np.ndarray | None,
    teacher_denormalized: np.ndarray | None,
    student_env: np.ndarray,
    teacher_env: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "env_action_mae_per_dim": mae_per_dim(student_env, teacher_env).tolist(),
        "env_action_mae": float(np.mean(mae_per_dim(student_env, teacher_env))),
        "sign_agreement_per_dim": sign_agreement_per_dim(student_env, teacher_env).tolist(),
        "clipping_rate_student_per_dim": clipping_rate_per_dim(student_env).tolist(),
        "clipping_rate_teacher_per_dim": clipping_rate_per_dim(teacher_env).tolist(),
    }
    if student_normalized is not None and teacher_normalized is not None:
        report["normalized_action_mae_per_dim"] = mae_per_dim(
            student_normalized, teacher_normalized
        ).tolist()
        report["normalized_action_mae"] = float(
            np.mean(mae_per_dim(student_normalized, teacher_normalized))
        )
    if student_denormalized is not None and teacher_denormalized is not None:
        report["denormalized_action_mae_per_dim"] = mae_per_dim(
            student_denormalized, teacher_denormalized
        ).tolist()
        report["denormalized_action_mae"] = float(
            np.mean(mae_per_dim(student_denormalized, teacher_denormalized))
        )
    return report


def _find_leaf(node: Any, key: str) -> Any | None:
    if isinstance(node, Mapping):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_leaf(value, key)
            if found is not None:
                return found
    return None


def _flatten_numeric(node: Any) -> np.ndarray:
    values: list[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for value in item.values():
                walk(value)
        else:
            array = np.asarray(item)
            if np.issubdtype(array.dtype, np.number):
                values.extend(array.astype(np.float64).reshape(-1).tolist())

    walk(node)
    return np.asarray(values, dtype=np.float64)


def summarize_horizon_sweep(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_h: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_h.setdefault(int(row["horizon"]), []).append(row)
    summary = {}
    for h, group in sorted(by_h.items()):
        summary[str(h)] = {
            "n": len(group),
            "base_successes": sum(bool(r.get("base_success")) for r in group),
            "adapted_successes": sum(bool(r.get("adapted_success")) for r in group),
            "invariant_pass_rate": float(
                np.mean([bool(r.get("invariant_passed", True)) for r in group])
            ),
        }
    return summary

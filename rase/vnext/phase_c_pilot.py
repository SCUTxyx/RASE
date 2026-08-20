"""Leakage-resistant utilities for the A-PARTIAL single-policy Phase C pilot."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .motion_trace import MotionSemanticMap, action_to_motion_trace
from .schema import CanonicalActionToken


SOURCE_OPERATORS = (
    "continue.source", "requery.source", "resample.source",
)


def stable_seed(*parts: object) -> int:
    token = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def pad_action_chunk(
    actions: Sequence[np.ndarray], *, horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a non-empty sequence of 7-D env actions without inventing steps."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7 or not 1 <= len(array) <= horizon:
        raise ValueError(f"actions must have shape [1..{horizon},7], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("actions must be finite")
    padded = np.zeros((horizon, 7), dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.bool_)
    padded[: len(array)] = array
    mask[: len(array)] = True
    return padded, mask


def choose_tasks(
    jobs: Sequence[Mapping[str, Any]], *, tasks_per_suite: int = 0,
) -> list[str]:
    """Outcome-independent deterministic task cohort from frozen metadata."""
    by_suite: dict[str, set[str]] = defaultdict(set)
    for job in jobs:
        by_suite[str(job["suite"])].add(str(job["task_id"]))
    selected: list[str] = []
    for suite in sorted(by_suite):
        tasks = sorted(by_suite[suite])
        selected.extend(tasks if tasks_per_suite <= 0 else tasks[:tasks_per_suite])
    return selected


def trace_feature_vector(
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    semantics: tuple[str, ...],
    policy_id: str,
    semantic_map: MotionSemanticMap,
    control_hz: float = 10.0,
) -> np.ndarray:
    """Fixed-size physical feature vector; masked steps never enter statistics."""
    token = CanonicalActionToken.from_array(
        np.asarray(actions, dtype=np.float32),
        semantics=semantics,
        control_hz=control_hz,
        coordinate_frame="robot.base.relative",
        policy_id=policy_id,
        step_mask=np.asarray(mask, dtype=np.bool_),
    )
    trace = action_to_motion_trace(token, semantic_map=semantic_map)
    valid = trace.valid_mask
    if not valid.any():
        raise ValueError("trace has no valid steps")
    delta = trace.ee_delta[valid].astype(np.float64)
    velocity = trace.velocity[valid].astype(np.float64)
    acceleration = trace.acceleration[valid].astype(np.float64)
    jerk = trace.jerk[valid].astype(np.float64)
    pose = trace.integrated_ee_pose_rel[np.flatnonzero(valid)[-1]].astype(np.float64)
    gripper = trace.gripper_state[trace.gripper_valid_mask].astype(np.float64)
    values = np.concatenate([
        delta.mean(axis=0), delta.std(axis=0), delta.min(axis=0), delta.max(axis=0),
        delta[-1], np.abs(velocity).max(axis=0), np.abs(acceleration).max(axis=0),
        np.abs(jerk).max(axis=0), pose,
        np.array([
            trace.path_length,
            float(trace.direction_reversal_count),
            float(np.count_nonzero(trace.gripper_events)),
            float(valid.sum()),
            float(gripper.mean()) if gripper.size else 0.0,
            float(gripper.std()) if gripper.size else 0.0,
            float(gripper[-1]) if gripper.size else 0.0,
        ]),
    ])
    if not np.isfinite(values).all():
        raise ValueError("trace feature vector is non-finite")
    return values.astype(np.float64)


def raw_action_feature_vector(actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float64)[np.asarray(mask, dtype=np.bool_)]
    if value.ndim != 2 or value.shape[1] != 7 or not len(value):
        raise ValueError("raw action feature requires at least one valid 7-D step")
    result = np.concatenate([
        value.mean(axis=0), value.std(axis=0), value.min(axis=0), value.max(axis=0),
        value[-1], np.array([float(len(value))]),
    ])
    if not np.isfinite(result).all():
        raise ValueError("raw action feature vector is non-finite")
    return result


def task_folds(
    tasks: Sequence[str], suites: Mapping[str, str], *, seed: int, folds: int = 5,
) -> dict[str, int]:
    """Suite-stratified, task-held-out deterministic folds."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    by_suite: dict[str, list[str]] = defaultdict(list)
    for task in sorted(set(tasks)):
        by_suite[str(suites[task])].append(task)
    assignment: dict[str, int] = {}
    for suite, values in sorted(by_suite.items()):
        ordered = sorted(values, key=lambda task: (stable_seed("fold", seed, task), task))
        offset = stable_seed("offset", seed, suite) % folds
        for index, task in enumerate(ordered):
            assignment[task] = (index + offset) % folds
    return assignment


def ridge_oof_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    tasks: Sequence[str],
    folds_by_task: Mapping[str, int],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fixed-alpha standardized ridge with strictly task-held-out OOF output."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(tasks) != len(x):
        raise ValueError("incompatible feature, target, or task shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    for fold in sorted(set(folds_by_task.values())):
        test = np.array([folds_by_task[task] == fold for task in tasks])
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(f"fold {fold} has no train or test rows")
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (x[train] - mean) / scale
        x_test = (x[test] - mean) / scale
        y_mean = float(y[train].mean())
        design = np.column_stack((np.ones(len(x_train)), x_train))
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ (y[train] - y_mean))
        predictions[test] = y_mean + np.column_stack((np.ones(len(x_test)), x_test)) @ beta
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions are incomplete or non-finite")
    return predictions


@dataclass(frozen=True)
class GroupMetric:
    pairwise_correct: int
    pairwise_total: int
    regret: float


def grouped_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
    *,
    tie_margin: float = 0.0,
) -> tuple[dict[str, float], dict[str, GroupMetric]]:
    """Same-state pairwise accuracy and selected-candidate oracle regret."""
    if tie_margin < 0 or not math.isfinite(tie_margin):
        raise ValueError("tie_margin must be finite and non-negative")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        by_group[str(group)].append(index)
    details: dict[str, GroupMetric] = {}
    correct = total = 0
    regrets: list[float] = []
    for group, indices in by_group.items():
        truth = np.asarray(targets)[indices]
        score = np.asarray(predictions)[indices]
        if len(indices) < 2:
            continue
        group_correct = group_total = 0
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                difference = float(truth[left] - truth[right])
                if abs(difference) <= tie_margin:
                    continue
                predicted = float(score[left] - score[right])
                group_correct += int(predicted * difference > 0)
                group_total += 1
        selected = int(np.argmax(score))
        regret = float(np.max(truth) - truth[selected])
        details[group] = GroupMetric(group_correct, group_total, regret)
        correct += group_correct
        total += group_total
        regrets.append(regret)
    return {
        "pairwise_accuracy": correct / total if total else 0.0,
        "pairwise_pairs": total,
        "mean_oracle_regret": float(np.mean(regrets)) if regrets else 0.0,
        "groups": len(details),
    }, details


def bootstrap_task_difference(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    *,
    replicates: int = 10000,
    seed: int = 202708,
) -> tuple[float, list[float]]:
    """Paired task bootstrap of per-task mean metric differences."""
    tasks = sorted(set(left) & set(right))
    if not tasks or replicates <= 0:
        raise ValueError("paired task bootstrap requires tasks and positive replicates")
    values = np.array([
        float(np.mean(left[task])) - float(np.mean(right[task])) for task in tasks
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), [
        float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)),
    ]


def finite_dict(value: Mapping[str, Any]) -> None:
    """Raise if a nested numeric report contains NaN/Inf."""
    def walk(item: Any) -> Iterable[float]:
        if isinstance(item, Mapping):
            for nested in item.values():
                yield from walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                yield from walk(nested)
        elif isinstance(item, (float, np.floating)):
            yield float(item)
    if any(not math.isfinite(number) for number in walk(value)):
        raise ValueError("report contains a non-finite number")

from __future__ import annotations

import numpy as np

from rase.vnext.libero import LIBERO_ACTION_SEMANTICS, LIBERO_MOTION_SEMANTIC_MAP
from rase.vnext.phase_c_pilot import (
    bootstrap_task_difference,
    choose_tasks,
    grouped_metrics,
    pad_action_chunk,
    raw_action_feature_vector,
    ridge_oof_predictions,
    task_folds,
    trace_feature_vector,
)


def test_pad_action_chunk_preserves_mask_and_values() -> None:
    values = [np.arange(7, dtype=np.float32), np.ones(7, dtype=np.float32)]
    padded, mask = pad_action_chunk(values, horizon=4)
    assert padded.shape == (4, 7)
    assert mask.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(padded[:2], values)
    np.testing.assert_array_equal(padded[2:], 0)


def test_choose_tasks_is_suite_stratified_and_outcome_independent() -> None:
    jobs = [
        {"suite": "Spatial", "task_id": "s2", "utility": 100},
        {"suite": "Spatial", "task_id": "s1", "utility": -100},
        {"suite": "Goal", "task_id": "g2", "utility": -100},
        {"suite": "Goal", "task_id": "g1", "utility": 100},
    ]
    assert choose_tasks(jobs, tasks_per_suite=1) == ["g1", "s1"]


def test_raw_and_trace_features_are_finite() -> None:
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:2, 0] = [1.0, -1.0]
    actions[:2, 6] = [-1.0, 1.0]
    mask = np.array([True, True] + [False] * 8)
    raw = raw_action_feature_vector(actions, mask)
    trace = trace_feature_vector(
        actions, mask,
        semantics=LIBERO_ACTION_SEMANTICS,
        policy_id="pi0fast.libero",
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
    )
    assert np.isfinite(raw).all()
    assert np.isfinite(trace).all()
    assert raw.ndim == trace.ndim == 1


def test_task_held_out_ridge_and_group_metrics() -> None:
    tasks = [f"task{i}" for i in range(10) for _ in range(3)]
    suites = {f"task{i}": "A" if i < 5 else "B" for i in range(10)}
    folds = task_folds(tasks, suites, seed=0, folds=5)
    x = np.array([[operator] for _ in range(10) for operator in range(3)], dtype=float)
    y = x[:, 0].copy()
    predictions = ridge_oof_predictions(x, y, tasks, folds, alpha=0.01)
    groups = [f"group{i}" for i in range(10) for _ in range(3)]
    metrics, details = grouped_metrics(y, predictions, groups)
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["mean_oracle_regret"] == 0.0
    assert len(details) == 10


def test_group_metrics_excludes_practical_ties() -> None:
    targets = np.array([1.0, 0.999, 1.0, 0.8])
    predictions = np.array([1.0, 0.0, 1.0, 0.0])
    metrics, details = grouped_metrics(
        targets, predictions, ["tie", "tie", "informative", "informative"],
        tie_margin=0.01,
    )
    assert metrics["pairwise_pairs"] == 1
    assert metrics["pairwise_accuracy"] == 1.0
    assert details["tie"].pairwise_total == 0
    assert details["informative"].pairwise_total == 1


def test_paired_task_bootstrap_detects_positive_difference() -> None:
    left = {f"t{i}": [0.8, 0.9] for i in range(8)}
    right = {f"t{i}": [0.5, 0.6] for i in range(8)}
    mean, interval = bootstrap_task_difference(left, right, replicates=1000, seed=3)
    assert np.isclose(mean, 0.3)
    assert interval[0] > 0

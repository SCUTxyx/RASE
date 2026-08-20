"""Unit tests for the R6-C two-boundary dwell controller metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.train_r6c_dynamic_risk_oof import controller_metrics


def make_data(lcbs: list[list[float]], source: list[bool],
              persistent: list[list[bool]], steps: list[list[float]]) -> dict:
    """Three trajectory groups with two boundaries each."""
    rows = []
    for gid, (group_lcbs, source_succ, group_persistent, group_steps) in enumerate(
            zip(lcbs, source, persistent, steps)):
        for boundary in range(len(group_lcbs)):
            rows.append({
                "group_id": f"g{gid}", "state_key": f"s{gid}",
                "task_id": f"t{gid}", "policy_id": "pi0fast_libero",
                "elapsed_source_steps": boundary * 16,
                "source_success": source_succ, "persistent_success": group_persistent[boundary],
                "persistent_teacher_steps": group_steps[boundary],
            })
    index = np.arange(len(rows))
    return {
        "group_id": np.asarray([row["group_id"] for row in rows]),
        "state_key": np.asarray([row["state_key"] for row in rows]),
        "task_id": np.asarray([row["task_id"] for row in rows]),
        "policy_id": np.asarray([row["policy_id"] for row in rows]),
        "elapsed_source_steps": np.asarray([row["elapsed_source_steps"] for row in rows], dtype=np.int32),
        "source_success": np.asarray([row["source_success"] for row in rows], dtype=np.float32),
        "persistent_success": np.asarray([row["persistent_success"] for row in rows], dtype=np.float32),
        "persistent_teacher_steps": np.asarray([row["persistent_teacher_steps"] for row in rows], dtype=np.float32),
    }, index


def test_dwell_requires_two_consecutive_risky_boundaries() -> None:
    """A single risky boundary is not enough; two consecutive risky boundaries enter."""
    # g0: not risky at b0, risky at b1 (dwell=2 never fires) -> source succeeds.
    # g1: risky at b0 AND b1 -> enters at b1.
    data, idx = make_data(
        lcbs=[[0.8, 0.2], [0.2, 0.1]],
        source=[True, True],
        persistent=[[False, False], [True, True]],
        steps=[[100.0, 80.0], [90.0, 70.0]],
    )
    lcb = np.asarray([0.8, 0.2, 0.2, 0.1])
    metrics = controller_metrics(data, idx, lcb, threshold=0.5, dwell=2)
    records = metrics["trajectories"]
    assert records[0]["entered_persistent"] is False
    assert records[0]["controller_success"] is True  # source completed
    assert records[1]["entered_persistent"] is True
    assert records[1]["enter_elapsed"] == 16
    # Baseline is persistent at t=0: successes = 0 + 1, controller = 1 + 1.
    assert metrics["success_gap"] == pytest.approx((2 - 1) / 2)


def test_single_boundary_trajectory_never_enters_with_dwell_2() -> None:
    """A trajectory with only one recorded boundary cannot fire a dwell=2 enter."""
    data, idx = make_data(
        lcbs=[[0.1]],
        source=[False],
        persistent=[[True]],
        steps=[[50.0]],
    )
    lcb = np.asarray([0.1])
    metrics = controller_metrics(data, idx, lcb, threshold=0.5, dwell=2)
    records = metrics["trajectories"]
    assert records[0]["entered_persistent"] is False
    # Source failed and persistent would have succeeded -> false continue.
    assert metrics["false_continue"] == 1.0


def test_false_continue_counted_only_when_persistent_would_rescue() -> None:
    """Continuing a failing source is only 'false continue' if persistent succeeds."""
    data, idx = make_data(
        lcbs=[[0.9], [0.9]],
        source=[False, False],
        persistent=[[True], [False]],
        steps=[[50.0], [60.0]],
    )
    lcb = np.asarray([0.9, 0.9])
    metrics = controller_metrics(data, idx, lcb, threshold=0.5, dwell=2)
    assert metrics["false_continue"] == 1.0  # only g0, not g1
    assert metrics["episodes"] == 2.0


def test_teacher_step_savings() -> None:
    """Savings = 1 - controller teacher steps / persistent-at-0 baseline."""
    # g0 enters at b1 (second consecutive risky boundary) -> persistent 60.
    # g1 never enters (source succeeds, 0 teacher steps).
    data, idx = make_data(
        lcbs=[[0.1, 0.1], [0.9, 0.9]],
        source=[False, True],
        persistent=[[True, True], [False, False]],
        steps=[[90.0, 60.0], [80.0, 50.0]],
    )
    lcb = np.asarray([0.1, 0.1, 0.9, 0.9])
    metrics = controller_metrics(data, idx, lcb, threshold=0.5, dwell=2)
    assert metrics["baseline_teacher_steps"] == pytest.approx(90.0 + 80.0)
    assert metrics["teacher_steps"] == pytest.approx(60.0 + 0.0)
    assert metrics["savings"] == pytest.approx(1 - 60.0 / 170.0)

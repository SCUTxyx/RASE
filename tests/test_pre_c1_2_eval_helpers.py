from __future__ import annotations

import numpy as np

from rase.adapt.pre_c1_2 import sample_recovery_row
from rase.adapt.pre_c1_2_eval import (
    action_space_report,
    mae_per_dim,
    successor_distance,
)


def test_mae_and_action_space_report():
    a = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    b = np.array([0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    report = action_space_report(
        student_normalized=a,
        teacher_normalized=b,
        student_denormalized=a,
        teacher_denormalized=b,
        student_env=a,
        teacher_env=b,
    )
    assert len(report["env_action_mae_per_dim"]) == 7
    assert report["env_action_mae"] == float(np.mean(mae_per_dim(a, b)))


def test_successor_distance_uses_agent_pos():
    obs_a = {"agent_pos": np.array([0.0, 0.0, 0.0])}
    obs_b = {"agent_pos": np.array([3.0, 4.0, 0.0])}
    dist = successor_distance(obs_a, obs_b)
    assert abs(dist["aggregate_l2"] - 5.0) < 1e-6


def test_sample_recovery_prefers_query_offset():
    rows = [
        {
            "anchor_id": "a",
            "source": "student_query_state",
            "offset_from_student_state": 0,
            "sample_id": "q",
        },
        {
            "anchor_id": "a",
            "source": "teacher_suffix_after_student_query",
            "offset_from_student_state": 2,
            "sample_id": "s",
        },
    ]
    import random

    rng = random.Random(0)
    counts = {"student_query_state": 0, "teacher_suffix_after_student_query": 0}
    for _ in range(200):
        row = sample_recovery_row(
            student_rows=rows,
            original_rows=[],
            student_weight=1.0,
            original_weight=0.0,
            rng=rng,
        )
        counts[str(row["source"])] += 1
    assert counts["student_query_state"] > counts["teacher_suffix_after_student_query"]

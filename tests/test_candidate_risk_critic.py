from __future__ import annotations

import numpy as np

from rase.risk import (
    action_chunk_features,
    evaluate_selector_baselines,
    export_candidate_rows,
    fit_logistic_scorer,
)


def test_action_chunk_features_shape():
    actions = np.zeros((4, 7), dtype=np.float64)
    actions[:, 0] = 1.0
    features = action_chunk_features(actions)
    assert features.shape == (12,)
    assert np.all(np.isfinite(features))


def test_export_and_selector_baselines_oracle_capture():
    states = [
        {
            "state_key": "s0",
            "episode_id": "e0",
            "task_id": "t0",
            "suite": "Spatial",
            "cell": "clean:L0",
            "stage": "T1",
            "source_episode_outcome": "failure",
            "arms": [
                {
                    "family": "current_suffix",
                    "arm_name": "current_suffix",
                    "success": False,
                    "action_tensor": np.zeros((2, 7)).tolist(),
                    "execution_horizon": None,
                },
                {
                    "family": "strict_resample",
                    "arm_name": "strict_resample_0",
                    "success": True,
                    "action_tensor": (np.ones((2, 7)) * 0.1).tolist(),
                    "execution_horizon": None,
                },
            ],
        }
    ]
    rows = export_candidate_rows(states)
    assert len(rows) == 2
    x = np.asarray([row["x_candidate"] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["success"]) for row in rows], dtype=np.float64)
    scorer = fit_logistic_scorer(x, y, kind="candidate_conditioned", steps=200)
    metrics = evaluate_selector_baselines(rows, candidate_scorer=scorer)
    assert metrics["success_rates"]["oracle_at_k"] == 1.0
    assert metrics["success_rates"]["current_suffix"] == 0.0

#!/usr/bin/env python3
"""Small regression tests for P1/P2 CRR analysis semantics."""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_crr_baselines import gain_curve, predict_antisym_mlp
from measure_within_task_heterogeneity import argmax_per_root, root_key


def pair(q_i, q_j, suite="libero_spatial"):
    return {"model_i": "oft_spatial", "model_j": "oft_object",
            "q_i": q_i, "q_j": q_j, "suite": suite}


def test_gain_is_signed_and_abstentions_are_zero():
    pairs = [pair(1.0, 0.0), pair(1.0, 0.0)]
    # First root stays with the spatial default; second confidently switches
    # to the worse object candidate.  Deployment gain must be -1/2, not +1.
    rep = gain_curve(pairs, np.array([0.9, 0.1]), [0.0])
    row = rep["curve"][0]
    assert row["coverage"] == 0.5
    assert row["precision"] == 0.0
    assert row["gain"] == -0.5
    assert row["gain_per_switch"] == -1.0


def test_antisymmetric_predictor_reverses_probability():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 7))
    xr = rng.normal(size=(5, 7))
    W1 = rng.normal(size=(7, 4))
    b1 = rng.normal(size=4)
    W2 = rng.normal(size=4)
    p = predict_antisym_mlp(x, xr, W1, b1, W2)
    pr = predict_antisym_mlp(xr, x, W1, b1, W2)
    assert np.allclose(p + pr, 1.0, atol=1e-12)


def test_p2_root_lookup_includes_model_identity():
    rows = [
        {"task": "t", "episode_idx": 0, "decision_idx": 0,
         "model": "a"},
        {"task": "t", "episode_idx": 0, "decision_idx": 0,
         "model": "b"},
    ]
    rk = root_key(rows[0])
    roots = {rk: rows}
    by_key = {(root_key(r), r["model"]): i for i, r in enumerate(rows)}
    assert argmax_per_root(roots, np.array([0.0, 1.0]), by_key)[rk] == "b"


if __name__ == "__main__":
    test_gain_is_signed_and_abstentions_are_zero()
    test_antisymmetric_predictor_reverses_probability()
    test_p2_root_lookup_includes_model_identity()
    print(json.dumps({"tests": 3, "status": "PASS"}))

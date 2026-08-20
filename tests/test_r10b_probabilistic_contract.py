from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_r10c_probabilistic_information_exploratory.py"
SPEC = importlib.util.spec_from_file_location("r10c_probability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_event_auc_expands_binomial_counts_without_group_duplication_bias() -> None:
    successes = np.asarray([0, 3], dtype=np.float64)
    trials = np.asarray([3, 3], dtype=np.float64)
    scores = np.asarray([0.1, 0.9], dtype=np.float64)
    assert MODULE.event_auc(successes, trials, scores) == 1.0


def test_event_auc_handles_mixed_groups() -> None:
    successes = np.asarray([1, 2], dtype=np.float64)
    trials = np.asarray([3, 3], dtype=np.float64)
    scores = np.asarray([0.2, 0.8], dtype=np.float64)
    value = MODULE.event_auc(successes, trials, scores)
    assert 0.5 < value < 1.0


def test_binomial_log_loss_prefers_calibrated_probability() -> None:
    successes = np.asarray([0, 3], dtype=np.float64)
    trials = np.asarray([3, 3], dtype=np.float64)
    good = MODULE.binomial_log_loss(successes, trials, np.asarray([0.1, 0.9]))
    bad = MODULE.binomial_log_loss(successes, trials, np.asarray([0.9, 0.1]))
    assert good < bad


def test_count_logistic_returns_finite_parameters() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    successes = np.asarray([0, 0, 3, 3], dtype=np.float64)
    trials = np.full(4, 3.0)
    weights, mean, std = MODULE.fit_logistic_counts(x, successes, trials, seed=1)
    assert np.isfinite(weights).all()
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert weights[0] > 0

from __future__ import annotations

import numpy as np
import pytest

from rase.vnext.selector import (
    fit_pairwise,
    fit_ridge,
    predict_pairwise,
    risk_coverage_curve,
    select_candidates,
    utility_lambda,
)


def _features(n: int, dim: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim))


def test_fit_predict_reproducible() -> None:
    x = _features(100)
    y = (x[:, 0] > 0).astype(np.float64)
    model_a = fit_ridge(x, y, alpha=1.0)
    model_b = fit_ridge(x, y, alpha=1.0)
    assert np.allclose(model_a.predict(x), model_b.predict(x))
    assert model_a.weights.shape == (6,)
    probabilities = model_a.predict(x)
    assert probabilities.shape == (100,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pairwise_model_separates_delta() -> None:
    rng = np.random.default_rng(1)
    x_a = rng.normal(size=(80, 6))
    x_b = x_a + np.column_stack([rng.normal(0, 0.3, 80), np.zeros((80, 5))])
    delta = (x_a[:, 0] > x_b[:, 0]).astype(np.float64) * 2 - 1
    model = fit_pairwise(x_a, x_b, delta, alpha=0.1)
    scores = predict_pairwise(model, x_a, x_b)
    assert np.mean((scores > 0.5) == (delta > 0)) > 0.7


def test_select_candidates_abstain() -> None:
    decision = select_candidates(
        {"continue.source": 0.9, "requery.source": 0.88, "fallback.persistent": 0.6},
        abstain_margin=0.05,
    )
    assert decision.abstained is True
    assert decision.chosen_operator == "continue.source"
    assert abs(decision.margin - 0.02) < 1e-9
    decision2 = select_candidates(
        {"continue.source": 0.5, "requery.source": 0.4, "fallback.persistent": 0.95},
        abstain_margin=0.05,
    )
    assert decision2.abstained is False
    assert decision2.chosen_operator == "fallback.persistent"


def test_risk_coverage_curve_monotonic() -> None:
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.2, 0.9, 200)
    targets = (scores + rng.normal(0, 0.2, 200) > 0.5).astype(np.float64)
    curve = risk_coverage_curve(scores, targets)
    coverages = [curve[str(t)]["coverage"] for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert coverages == sorted(coverages, reverse=True)


def test_utility_lambda() -> None:
    assert utility_lambda(1.0, 0.0, 0.1) == 1.0
    assert abs(utility_lambda(1.0, 0.5, 0.2) - 0.9) < 1e-9
    assert abs(utility_lambda(0.0, 0.5, 0.2) + 0.1) < 1e-9


def test_ridge_save_load_roundtrip(tmp_path) -> None:
    x = _features(60)
    y = (x[:, 1] > 0).astype(np.float64)
    model = fit_ridge(x, y, alpha=2.0, model_type="candidate",
                      feature_version="test/v1", training_manifest_sha256="abc",
                      code_version="test")
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = model.__class__.load(path)
    assert loaded.model_type == "candidate"
    assert loaded.alpha == 2.0
    assert loaded.feature_version == "test/v1"
    assert loaded.training_manifest_sha256 == "abc"
    assert np.allclose(loaded.predict(x), model.predict(x))

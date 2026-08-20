"""Lightweight logistic candidate risk scorer and selector baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CandidateRiskScorer:
    weights: np.ndarray
    bias: float
    feature_dim: int
    kind: str

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return predict_proba(self, features)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_scorer(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    lr: float = 0.1,
    steps: int = 800,
    l2: float = 1e-3,
    seed: int = 2_026_080_405,
) -> CandidateRiskScorer:
    """Fit a binary logistic model with L2 regularization via GD."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x must be [N,D] aligned with y")
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=x.shape[1])
    bias = 0.0
    for _ in range(int(steps)):
        logits = x @ weights + bias
        probs = _sigmoid(logits)
        err = probs - y
        grad_w = (x.T @ err) / len(y) + l2 * weights
        grad_b = float(np.mean(err))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return CandidateRiskScorer(
        weights=weights,
        bias=float(bias),
        feature_dim=int(x.shape[1]),
        kind=kind,
    )


def predict_proba(scorer: CandidateRiskScorer, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != scorer.feature_dim:
        raise ValueError(
            f"feature dim mismatch: got {x.shape[1]}, expected {scorer.feature_dim}"
        )
    return _sigmoid(x @ scorer.weights + scorer.bias)


def _group_by_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_key"])].append(row)
    return grouped


def evaluate_selector_baselines(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_scorer: CandidateRiskScorer | None = None,
    history_scorer: CandidateRiskScorer | None = None,
) -> dict[str, Any]:
    """Compare random / action-norm / history-only / candidate / oracle@K selectors."""

    by_state = _group_by_state(rows)
    methods = {
        "random_first": 0,
        "action_l2_min": 0,
        "history_only": 0,
        "candidate_conditioned": 0,
        "oracle_at_k": 0,
        "current_suffix": 0,
    }
    n_states = 0
    for state_key, candidates in by_state.items():
        if not candidates:
            continue
        n_states += 1
        current = next((c for c in candidates if c.get("family") == "current_suffix"), None)
        methods["current_suffix"] += int(bool(current and current.get("success")))
        methods["random_first"] += int(bool(candidates[0].get("success")))
        methods["action_l2_min"] += int(
            bool(min(candidates, key=lambda c: float(c.get("action_l2", 0.0))).get("success"))
        )
        methods["oracle_at_k"] += int(any(bool(c.get("success")) for c in candidates))
        if history_scorer is not None:
            scores = predict_proba(
                history_scorer,
                np.asarray([c["x_history"] for c in candidates], dtype=np.float64),
            )
            methods["history_only"] += int(bool(candidates[int(np.argmax(scores))].get("success")))
        if candidate_scorer is not None:
            scores = predict_proba(
                candidate_scorer,
                np.asarray([c["x_candidate"] for c in candidates], dtype=np.float64),
            )
            methods["candidate_conditioned"] += int(
                bool(candidates[int(np.argmax(scores))].get("success"))
            )

    rates = {name: (count / n_states if n_states else 0.0) for name, count in methods.items()}
    oracle = rates["oracle_at_k"]
    current = rates["current_suffix"]
    candidate = rates["candidate_conditioned"]
    return {
        "n_states": n_states,
        "n_candidates": len(rows),
        "success_counts": methods,
        "success_rates": rates,
        "headroom_pp": {
            "oracle_vs_current": 100.0 * (oracle - current),
            "candidate_vs_current": 100.0 * (candidate - current),
            "candidate_capture_of_oracle": (
                (candidate - current) / (oracle - current)
                if (oracle - current) > 1e-12
                else None
            ),
        },
    }

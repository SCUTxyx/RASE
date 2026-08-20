"""Low-capacity same-root candidate selector (v2 plan §6-§7).

Two model families over pre-registered action features:

  - candidate-level ridge: absolute P(success) per candidate row;
  - explicit pairwise ridge: delta model over (x_a - x_b) predicting
    success_a - success_b within the same root.

Selection is pure-prediction: no realized outcome/utility is read at decision
time.  Abstention uses the top-1/top-2 score margin with a frozen threshold;
costs enter only at the deployment layer as U_lambda = success - lambda*cost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _stable_hash(*parts: object) -> str:
    token = "\x1f".join(str(part) for part in parts).encode()
    return hashlib.sha256(token).hexdigest()


@dataclass(frozen=True)
class RidgeModel:
    """Standardized ridge (or ridge-logistic via clip) with provenance."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    alpha: float
    model_type: str  # "candidate" | "pairwise"
    feature_version: str
    training_manifest_sha256: str
    code_version: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        logit = self.intercept + x @ self.weights
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: Path) -> None:
        payload = {
            "model_type": self.model_type,
            "alpha": self.alpha,
            "feature_version": self.feature_version,
            "training_manifest_sha256": self.training_manifest_sha256,
            "code_version": self.code_version,
            "intercept": self.intercept,
        }
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, mean=self.mean, scale=self.scale, weights=self.weights, **payload,
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                mean=arrays["mean"], scale=arrays["scale"],
                weights=arrays["weights"],
                intercept=float(arrays["intercept"]),
                alpha=float(arrays["alpha"]),
                model_type=str(arrays["model_type"]),
                feature_version=str(arrays["feature_version"]),
                training_manifest_sha256=str(arrays["training_manifest_sha256"]),
                code_version=str(arrays["code_version"]),
            )


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1.0,
    model_type: str = "candidate",
    feature_version: str = "rase-vnext-selector-features/v1",
    training_manifest_sha256: str = "",
    code_version: str = "rase-vnext-selector/v1",
) -> RidgeModel:
    """Standardized ridge; intercept unpenalized.  Targets may be 0/1 (candidate)
    or deltas in {-1,0,1} (pairwise)."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("incompatible feature/target shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_std = (x - mean) / scale
    y_mean = float(y.mean())
    design = np.column_stack((np.ones(len(x_std)), x_std))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        design.T @ design + penalty, design.T @ (y - y_mean)
    )
    return RidgeModel(
        mean=mean, scale=scale, weights=beta[1:], intercept=y_mean + beta[0],
        alpha=float(alpha), model_type=model_type,
        feature_version=feature_version,
        training_manifest_sha256=training_manifest_sha256,
        code_version=code_version,
    )


def fit_pairwise(
    features_a: np.ndarray,
    features_b: np.ndarray,
    targets_delta: np.ndarray,
    **kwargs: Any,
) -> RidgeModel:
    """Explicit pairwise ridge over (x_a - x_b) -> success_a - success_b."""
    x = np.asarray(features_a, dtype=np.float64) - np.asarray(features_b, dtype=np.float64)
    return fit_ridge(x, np.asarray(targets_delta, dtype=np.float64), model_type="pairwise", **kwargs)


def predict_pairwise(model: RidgeModel, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """P(success_a > success_b) from the delta model."""
    x = np.asarray(features_a, dtype=np.float64) - np.asarray(features_b, dtype=np.float64)
    return model.predict(x)


@dataclass(frozen=True)
class SelectorDecision:
    chosen_operator: str
    abstained: bool
    margin: float
    scores: Mapping[str, float]


def select_candidates(
    scores: Mapping[str, float],
    *,
    abstain_margin: float,
    default_operator: str = "continue.source",
) -> SelectorDecision:
    """Rank candidates by predicted score; abstain (keep default) when the
    top-1/top-2 margin is below the frozen threshold."""
    if not scores:
        raise ValueError("select_candidates requires at least one score")
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    top, second = ordered[0], ordered[1] if len(ordered) >= 2 else None
    margin = top[1] - (second[1] if second is not None else 0.0)
    abstained = margin < abstain_margin
    chosen = default_operator if abstained else top[0]
    return SelectorDecision(
        chosen_operator=chosen, abstained=abstained,
        margin=float(margin), scores=dict(scores),
    )


def risk_coverage_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9),
) -> dict[str, dict[str, float]]:
    """Coverage / accuracy / brier at frozen score thresholds."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    targets = np.asarray(targets, dtype=np.float64)
    curve: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        selected = scores >= float(threshold)
        coverage = float(selected.mean())
        if selected.any():
            accuracy = float((scores[selected] > 0.5).astype(float).mean())
            brier = float(((scores[selected] - targets[selected]) ** 2).mean())
        else:
            accuracy, brier = float("nan"), float("nan")
        curve[str(threshold)] = {
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4) if np.isfinite(accuracy) else None,
            "brier": round(brier, 4) if np.isfinite(brier) else None,
        }
    return curve


def utility_lambda(success: float, cost: float, lam: float) -> float:
    """Deployment-layer utility: U_lambda = success - lambda * cost."""
    return float(success) - float(lam) * float(cost)


def cost_of_row(row: Mapping[str, Any]) -> float:
    """Total normalized cost (query + fallback + latency) of a collected row."""
    return (
        float(row.get("query_cost") or 0.0)
        + float(row.get("fallback_cost") or 0.0)
        + float(row.get("latency_cost") or 0.0)
    )


@dataclass
class SelectorArtifact:
    """Versioned, reproducible selector package."""

    model: RidgeModel
    abstain_margin: float
    cost_policy: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain_margin": self.abstain_margin,
            "cost_policy": self.cost_policy,
            "model": {
                "model_type": self.model.model_type,
                "feature_version": self.model.feature_version,
                "training_manifest_sha256": self.model.training_manifest_sha256,
                "code_version": self.model.code_version,
                "alpha": self.model.alpha,
            },
            "extra": self.extra,
        }

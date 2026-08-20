"""Unified DynamicsBackend API for R4 safe-handback world model.

Provides a common interface for ridge, MLP, and linear dynamics predictors.
All backends implement fit/predict for one-step action-conditioned latent delta
prediction, enabling A/B comparison in the nested OOF pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DynamicsBackend(ABC):
    """Abstract base class for action-conditioned latent dynamics prediction.

    Each backend predicts the one-step latent delta given current state
    features and an action transition tensor.
    """

    @abstractmethod
    def fit(self, data: dict[str, np.ndarray], **kwargs: Any) -> "DynamicsBackend":
        """Fit the dynamics model on training data.

        Args:
            data: Dict with keys 'state', 'transition', 'delta', 'terminal',
                  'latent' (from build_arrays).
        """
        ...

    @abstractmethod
    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Predict delta of shape (n, arms, latent_dim).

        Args:
            data: Dict with same keys as fit.

        Returns:
            Predicted delta array of shape (n, arms, latent_dim).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for reporting."""
        ...


# ---------------------------------------------------------------------------
# Ridge Dynamics Backend
# ---------------------------------------------------------------------------

class RidgeDynamicsBackend(DynamicsBackend):
    """Ridge regression with latent*action interaction terms."""

    def __init__(self, alpha: float = 1000.0) -> None:
        self.alpha = alpha
        self.model = None
        self._fitted = False

    @property
    def name(self) -> str:
        return f"ridge_alpha{self.alpha:.0f}"

    def fit(self, data: dict[str, np.ndarray], **kwargs: Any) -> "RidgeDynamicsBackend":
        x, y, keep = self._build_features(data)
        self.model = _Ridge(alpha=self.alpha).fit(x[keep], y[keep])
        self._fitted = True
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RidgeDynamicsBackend not fitted")
        x, _, _ = self._build_features(data)
        n, arms = data["delta"].shape[:2]
        pred_flat = self.model.predict(x).astype(np.float32)
        return pred_flat.reshape(n, arms, -1)

    @staticmethod
    def _build_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, arms, action_dim = data["transition"].shape
        latent_dim = data["delta"].shape[-1]
        state = np.repeat(data["state"], arms, axis=0)
        action = data["transition"].reshape(n * arms, action_dim)
        latent = data["latent"][:, None, :].repeat(arms, axis=0).reshape(n * arms, latent_dim)
        features = np.concatenate([
            state,
            action,
            (latent[:, :, None] * action[:, None, :]).reshape(n * arms, -1),
        ], axis=1)
        target = data["delta"].reshape(n * arms, -1)
        keep = (1.0 - data["terminal"].reshape(-1)).astype(bool)
        return features, target, keep


class _Ridge:
    """Minimal dependency-free multi-output ridge regressor (internal)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.x_mean: np.ndarray = np.array([])
        self.y_mean: np.ndarray = np.array([])
        self.coef: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Ridge":
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc, yc = x - self.x_mean, y - self.y_mean
        if xc.shape[1] <= xc.shape[0]:
            gram = xc.T @ xc
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            self.coef = np.linalg.solve(gram, xc.T @ yc)
        else:
            gram = xc @ xc.T
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            self.coef = xc.T @ np.linalg.solve(gram, yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.x_mean) @ self.coef + self.y_mean


# ---------------------------------------------------------------------------
# Persistence (Zero-Delta) Backend
# ---------------------------------------------------------------------------

class PersistenceBackend(DynamicsBackend):
    """Predicts zero delta (the persistence baseline)."""

    @property
    def name(self) -> str:
        return "persistence"

    def fit(self, data: dict[str, np.ndarray], **kwargs: Any) -> "PersistenceBackend":
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros_like(data["delta"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Linear (No-Interaction Ridge) Backend
# ---------------------------------------------------------------------------

class LinearDynamicsBackend(DynamicsBackend):
    """Ridge regression WITHOUT latent*action interaction terms."""

    def __init__(self, alpha: float = 1000.0) -> None:
        self.alpha = alpha
        self.model = None
        self._fitted = False

    @property
    def name(self) -> str:
        return f"linear_alpha{self.alpha:.0f}"

    def fit(self, data: dict[str, np.ndarray], **kwargs: Any) -> "LinearDynamicsBackend":
        x, y, keep = self._build_features(data)
        self.model = _Ridge(alpha=self.alpha).fit(x[keep], y[keep])
        self._fitted = True
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LinearDynamicsBackend not fitted")
        x, _, _ = self._build_features(data)
        n, arms = data["delta"].shape[:2]
        pred_flat = self.model.predict(x).astype(np.float32)
        return pred_flat.reshape(n, arms, -1)

    @staticmethod
    def _build_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, arms, action_dim = data["transition"].shape
        state = np.repeat(data["state"], arms, axis=0)
        action = data["transition"].reshape(n * arms, action_dim)
        features = np.concatenate([state, action], axis=1)
        target = data["delta"].reshape(n * arms, -1)
        keep = (1.0 - data["terminal"].reshape(-1)).astype(bool)
        return features, target, keep


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BACKEND_REGISTRY = {
    "ridge": RidgeDynamicsBackend,
    "persistence": PersistenceBackend,
    "linear": LinearDynamicsBackend,
}


def create_backend(name: str, **kwargs: Any) -> DynamicsBackend:
    """Create a dynamics backend by name.

    Args:
        name: One of 'ridge', 'persistence', 'linear'.
        **kwargs: Passed to backend constructor.

    Returns:
        A DynamicsBackend instance.
    """
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown dynamics backend: {name}. "
                         f"Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name](**kwargs)

"""Conformal calibration for safe-handback controller thresholds.

Computes one-sided conformal correction on inner calibration tasks to ensure
that LCB-gated handback decisions respect the target false-handback rate.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


class ConformalCalibrator:
    """One-sided split-conformal calibrator for handback LCB scores.

    Given calibration-set predictions (handback probabilities and true labels),
    computes a conformal correction that, when subtracted from the LCB, achieves
    a target false-handback rate with high probability.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Args: alpha = target false-handback rate (default 5%)."""
        self.alpha = float(alpha)
        self._correction: float = 0.0
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ConformalCalibrator":
        """Fit conformal correction.

        Args:
            scores: Handback probabilities for calibration samples (N,).
            labels: Binary handback-success labels (N,).  1 = safe to hand back.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if len(scores) < 2:
            raise ValueError("need at least 2 calibration samples")

        # Conformity score: for samples where handback FAILS (label=0),
        # we compute how much the model over-estimated safety.
        # Nonconformity = score - label, so high nonconformity = overconfidence.
        nonconformity = scores[labels == 0]
        if len(nonconformity) == 0:
            # No negative samples: correction = 0 (conservative, no shift needed)
            self._correction = 0.0
        else:
            n_cal = len(scores)
            q_level = np.ceil((1.0 - self.alpha) * (n_cal + 1)) / n_cal
            q_level = min(q_level, 1.0)
            self._correction = float(np.quantile(nonconformity, max(0.0, q_level)))

        self._fitted = True
        return self

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Apply conformal correction to scores. Returns corrected scores."""
        if not self._fitted:
            raise RuntimeError("ConformalCalibrator not fitted")
        return np.asarray(scores, dtype=np.float64) - self._correction

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": "ConformalCalibrator",
            "alpha": self.alpha,
            "correction": self._correction,
            "fitted": self._fitted,
        }

    @property
    def correction(self) -> float:
        return self._correction

    @property
    def is_fitted(self) -> bool:
        return self._fitted

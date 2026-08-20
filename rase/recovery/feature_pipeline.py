"""Shared RecoveryFeaturePipeline for Route C.

Collector, trainer, and evaluator MUST use the same pipeline instance
(or an identical one from the same SmolVLA checkpoint) so that
``obs_features`` is identical across the three stages.

Feature levels:
  F0 – zero features only (diagnostic baseline, not for production)
  F1 – proprio + student_action + stagnation stats (action-only)
  F2 – frozen SmolVLA latent (128-D) + proprio + action + stats (production)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from rase.collect.smolvla_feature_extractor import (
    ACTION_DIM,
    F2_FEATURE_DIM,
    PROPRIO_DIM,
    SMOLVLA_LATENT_DIM,
    STAGNATION_STATS_DIM,
    SmolVLAFeatureExtractor,
    build_feature_vector,
    build_proprio_features,
    build_stagnation_features,
)

# Increment on every schema or extraction change that affects output.
FEATURE_PIPELINE_VERSION = "route-c-recovery/v2"


@dataclass(frozen=True)
class RecoveryFeatures:
    """Named container for a single-timestep feature vector.

    All fields are 1-D float32 numpy arrays.
    """

    obs_features: np.ndarray  # shape (F2_FEATURE_DIM,) or equivalent
    smolvla_latent: np.ndarray  # shape (SMOLVLA_LATENT_DIM,)
    proprio: np.ndarray  # shape (PROPRIO_DIM,)
    student_action: np.ndarray  # shape (ACTION_DIM,)
    stagnation_length: int
    progress_delta: float
    feature_level: str  # "F0" | "F1" | "F2"
    pipeline_version: str

    @property
    def total_dim(self) -> int:
        return int(self.obs_features.size)


class RecoveryFeaturePipeline:
    """Unified feature extraction for Route C recovery pipeline.

    All three stages (collect, train, eval) instantiate this class with
    the same SmolVLA bundle, guaranteeing identical feature semantics.

    Parameters
    ----------
    smolvla_bundle : dict
        Loaded policy bundle from ``load_smolvla_policy_bundle``.
    latent_dim : int
        SmolVLA latent vector dimension (default 128).
    allow_missing_smolvla : bool
        If True, return zero latent when the extractor cannot capture a
        hidden state.  Only intended for F0/F1 diagnostics.
    """

    def __init__(
        self,
        smolvla_bundle: dict[str, Any] | None,
        *,
        latent_dim: int = SMOLVLA_LATENT_DIM,
        allow_missing_smolvla: bool = False,
    ):
        self._allow_missing = allow_missing_smolvla
        self._latent_dim = latent_dim
        if smolvla_bundle is not None:
            self._extractor = SmolVLAFeatureExtractor(
                smolvla_bundle, latent_dim=latent_dim
            )
        else:
            self._extractor = None
        self._version = FEATURE_PIPELINE_VERSION
        self._sha = self._compute_extractor_sha()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    @property
    def extractor_sha(self) -> str:
        return self._sha

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def extract(
        self,
        obs: dict[str, Any],
        proprio: np.ndarray,
        student_action: np.ndarray,
        *,
        stagnation_length: int = 0,
        progress_delta: float = 0.0,
        feature_level: str = "F2",
        seed: int | None = None,
    ) -> RecoveryFeatures:
        """Extract the full feature vector for one timestep.

        Parameters
        ----------
        obs : dict
            Raw LIBERO observation dict (must contain ``robot0_eef_pos``,
            ``pixels``, ``task`` etc.).
        proprio : np.ndarray
            Raw proprioceptive data (eef pos + quat).
        student_action : np.ndarray
            Current SmolVLA student action.
        stagnation_length : int
            Number of consecutive stagnation steps (0 if none).
        progress_delta : float
            Progress change since last step.
        feature_level : str
            ``"F0"``, ``"F1"``, or ``"F2"``.
        seed : int | None
            Random seed for deterministic latent extraction (passed to
            SmolVLAFeatureExtractor).

        Returns
        -------
        RecoveryFeatures

        Raises
        ------
        RuntimeError
            If ``feature_level == "F2"`` and no SmolVLA extractor is
            available (and ``allow_missing_smolvla`` is False).
        """
        if feature_level == "F2":
            latent = self._extract_smolvla_latent(obs, seed=seed)
            if np.count_nonzero(latent) == 0 and not self._allow_missing:
                raise RuntimeError(
                    "SmolVLA latent is all-zero in F2 mode. "
                    "This usually means the extractor hook did not fire. "
                    "Check that the observation dict contains valid 'pixels' "
                    "and 'task' keys."
                )
        else:
            latent = np.zeros(self._latent_dim, dtype=np.float32)

        proprio_arr = build_proprio_features(proprio)
        action_arr = np.asarray(student_action, dtype=np.float32).flatten()[:ACTION_DIM]
        if len(action_arr) < ACTION_DIM:
            padded = np.zeros(ACTION_DIM, dtype=np.float32)
            padded[: len(action_arr)] = action_arr
            action_arr = padded

        stag = build_stagnation_features(stagnation_length, progress_delta)

        obs_feat = np.concatenate([latent, proprio_arr, action_arr, stag]).astype(
            np.float32
        )

        return RecoveryFeatures(
            obs_features=obs_feat,
            smolvla_latent=latent,
            proprio=proprio_arr,
            student_action=action_arr,
            stagnation_length=int(stagnation_length),
            progress_delta=float(progress_delta),
            feature_level=feature_level,
            pipeline_version=self._version,
        )

    def extract_standalone(
        self,
        proprio: np.ndarray,
        student_action: np.ndarray,
        *,
        stagnation_length: int = 0,
        progress_delta: float = 0.0,
        feature_level: str = "F2",
    ) -> RecoveryFeatures:
        """Extract features WITHOUT a SmolVLA forward pass.

        The ``smolvla_latent`` portion will always be zero.  Intended for
        F0/F1 diagnostic modes where running the full VLA encoder is too
        expensive or unavailable (e.g., during offline dataset construction
        from pre-recorded episodes that lack raw images).
        """
        return self.extract(
            obs={},  # will trigger zero latent
            proprio=proprio,
            student_action=student_action,
            stagnation_length=stagnation_length,
            progress_delta=progress_delta,
            feature_level=feature_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline identity for round-trip verification."""
        return {
            "feature_pipeline_version": self._version,
            "feature_extractor_sha": self._sha,
            "latent_dim": self._latent_dim,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_smolvla_latent(self, obs: dict[str, Any], *, seed: int | None = None) -> np.ndarray:
        if self._extractor is None:
            if self._allow_missing:
                return np.zeros(self._latent_dim, dtype=np.float32)
            raise RuntimeError(
                "No SmolVLA extractor available. Pass smolvla_bundle to the "
                "constructor or set allow_missing_smolvla=True for F0/F1."
            )
        try:
            return self._extractor.extract(obs, seed=seed)
        except Exception:
            if not self._allow_missing:
                raise
            return np.zeros(self._latent_dim, dtype=np.float32)

    def _compute_extractor_sha(self) -> str:
        """Stable identifier representing the extractor configuration."""
        # Use the source module bytes as a proxy; in production this would
        # be the SmolVLA checkpoint SHA.
        try:
            import sys
            from pathlib import Path

            src = Path(__file__).resolve()
            digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
            return digest
        except Exception:
            return "unknown"

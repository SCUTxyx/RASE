"""Structured, simulator-privileged recovery scores.

These utilities are an upper-bound measurement tool.  They do not consume
observations available to a deployed policy and should not be presented as a
deployable recovery critic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionSignals:
    """Privileged measurements for one candidate continuation."""

    progress_delta: float
    grasp_stability: float
    collision_harm: float = 0.0
    irreversible: bool = False


@dataclass(frozen=True)
class RecoveryScoreWeights:
    """Frozen linear score weights; harmful quantities have positive weights."""

    progress: float = 1.0
    grasp_stability: float = 1.0
    collision_harm: float = 2.0
    irreversible_penalty: float = 100.0

    def __post_init__(self) -> None:
        values = (
            self.progress,
            self.grasp_stability,
            self.collision_harm,
            self.irreversible_penalty,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("score weights must be finite and non-negative")


@dataclass(frozen=True)
class RecoveryScore:
    """Auditable score with each contribution retained."""

    total: float
    progress: float
    grasp_stability: float
    collision_harm: float
    irreversible_penalty: float
    irreversible: bool


@dataclass(frozen=True)
class BestOfKSelection:
    """Result of scoring exactly ``k`` candidates at matched evaluation cost."""

    index: int
    candidate_id: str
    score: RecoveryScore
    scores: tuple[RecoveryScore, ...]
    evaluated_count: int


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def score_transition(
    signals: TransitionSignals,
    weights: RecoveryScoreWeights | None = None,
) -> RecoveryScore:
    """Compute progress + grasp - harm - irreversible penalty."""

    weights = weights or RecoveryScoreWeights()
    progress_delta = _finite_scalar("progress_delta", signals.progress_delta)
    grasp_stability = _finite_scalar("grasp_stability", signals.grasp_stability)
    collision_harm = _finite_scalar("collision_harm", signals.collision_harm)
    if collision_harm < 0:
        raise ValueError("collision_harm must be non-negative")

    progress = weights.progress * progress_delta
    grasp = weights.grasp_stability * grasp_stability
    harm = weights.collision_harm * collision_harm
    irreversible = weights.irreversible_penalty if bool(signals.irreversible) else 0.0
    total = progress + grasp - harm - irreversible
    return RecoveryScore(
        total=float(total),
        progress=float(progress),
        grasp_stability=float(grasp),
        collision_harm=float(-harm),
        irreversible_penalty=float(-irreversible),
        irreversible=bool(signals.irreversible),
    )


def select_best_of_k(
    transitions: Sequence[TransitionSignals],
    *,
    k: int | None = None,
    candidate_ids: Sequence[str] | None = None,
    weights: RecoveryScoreWeights | None = None,
) -> BestOfKSelection:
    """Score exactly K candidates and select deterministically.

    All candidates receive the same score computation.  Ties are broken by
    candidate id and then original index, making selection independent of sort
    stability while preserving a deterministic final fallback.
    """

    count = len(transitions)
    matched_k = count if k is None else int(k)
    if matched_k < 1:
        raise ValueError("k must be positive")
    if count != matched_k:
        raise ValueError(f"matched-compute selection requires exactly k={matched_k}, got {count}")

    if candidate_ids is None:
        ids = tuple(f"{index:08d}" for index in range(count))
    else:
        if len(candidate_ids) != count:
            raise ValueError("candidate_ids length must match transitions")
        ids = tuple(str(value) for value in candidate_ids)
        if any(not value for value in ids):
            raise ValueError("candidate ids must be non-empty")
        if len(set(ids)) != count:
            raise ValueError("candidate ids must be unique")

    scores = tuple(score_transition(transition, weights) for transition in transitions)
    # min() over (-score, id, index) gives maximum score and documented tie break.
    selected = min(range(count), key=lambda index: (-scores[index].total, ids[index], index))
    return BestOfKSelection(
        index=selected,
        candidate_id=ids[selected],
        score=scores[selected],
        scores=scores,
        evaluated_count=count,
    )

"""Low-capacity causal dynamic-boundary detector (Phase 1).

Pure NumPy.  Scores are computed online from information available at time t
and before: source action chunk history, queue cursor, proprio, motion history.
Future trajectory / branch outcome / simulator hidden state are never inputs.

Primary preregistered rule: combined = phase OR disagreement OR stagnation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rase.vnext.schema import BoundaryTriggerProvenance


@dataclass(frozen=True)
class DetectorConfig:
    """Frozen thresholds (tuned only on inner development folds).

    All thresholds are in *physical units*: normalized actions are converted to
    physical displacement before scoring (translation x 0.05 m, rotation x
    0.5 rad), because normalized action norms are ~1.0 and carry no stagnation
    semantics for pi0-fast.
    """

    first_eligible_step: int = 4
    max_steps: int = 80
    translation_scale: float = 0.05  # m per normalized unit
    rotation_scale: float = 0.5  # rad per normalized unit
    # phase: local minima of physical displacement magnitude
    phase_window: int = 6
    phase_rel_depth: float = 0.35  # trough depth relative to window max
    phase_threshold: float = 0.010  # trough displacement (m) must be small
    # stagnation: sustained small *state* change (proprio delta), which is the
    # deployable meaning of stagnation: the arm is commanded but not moving.
    # Calibrated on pi0-fast prefix proprio deltas (p10 ~0.01, median ~0.03,
    # min 5-step window mean ~0.007-0.010); 0.012 misfired on normal windows,
    # so 0.006 now only fires on true stalls.
    stagnation_window: int = 5
    stagnation_norm: float = 0.006  # mean proprio delta per step
    stagnation_threshold: float = 0.005  # mean proprio delta <= this over window
    # disagreement: max pair distance among M proposals (inference-only)
    disagreement_threshold: float = 0.25
    disagreement_pairs: tuple[tuple[str, str], ...] = (("candidate.0", "candidate.1"),)
    # combined rule: trigger if any component fires
    combined_any: bool = True


class DynamicBoundaryDetector:
    """Online causal detector; call ``update`` once per source step, then
    ``evaluate`` at each step (or rely on ``update``'s internal check)."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._displacements: list[np.ndarray] = []
        self._proprio: list[np.ndarray] = []
        self._scores: dict[str, float] = {}
        self._proposals: dict[str, np.ndarray] = {}
        self._triggered_at: int | None = None
        self._first_eligible = self.config.first_eligible_step

    def reset(self) -> None:
        self._displacements.clear()
        self._proprio.clear()
        self._scores.clear()
        self._proposals.clear()
        self._triggered_at = None

    def _physical_displacement(self, action: np.ndarray) -> np.ndarray:
        config = self.config
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        displacement = np.zeros(6, dtype=np.float64)
        if value.shape[0] >= 6:
            displacement[:3] = value[:3] * config.translation_scale
            displacement[3:6] = value[3:6] * config.rotation_scale
        return displacement

    def update(self, action: np.ndarray, proprio: np.ndarray | None = None) -> dict[str, float]:
        """Record one executed source action and the resulting proprio state."""
        self._displacements.append(self._physical_displacement(action))
        if proprio is not None:
            self._proprio.append(np.asarray(proprio, dtype=np.float64).reshape(-1))
        scores = self._compute_scores()
        self._scores = scores
        return scores

    def set_proposals(self, proposals: Mapping[str, np.ndarray]) -> None:
        """Register M source proposals (first actions) for disagreement scoring."""
        self._proposals = {str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in proposals.items()}

    def _compute_scores(self) -> dict[str, float]:
        config = self.config
        norms = np.asarray([float(np.linalg.norm(d)) for d in self._displacements])
        phase = 0.0
        if len(norms) >= config.phase_window:
            window = norms[-(config.phase_window + 1):]
            peak = float(window.max())
            trough = float(window[-1])
            if peak > 1e-9:
                rel_depth = (peak - trough) / peak
                if rel_depth >= config.phase_rel_depth and trough <= config.phase_threshold:
                    phase = rel_depth
        stagnation = 0.0
        if len(self._proprio) >= config.stagnation_window:
            deltas = np.asarray([
                float(np.linalg.norm(self._proprio[i] - self._proprio[i - 1]))
                for i in range(len(self._proprio) - config.stagnation_window + 1, len(self._proprio))
            ])
            if len(deltas) >= config.stagnation_window - 1:
                mean_delta = float(deltas.mean())
                if mean_delta <= config.stagnation_norm:
                    stagnation = 1.0 - mean_delta / max(config.stagnation_norm, 1e-9)
        disagreement = 0.0
        if len(self._proposals) >= 2:
            distances: list[float] = []
            for left, right in config.disagreement_pairs:
                if left in self._proposals and right in self._proposals:
                    distances.append(float(np.linalg.norm(
                        self._proposals[left] - self._proposals[right]
                    )))
            if distances:
                disagreement = float(max(distances))
        return {"phase": phase, "disagreement": disagreement, "stagnation": stagnation}

    def evaluate(self, timestep: int, now_s: float) -> BoundaryTriggerProvenance | None:
        """Return the frozen provenance when a trigger fires (or a no-trigger
        record when ``timestep`` reaches max_steps).  Caller decides when to
        stop the prefix; this method returns a record exactly once."""
        config = self.config
        scores = self._scores
        threshold = {
            "phase": config.phase_threshold,
            "disagreement": config.disagreement_threshold,
            "stagnation": config.stagnation_threshold,
        }
        if timestep < self._first_eligible:
            return None
        fired: list[str] = []
        if scores.get("phase", 0.0) > 0.0:
            fired.append("phase")
        if scores.get("disagreement", 0.0) >= config.disagreement_threshold:
            fired.append("disagreement")
        if scores.get("stagnation", 0.0) > 0.0:
            fired.append("stagnation")
        rule = "none"
        trigger_step: int | None = None
        reason: str | None = None
        boundary_step = timestep
        if fired:
            rule = "combined" if len(fired) > 1 or config.combined_any else fired[0]
            if not config.combined_any and len(fired) > 1:
                rule = fired[0]
            trigger_step = timestep
        elif timestep >= config.max_steps:
            reason = "max_steps_without_trigger"
            boundary_step = config.max_steps
        else:
            return None
        provenance = BoundaryTriggerProvenance(
            rule=rule,
            phase_score=float(scores.get("phase", 0.0)),
            disagreement_score=float(scores.get("disagreement", 0.0)),
            stagnation_score=float(scores.get("stagnation", 0.0)),
            threshold=threshold,
            first_eligible_step=self._first_eligible,
            trigger_step=trigger_step,
            no_trigger_reason=reason,
            boundary_step=boundary_step,
            timestamps={"decision": now_s, "env_step": float(timestep)},
        )
        provenance.validate()
        if trigger_step is not None:
            self._triggered_at = trigger_step
        return provenance

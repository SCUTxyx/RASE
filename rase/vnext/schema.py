"""Lossless, variable-dimension contracts for the RASE vNext control layer."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{1,95}$")


def _identifier(name: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {name}: {value!r}")


def _readonly_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalRobotSpec:
    """Embodiment metadata; action/proprio dimensions are semantic, not fixed at seven."""

    spec_id: str
    embodiment_id: str
    action_semantics: tuple[str, ...]
    proprio_semantics: tuple[str, ...]
    control_hz: float
    coordinate_frame: str
    rotation_representation: str
    gripper_range: tuple[float, float] = (-1.0, 1.0)
    camera_roles: tuple[str, ...] = ("agentview", "wrist")

    def validate(self) -> None:
        _identifier("spec_id", self.spec_id)
        _identifier("embodiment_id", self.embodiment_id)
        if not self.action_semantics or len(set(self.action_semantics)) != len(
            self.action_semantics
        ):
            raise ValueError("action semantics must be non-empty and unique")
        if not self.proprio_semantics or len(set(self.proprio_semantics)) != len(
            self.proprio_semantics
        ):
            raise ValueError("proprio semantics must be non-empty and unique")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if not self.coordinate_frame or not self.rotation_representation:
            raise ValueError("coordinate frame and rotation representation are required")
        low, high = self.gripper_range
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("gripper_range must be finite and ordered")
        if not self.camera_roles or len(set(self.camera_roles)) != len(self.camera_roles):
            raise ValueError("camera roles must be non-empty and unique")

    @property
    def action_dim(self) -> int:
        return len(self.action_semantics)

    @property
    def proprio_dim(self) -> int:
        return len(self.proprio_semantics)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CanonicalActionToken:
    """Lossless action-token sequence with per-step and per-dimension masks."""

    values: np.ndarray
    semantics: tuple[str, ...]
    dimension_mask: np.ndarray
    step_mask: np.ndarray
    delta_time_s: np.ndarray
    coordinate_frame: str
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_array(self.values, dtype=np.float32))
        object.__setattr__(
            self, "dimension_mask", _readonly_array(self.dimension_mask, dtype=np.bool_)
        )
        object.__setattr__(self, "step_mask", _readonly_array(self.step_mask, dtype=np.bool_))
        object.__setattr__(
            self, "delta_time_s", _readonly_array(self.delta_time_s, dtype=np.float32)
        )
        self.validate()

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"action values must have shape [H,D], got {self.values.shape}")
        horizon, dimension = self.values.shape
        if len(self.semantics) != dimension or len(set(self.semantics)) != dimension:
            raise ValueError("action semantics must uniquely match the token dimension")
        if self.dimension_mask.shape != (horizon, dimension):
            raise ValueError("dimension_mask must match action values")
        if self.step_mask.shape != (horizon,):
            raise ValueError("step_mask must have shape [H]")
        if self.delta_time_s.shape != (horizon,):
            raise ValueError("delta_time_s must have shape [H]")
        if not np.isfinite(self.values).all():
            raise ValueError("action values must be finite")
        if not np.isfinite(self.delta_time_s).all() or np.any(self.delta_time_s <= 0):
            raise ValueError("delta_time_s must be finite and positive")
        if np.any(self.dimension_mask & ~self.step_mask[:, None]):
            raise ValueError("masked-out steps cannot contain valid action dimensions")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame is required")
        _identifier("policy_id", self.policy_id)

    @property
    def horizon(self) -> int:
        return self.values.shape[0]

    @property
    def action_dim(self) -> int:
        return self.values.shape[1]

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        semantics: tuple[str, ...],
        control_hz: float,
        coordinate_frame: str,
        policy_id: str,
        dimension_mask: np.ndarray | None = None,
        step_mask: np.ndarray | None = None,
    ) -> CanonicalActionToken:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("action values must be one or two dimensional")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        steps = np.ones(array.shape[0], dtype=np.bool_) if step_mask is None else step_mask
        dims = (
            np.broadcast_to(np.asarray(steps, dtype=np.bool_)[:, None], array.shape).copy()
            if dimension_mask is None
            else dimension_mask
        )
        return cls(
            values=array,
            semantics=semantics,
            dimension_mask=dims,
            step_mask=steps,
            delta_time_s=np.full(array.shape[0], 1.0 / control_hz, dtype=np.float32),
            coordinate_frame=coordinate_frame,
            policy_id=policy_id,
        )


@dataclass(frozen=True)
class CanonicalObservation:
    """Public deployable observation; simulator restore state is deliberately absent."""

    images: Mapping[str, np.ndarray]
    proprio: np.ndarray
    proprio_semantics: tuple[str, ...]
    proprio_mask: np.ndarray
    task_text: str
    timestamp_s: float

    def __post_init__(self) -> None:
        frozen_images = {
            str(role): _readonly_array(image, dtype=np.uint8)
            for role, image in self.images.items()
        }
        object.__setattr__(self, "images", frozen_images)
        object.__setattr__(self, "proprio", _readonly_array(self.proprio, dtype=np.float32))
        object.__setattr__(
            self, "proprio_mask", _readonly_array(self.proprio_mask, dtype=np.bool_)
        )
        self.validate()

    def validate(self) -> None:
        if not self.images:
            raise ValueError("at least one camera image is required")
        for role, image in self.images.items():
            if not role or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {role!r} must be HWC RGB")
        if self.proprio.ndim != 1:
            raise ValueError("proprio must be one dimensional")
        if len(self.proprio_semantics) != self.proprio.size:
            raise ValueError("proprio semantics must match proprio dimension")
        if self.proprio_mask.shape != self.proprio.shape:
            raise ValueError("proprio_mask must match proprio")
        if not np.isfinite(self.proprio).all():
            raise ValueError("proprio must be finite")
        if not self.task_text:
            raise ValueError("task_text is required")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")


@dataclass(frozen=True)
class PolicyDescriptor:
    policy_id: str
    family: str
    action_semantics: tuple[str, ...]
    supports_requery: bool
    supports_resample: bool
    supports_short_chunk: bool
    stochastic_sampling: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _identifier("policy_id", self.policy_id)
        _identifier("family", self.family)
        if not self.action_semantics:
            raise ValueError("policy action semantics cannot be empty")


@dataclass(frozen=True)
class SeedLedger:
    """Five independent randomness identities required by the vNext protocol."""

    init_state_id: int
    environment_seed: int
    source_sampling_seed: int
    operator_seed: int
    exact_repeat_replica: int

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        self.validate()
        return tuple(asdict(self).values())  # type: ignore[return-value]


class CorrectionKind(str, Enum):
    CONTINUE = "continue"
    REQUERY = "requery"
    RESAMPLE = "resample"
    FALLBACK = "fallback"
    ABORT = "abort"


@dataclass(frozen=True)
class CorrectionProfile:
    operator_id: str
    kind: CorrectionKind
    candidate_ids: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def validate(self) -> None:
        _identifier("operator_id", self.operator_id)
        for value in self.candidate_ids + self.requires:
            _identifier("operator token", value)
        if self.kind is not CorrectionKind.RESAMPLE and self.candidate_ids:
            raise ValueError("candidate_ids are only valid for the RESAMPLE operator")


def operator_mask(
    profiles: tuple[CorrectionProfile, ...],
    policy: PolicyDescriptor,
    *,
    fallback_available: bool,
    abort_available: bool = True,
) -> dict[str, bool]:
    """Return one mask entry per semantic operator, never per resample candidate."""
    policy.validate()
    result: dict[str, bool] = {}
    for profile in profiles:
        profile.validate()
        if profile.operator_id in result:
            raise ValueError(f"duplicate correction operator: {profile.operator_id}")
        enabled = {
            CorrectionKind.CONTINUE: True,
            CorrectionKind.REQUERY: policy.supports_requery,
            CorrectionKind.RESAMPLE: policy.supports_resample,
            CorrectionKind.FALLBACK: fallback_available,
            CorrectionKind.ABORT: abort_available,
        }[profile.kind]
        result[profile.operator_id] = bool(enabled)
    return result


# --- Event-triggered dynamic boundary (Phase 0 contract) ----------------------

TRIGGER_RULES = ("combined", "phase", "disagreement", "stagnation", "none")


@dataclass(frozen=True)
class BoundaryTriggerProvenance:
    """Causal provenance of a dynamic decision boundary.

    Only information available at time ``trigger_step`` and before may enter the
    scores: observation, source action chunk, queue cursor, motion history.
    Future trajectories, branch outcomes and simulator hidden state are never
    inputs.
    """

    rule: str  # combined / phase / disagreement / stagnation / none
    phase_score: float
    disagreement_score: float
    stagnation_score: float
    threshold: Mapping[str, float]
    first_eligible_step: int
    trigger_step: int | None  # None => no trigger before max_steps
    no_trigger_reason: str | None
    boundary_step: int  # actual boundary used (trigger_step or max_steps)
    timestamps: Mapping[str, float]  # observation/inference/decision/command/env_step

    def validate(self) -> None:
        if self.rule not in TRIGGER_RULES:
            raise ValueError(f"invalid trigger rule: {self.rule!r}")
        for name in ("phase_score", "disagreement_score", "stagnation_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.first_eligible_step < 0 or self.boundary_step < self.first_eligible_step:
            raise ValueError("boundary_step must be >= first_eligible_step")
        if self.trigger_step is not None and self.trigger_step < self.first_eligible_step:
            raise ValueError("trigger_step must be >= first_eligible_step")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingComponents:
    """Raw wall-time decomposition; incremental values are derived only from
    same-window raw components, never recorded directly."""

    source_prefix_inference_s: float
    source_prefix_env_s: float
    source_prefix_total_s: float
    branch_inference_s: float
    branch_env_s: float
    branch_oracle_s: float
    branch_restore_s: float
    branch_total_s: float

    def validate(self, *, tolerance: float = 0.05) -> None:
        values = [float(getattr(self, field_name)) for field_name in self.__dataclass_fields__]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("timing components must be finite and non-negative")
        prefix_sum = self.source_prefix_inference_s + self.source_prefix_env_s
        if abs(prefix_sum - self.source_prefix_total_s) > tolerance * max(
            self.source_prefix_total_s, 1e-9
        ):
            raise ValueError("prefix components do not sum to total within tolerance")
        branch_sum = (
            self.branch_inference_s + self.branch_env_s
            + self.branch_oracle_s + self.branch_restore_s
        )
        if abs(branch_sum - self.branch_total_s) > tolerance * max(self.branch_total_s, 1e-9):
            raise ValueError("branch components do not sum to total within tolerance")

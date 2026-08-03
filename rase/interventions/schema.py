"""Versioned, deployment-safe contracts for RASE-UI intervention data."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

INTERVENTION_REGISTRY_SCHEMA_VERSION = "rase-intervention-registry/v1"
INTERVENTION_SNAPSHOT_SCHEMA_VERSION = "rase-intervention-snapshot/v1"
INTERVENTION_OUTCOME_SCHEMA_VERSION = "rase-intervention-outcome/v1"

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,95}$")
_SNAPSHOT_RE = re.compile(r"^[A-Za-z0-9_.:-]{3,160}$")


class OperatorFamily(str, Enum):
    """Core operator families frozen for the first RASE-UI benchmark."""

    CONTINUE = "continue"
    REPLAN = "replan"
    LOCAL_CORRECT = "local_correct"
    REWIND = "rewind"
    SWITCH_POLICY = "switch_policy"
    ABSTAIN = "abstain"


def _finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _json_value(value: Any, *, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} mappings require non-empty string keys")
            _json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__qualname__}")


@dataclass(frozen=True)
class CostVector:
    """Measured intervention costs; scalar utility cost remains separately frozen."""

    compute_seconds: float = 0.0
    latency_seconds: float = 0.0
    env_steps: int = 0
    energy_joules: float = 0.0
    progress_loss: float = 0.0
    human_seconds: float = 0.0
    safety_penalty: float = 0.0

    def validate(self) -> None:
        for name in (
            "compute_seconds",
            "latency_seconds",
            "energy_joules",
            "progress_loss",
            "human_seconds",
            "safety_penalty",
        ):
            _finite_non_negative(name, float(getattr(self, name)))
        if isinstance(self.env_steps, bool) or self.env_steps < 0:
            raise ValueError("env_steps must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> CostVector:
        result = cls(**dict(value or {}))
        result.validate()
        return result


@dataclass(frozen=True)
class Feasibility:
    feasible: bool
    reason_codes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.feasible and self.reason_codes:
            raise ValueError("a feasible operator cannot carry infeasibility reasons")
        if not self.feasible and not self.reason_codes:
            raise ValueError("an infeasible operator requires at least one reason code")
        for reason in self.reason_codes:
            if not _IDENTIFIER_RE.fullmatch(reason):
                raise ValueError(f"invalid feasibility reason code: {reason!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"feasible": self.feasible, "reason_codes": list(self.reason_codes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Feasibility:
        result = cls(
            feasible=bool(value["feasible"]),
            reason_codes=tuple(str(item) for item in value.get("reason_codes") or ()),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class OperatorSpec:
    """One fixed executable profile, not a free-form family label."""

    operator_id: str
    family: OperatorFamily
    executor: str
    recovery_target: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    enabled_for_pilot: bool = True

    def validate(self) -> None:
        for name, value in (
            ("operator_id", self.operator_id),
            ("executor", self.executor),
            ("recovery_target", self.recovery_target),
        ):
            if not _IDENTIFIER_RE.fullmatch(value):
                raise ValueError(f"invalid {name}: {value!r}")
        for requirement in self.requires:
            if not _IDENTIFIER_RE.fullmatch(requirement):
                raise ValueError(f"invalid operator requirement: {requirement!r}")
        _json_value(self.parameters, path="parameters")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "operator_id": self.operator_id,
            "family": self.family.value,
            "executor": self.executor,
            "recovery_target": self.recovery_target,
            "parameters": dict(self.parameters),
            "requires": list(self.requires),
            "enabled_for_pilot": self.enabled_for_pilot,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorSpec:
        result = cls(
            operator_id=str(value["operator_id"]),
            family=OperatorFamily(str(value["family"])),
            executor=str(value["executor"]),
            recovery_target=str(value["recovery_target"]),
            parameters=dict(value.get("parameters") or {}),
            requires=tuple(str(item) for item in value.get("requires") or ()),
            enabled_for_pilot=bool(value.get("enabled_for_pilot", True)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class InterventionSnapshot:
    """Public decision identity separated from privileged restore state."""

    snapshot_id: str
    state_key: str
    task_id: str
    episode_id: str
    step: int
    source_policy: str
    restore_state_ref: str
    public_history_ref: str | None = None
    active_action_suffix_ref: str | None = None
    suite: str | None = None
    cohort: str | None = None
    perturbation: Mapping[str, Any] = field(default_factory=dict)
    split: str | None = None
    schema_version: str = INTERVENTION_SNAPSHOT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != INTERVENTION_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema {self.schema_version!r}")
        if not _SNAPSHOT_RE.fullmatch(self.snapshot_id):
            raise ValueError(f"invalid snapshot_id: {self.snapshot_id!r}")
        for name, value in (
            ("state_key", self.state_key),
            ("task_id", self.task_id),
            ("episode_id", self.episode_id),
            ("source_policy", self.source_policy),
            ("restore_state_ref", self.restore_state_ref),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        _json_value(self.perturbation, path="perturbation")

    @property
    def supports_true_continue(self) -> bool:
        return bool(self.active_action_suffix_ref)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InterventionSnapshot:
        result = cls(**dict(value))
        result.validate()
        return result


@dataclass(frozen=True)
class InterventionOutcome:
    """One real continuation outcome for a fixed snapshot/operator/seed arm."""

    snapshot_id: str
    operator_id: str
    continuation_seed: int
    feasibility: Feasibility
    observed: bool
    success: bool | None
    operator_completed: bool
    stop_reason: str
    utility_cost: float
    cost_source: str
    costs: CostVector = field(default_factory=CostVector)
    progress_before: float | None = None
    progress_after: float | None = None
    safety_violation: bool = False
    harm: bool | None = None
    outcome_semantics: str = ""
    proxy: bool = False
    schema_version: str = INTERVENTION_OUTCOME_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != INTERVENTION_OUTCOME_SCHEMA_VERSION:
            raise ValueError(f"unsupported outcome schema {self.schema_version!r}")
        if not _SNAPSHOT_RE.fullmatch(self.snapshot_id):
            raise ValueError(f"invalid snapshot_id: {self.snapshot_id!r}")
        if not _IDENTIFIER_RE.fullmatch(self.operator_id):
            raise ValueError(f"invalid operator_id: {self.operator_id!r}")
        if self.continuation_seed < 0:
            raise ValueError("continuation_seed must be non-negative")
        self.feasibility.validate()
        self.costs.validate()
        _finite_non_negative("utility_cost", float(self.utility_cost))
        if not _IDENTIFIER_RE.fullmatch(self.cost_source):
            raise ValueError(f"invalid cost_source: {self.cost_source!r}")
        if self.observed and not self.feasibility.feasible:
            raise ValueError("an infeasible operator cannot have an observed rollout")
        if self.observed != (self.success is not None):
            raise ValueError("success must be present exactly when the rollout was observed")
        if not self.observed and self.operator_completed:
            raise ValueError("an unobserved rollout cannot have completed its operator")
        if not self.stop_reason:
            raise ValueError("stop_reason must be non-empty")
        for name in ("progress_before", "progress_after"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")

    @property
    def arm_key(self) -> tuple[str, str, int]:
        return (self.snapshot_id, self.operator_id, self.continuation_seed)

    def utility(self, *, success_reward: float = 1.0) -> float | None:
        if not self.observed:
            return None
        return success_reward * float(bool(self.success)) - float(self.utility_cost)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["feasibility"] = self.feasibility.to_dict()
        value["costs"] = self.costs.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InterventionOutcome:
        fields = dict(value)
        fields["feasibility"] = Feasibility.from_dict(dict(fields["feasibility"]))
        fields["costs"] = CostVector.from_dict(fields.get("costs"))
        result = cls(**fields)
        result.validate()
        return result

"""Reusable PRE-C0 same-policy corrective inference primitives.

This module is intentionally lightweight: arm construction, hashing, action
validation, and oracle aggregation can all be tested without torch or LIBERO.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from rase.collect.forked_rollout import InProcessSmolVLAContinuation

ArmFamily = Literal[
    "current_suffix",
    "strict_resample",
    "fresh_replan",
    "receding_horizon",
]

PRE_C0_ARM_SPEC_VERSION = "rase-pre-c0-same-policy-arm/v1"
RECEDING_HORIZONS = (1, 2, 4)


def _canonicalize(value: Any) -> Any:
    """Convert common scientific-Python values to deterministic JSON data."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__qualname__}")


def deterministic_sha256(value: Any) -> str:
    """Hash a value through a stable, type-aware canonical representation."""
    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_action_tensor(actions: Any, *, name: str = "actions") -> np.ndarray:
    """Return contiguous float32 actions after enforcing finite ``[T, 7]``."""
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 7:
        raise ValueError(f"{name} must have shape [T, 7] with T >= 1, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def action_tensor_sha256(actions: Any) -> str:
    """Hash validated actions, including canonical dtype and shape."""
    array = validate_action_tensor(actions)
    digest = hashlib.sha256()
    digest.update(b"rase-actions/float32/v1\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def provenance_fingerprint(provenance: Mapping[str, Any]) -> str:
    """Hash non-empty run provenance."""
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("provenance must be a non-empty mapping")
    return deterministic_sha256(provenance)


def _normalize_seeds(seeds: Sequence[int], *, name: str) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError(f"{name} seeds must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} seeds must be unique")
    if any(seed < 0 or seed > 2**32 - 1 for seed in normalized):
        raise ValueError(f"{name} seeds must be in [0, 2**32 - 1]")
    return normalized


@dataclass(frozen=True)
class CorrectiveArmSpec:
    """Frozen family-level specification for one PRE-C0 audit arm."""

    name: str
    family: ArmFamily
    seeds: tuple[int, ...]
    execution_horizon: int | None
    uses_active_suffix: bool
    fresh_cache: bool
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm name must be non-empty")
        if self.family not in {
            "current_suffix",
            "strict_resample",
            "fresh_replan",
            "receding_horizon",
        }:
            raise ValueError(f"unsupported arm family: {self.family}")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if self.family == "receding_horizon" and self.execution_horizon not in RECEDING_HORIZONS:
            raise ValueError(f"receding horizon must be one of {RECEDING_HORIZONS}")
        if self.family != "receding_horizon" and self.execution_horizon is not None:
            raise ValueError("execution_horizon is only valid for receding_horizon arms")
        if self.family == "current_suffix" and self.seeds:
            raise ValueError("current_suffix reuses the active suffix and has no seeds")
        if self.family != "current_suffix" and not self.seeds:
            raise ValueError(f"{self.family} requires at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("arm seeds must be unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("arm seeds must be in [0, 2**32 - 1]")
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("provenance must be a non-empty mapping")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def fingerprint(self) -> str:
        return deterministic_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PRE_C0_ARM_SPEC_VERSION,
            "name": self.name,
            "family": self.family,
            "seeds": list(self.seeds),
            "execution_horizon": self.execution_horizon,
            "uses_active_suffix": self.uses_active_suffix,
            "fresh_cache": self.fresh_cache,
            "provenance": dict(self.provenance),
        }


def build_pre_c0_arm_specs(
    *,
    strict_resample_seeds: Sequence[int],
    fresh_replan_seeds: Sequence[int],
    provenance: Mapping[str, Any],
    receding_horizons: Sequence[int] = RECEDING_HORIZONS,
) -> tuple[CorrectiveArmSpec, ...]:
    """Build the frozen current/resample/replan/receding PRE-C0 arm families."""
    provenance_fingerprint(provenance)
    strict = _normalize_seeds(strict_resample_seeds, name="strict_resample")
    fresh = _normalize_seeds(fresh_replan_seeds, name="fresh_replan")
    horizons = tuple(int(horizon) for horizon in receding_horizons)
    if horizons != RECEDING_HORIZONS:
        raise ValueError(f"PRE-C0 receding horizons must be exactly {RECEDING_HORIZONS}")
    common = dict(provenance)
    return (
        CorrectiveArmSpec(
            "current_suffix", "current_suffix", (), None, True, False, common
        ),
        CorrectiveArmSpec(
            "strict_resample", "strict_resample", strict, None, False, False, common
        ),
        CorrectiveArmSpec(
            "fresh_replan", "fresh_replan", fresh, None, False, True, common
        ),
        *(
            CorrectiveArmSpec(
                f"receding_horizon@{horizon}",
                "receding_horizon",
                fresh,
                horizon,
                False,
                True,
                common,
            )
            for horizon in horizons
        ),
    )


class RecedingHorizonSmolVLAContinuation:
    """ContinuationPolicy that flushes SmolVLA's action queue every H actions.

    At each receding boundary this calls ``continuation.reset()``, which must
    clear the policy action queue so the next ``act()`` triggers a fresh model
    forward. Metrics expose forward/queue-reset counts for PRE-C1.2 invariants.
    """

    def __init__(
        self,
        policy_bundle: Mapping[str, Any] | None = None,
        *,
        execution_horizon: int,
        temperature: float = 0.5,
        seed: int | None = None,
        continuation: Any | None = None,
    ) -> None:
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if continuation is not None and policy_bundle is not None:
            raise ValueError("pass either policy_bundle or continuation, not both")
        if continuation is None:
            if policy_bundle is None:
                raise ValueError("policy_bundle or continuation is required")
            continuation = InProcessSmolVLAContinuation(
                policy_bundle, temperature=temperature, seed=seed
            )
        if not callable(getattr(continuation, "reset", None)) or not callable(
            getattr(continuation, "act", None)
        ):
            raise TypeError("continuation must implement reset() and act()")
        self.execution_horizon = int(execution_horizon)
        self.continuation = continuation
        self._actions_since_reset = 0
        self._cache_resets = 0
        self._total_actions = 0
        self._boundary_queue_clears = 0

    def reset(self) -> None:
        reset_metrics = getattr(self.continuation, "reset_metrics", None)
        if callable(reset_metrics):
            reset_metrics()
        self.continuation.reset()
        self._actions_since_reset = 0
        self._cache_resets = 1
        self._total_actions = 0
        self._boundary_queue_clears = 1

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray:
        if self._actions_since_reset >= self.execution_horizon:
            self.continuation.reset()
            self._actions_since_reset = 0
            self._cache_resets += 1
            self._boundary_queue_clears += 1
        action = np.asarray(self.continuation.act(observation, task=task))
        validated = validate_action_tensor(
            action[None, :] if action.ndim == 1 else action,
            name="continuation action",
        )
        if validated.shape[0] != 1:
            raise ValueError(
                f"continuation act() must return one action [7], got {action.shape}"
            )
        self._actions_since_reset += 1
        self._total_actions += 1
        return validated[0]

    def metrics(self) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        nested_fn = getattr(self.continuation, "metrics", None)
        if callable(nested_fn):
            nested = dict(nested_fn())
        model_forward_calls = int(nested.get("model_forward_calls", 0) or 0)
        action_queue_resets = int(
            nested.get("action_queue_resets", self._boundary_queue_clears) or 0
        )
        mean_steps = (
            float(self._total_actions) / float(model_forward_calls)
            if model_forward_calls > 0
            else 0.0
        )
        metrics: dict[str, Any] = {
            "execution_horizon": self.execution_horizon,
            "cache_resets": self._cache_resets,
            "actions": self._total_actions,
            "env_steps": self._total_actions,
            "model_forward_calls": model_forward_calls,
            "action_queue_resets": action_queue_resets,
            "boundary_queue_clears": self._boundary_queue_clears,
            "mean_steps_per_forward": mean_steps,
            "force_fresh_forward_at_boundary": True,
            "clear_action_queue_at_boundary": True,
            "wrapped": nested,
        }
        return metrics


def _arm_success(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Mapping):
        if "success" not in value:
            raise ValueError("arm outcome mappings must contain 'success'")
        return bool(value["success"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_arm_success(item) for item in value)
    raise TypeError(f"unsupported arm outcome: {type(value).__qualname__}")


def nested_oracle_metrics(
    per_state_arm_outcomes: Mapping[str, Mapping[str, Any]],
    *,
    arm_specs: Sequence[CorrectiveArmSpec] | None = None,
) -> dict[str, Any]:
    """Compute nested S0/S1/S2/S3 oracles over matched per-state outcomes."""
    if not per_state_arm_outcomes:
        raise ValueError("per_state_arm_outcomes must be non-empty")
    family_by_name = (
        {spec.name: spec.family for spec in arm_specs}
        if arm_specs is not None
        else {
            name: (
                "receding_horizon"
                if name.startswith("receding_horizon@")
                else name.split("@", 1)[0]
            )
            for outcomes in per_state_arm_outcomes.values()
            for name in outcomes
        }
    )
    required = {"current_suffix", "strict_resample", "fresh_replan", "receding_horizon"}
    if not required.issubset(set(family_by_name.values())):
        missing = sorted(required - set(family_by_name.values()))
        raise ValueError(f"missing arm families: {missing}")

    counts = [0, 0, 0, 0]
    per_state: dict[str, dict[str, bool]] = {}
    for state_key, outcomes in per_state_arm_outcomes.items():
        if not isinstance(outcomes, Mapping):
            raise TypeError(f"outcomes for {state_key!r} must be a mapping")
        family_success = {family: False for family in required}
        for arm_name, value in outcomes.items():
            if arm_name not in family_by_name:
                raise ValueError(f"unknown arm {arm_name!r} for state {state_key!r}")
            family_success[family_by_name[arm_name]] |= _arm_success(value)
        missing = [family for family in required if not any(
            family_by_name.get(name) == family for name in outcomes
        )]
        if missing:
            raise ValueError(f"state {state_key!r} missing arm families: {sorted(missing)}")
        s0 = family_success["current_suffix"]
        s1 = s0 or family_success["strict_resample"]
        s2 = s1 or family_success["fresh_replan"]
        s3 = s2 or family_success["receding_horizon"]
        nested = (s0, s1, s2, s3)
        counts = [count + int(success) for count, success in zip(counts, nested)]
        per_state[str(state_key)] = {
            f"S{index}": success for index, success in enumerate(nested)
        }

    n_states = len(per_state_arm_outcomes)
    rates = [count / n_states for count in counts]
    return {
        "n_states": n_states,
        "successes": {f"S{index}": count for index, count in enumerate(counts)},
        "rates": {f"S{index}": rate for index, rate in enumerate(rates)},
        "headroom": {
            "H_sampling": rates[1] - rates[0],
            "H_reconditioning": rates[2] - rates[1],
            "H_closed_loop": rates[3] - rates[2],
            "H_total": rates[3] - rates[0],
        },
        "per_state": per_state,
    }

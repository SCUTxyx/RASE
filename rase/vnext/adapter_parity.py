"""CPU-only parity and empirical capability checks for VLA adapters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .motion_trace import MotionSemanticMap, action_to_motion_trace
from .schema import CanonicalActionToken, PolicyDescriptor


class ActionCodec(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    def raw_to_canonical(self, value: np.ndarray) -> CanonicalActionToken: ...

    def canonical_to_raw(self, token: CanonicalActionToken) -> np.ndarray: ...


@dataclass(frozen=True)
class ParityResult:
    check_id: str
    status: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_action_roundtrip(
    adapter: ActionCodec,
    samples: Sequence[np.ndarray],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> ParityResult:
    """Verify raw -> canonical -> raw without importing a policy model."""
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("roundtrip tolerances must be non-negative")
    if not samples:
        raise ValueError("roundtrip audit requires at least one sample")
    failures: list[dict[str, Any]] = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for index, sample in enumerate(samples):
        raw = np.asarray(sample, dtype=np.float32)
        token = adapter.raw_to_canonical(raw)
        restored = np.asarray(adapter.canonical_to_raw(token), dtype=np.float32)
        expected = raw.reshape(1, -1) if raw.ndim == 1 else raw
        if restored.shape != expected.shape:
            failures.append({
                "sample": index,
                "reason": "shape_mismatch",
                "expected": list(expected.shape),
                "observed": list(restored.shape),
            })
            continue
        absolute = np.abs(restored.astype(np.float64) - expected.astype(np.float64))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
        relative = absolute / denominator
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(absolute, initial=0.0)))
        maximum_relative_error = max(maximum_relative_error, float(np.max(relative, initial=0.0)))
        if not np.allclose(
            restored, expected, atol=absolute_tolerance, rtol=relative_tolerance,
        ):
            failures.append({
                "sample": index,
                "reason": "value_mismatch",
                "maximum_absolute_error": float(np.max(absolute)),
                "maximum_relative_error": float(np.max(relative)),
            })
    return ParityResult(
        check_id="action_roundtrip",
        status="PASS" if not failures else "FAIL",
        details={
            "policy_id": adapter.descriptor.policy_id,
            "samples": len(samples),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
            "failures": failures,
        },
    )


def audit_motion_trace_conversion(
    tokens: Sequence[CanonicalActionToken],
    *,
    semantic_map: MotionSemanticMap | None = None,
) -> ParityResult:
    """Check that every deployable token maps without silent dimension invention."""
    if not tokens:
        raise ValueError("motion trace audit requires at least one token")
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        try:
            trace = action_to_motion_trace(token, semantic_map=semantic_map)
        except (TypeError, ValueError) as exc:
            failures.append({"sample": index, "reason": str(exc)})
            continue
        summary = trace.summary()
        summaries.append(summary)
        if token.step_mask.any() and not trace.kinematic_map_valid:
            failures.append({
                "sample": index,
                "reason": "deployable action lacks a complete verified 6-DoF map",
            })
    return ParityResult(
        check_id="motion_trace_conversion",
        status="PASS" if not failures else "FAIL",
        details={"samples": len(tokens), "summaries": summaries, "failures": failures},
    )


def audit_resample_capability(
    candidate_groups: Iterable[Sequence[np.ndarray]],
    *,
    absolute_tolerance: float = 1e-7,
    minimum_distinct_fraction: float = 0.1,
) -> ParityResult:
    """Empirically determine whether resample produces distinct action chunks.

    The function recommends a capability mask but never mutates a descriptor or
    manifest.  A group is distinct when any candidate differs from candidate 0
    beyond ``absolute_tolerance`` after shape checking.
    """
    if absolute_tolerance < 0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if not 0 <= minimum_distinct_fraction <= 1:
        raise ValueError("minimum_distinct_fraction must lie in [0,1]")
    total = 0
    distinct = 0
    invalid_groups: list[dict[str, Any]] = []
    for group_index, candidates in enumerate(candidate_groups):
        total += 1
        if len(candidates) < 2:
            invalid_groups.append({"group": group_index, "reason": "fewer_than_two_candidates"})
            continue
        arrays = [np.asarray(candidate, dtype=np.float64) for candidate in candidates]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            invalid_groups.append({"group": group_index, "reason": "shape_mismatch"})
            continue
        if any(not np.isfinite(array).all() for array in arrays):
            invalid_groups.append({"group": group_index, "reason": "non_finite_candidate"})
            continue
        if any(
            not np.allclose(arrays[0], candidate, atol=absolute_tolerance, rtol=0)
            for candidate in arrays[1:]
        ):
            distinct += 1
    valid = total - len(invalid_groups)
    fraction = distinct / valid if valid else 0.0
    supported = valid > 0 and fraction >= minimum_distinct_fraction
    return ParityResult(
        check_id="resample_capability",
        status="PASS" if supported and not invalid_groups else "FAIL",
        details={
            "groups": total,
            "valid_groups": valid,
            "distinct_groups": distinct,
            "distinct_fraction": fraction,
            "minimum_distinct_fraction": minimum_distinct_fraction,
            "recommended_capability_mask": not supported,
            "invalid_groups": invalid_groups,
        },
    )


def build_capability_report(
    descriptor: PolicyDescriptor,
    *,
    resample_audit: ParityResult | None = None,
    fallback_available: bool,
    abort_available: bool = True,
) -> dict[str, Any]:
    """Combine declared and observed capabilities without hiding disagreement."""
    descriptor.validate()
    empirical_resample = None
    if resample_audit is not None:
        if resample_audit.check_id != "resample_capability":
            raise ValueError("resample_audit has the wrong check_id")
        empirical_resample = (
            resample_audit.status == "PASS"
            and not resample_audit.details["recommended_capability_mask"]
        )
    effective_resample = descriptor.supports_resample and (
        empirical_resample if empirical_resample is not None else True
    )
    disagreements = []
    if empirical_resample is not None and descriptor.supports_resample != empirical_resample:
        disagreements.append("declared_resample_differs_from_empirical_behavior")
    return {
        "schema_version": "rase-vnext-operator-capability/v1",
        "policy_id": descriptor.policy_id,
        "family": descriptor.family,
        "declared": {
            "requery.source": descriptor.supports_requery,
            "resample.source": descriptor.supports_resample,
            "fallback.persistent": fallback_available,
            "abort.safe": abort_available,
        },
        "empirical": {"resample.source": empirical_resample},
        "effective_mask": {
            "continue.source": True,
            "requery.source": descriptor.supports_requery,
            "resample.source": effective_resample,
            "fallback.persistent": fallback_available,
            "abort.safe": abort_available,
        },
        "disagreements": disagreements,
        "status": "PASS" if not disagreements else "REVIEW_REQUIRED",
    }

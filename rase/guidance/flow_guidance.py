"""Generic numerical guidance for action chunks.

The module intentionally has no dependency on, or integration claim for, any
particular policy implementation.  Callers provide numerical ``[T, 7]`` action
chunks and guidance arrays (or a callback that computes them).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_DIM = 7


@dataclass(frozen=True)
class GuidanceResult:
    actions: np.ndarray
    used_fallback: bool
    reason: str | None
    raw_guidance_norm: float
    applied_guidance_norm: float
    update_norm: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GuidanceResult):
            return NotImplemented
        return (
            np.array_equal(self.actions, other.actions)
            and self.used_fallback == other.used_fallback
            and self.reason == other.reason
            and (
                self.raw_guidance_norm == other.raw_guidance_norm
                or (
                    np.isnan(self.raw_guidance_norm)
                    and np.isnan(other.raw_guidance_norm)
                )
            )
            and self.applied_guidance_norm == other.applied_guidance_norm
            and self.update_norm == other.update_norm
        )


def _action_chunk(name: str, value: Any, *, finite: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} must have shape [T,7], got {array.shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _bounds(
    action_low: Any,
    action_high: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        low = np.broadcast_to(np.asarray(action_low, dtype=np.float64), shape)
        high = np.broadcast_to(np.asarray(action_high, dtype=np.float64), shape)
    except ValueError as exc:
        raise ValueError("action bounds must broadcast to [T,7]") from exc
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("action bounds must be finite")
    if np.any(low > high):
        raise ValueError("each action lower bound must be <= its upper bound")
    return low, high


def _non_negative_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def clip_by_norm(value: Any, max_norm: float) -> np.ndarray:
    """Clip an array to a global L2 norm without changing its direction."""

    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("value must contain only finite values")
    limit = _non_negative_finite("max_norm", max_norm)
    norm = float(np.linalg.norm(array.ravel()))
    if norm == 0.0 or norm <= limit:
        return np.array(array, copy=True)
    return np.asarray(array * (limit / norm), dtype=np.float64)


def project_to_trust_region(
    actions: Any,
    reference_actions: Any,
    radius: float,
) -> np.ndarray:
    """Project actions onto a global L2 ball around ``reference_actions``."""

    proposed = _action_chunk("actions", actions)
    reference = _action_chunk("reference_actions", reference_actions)
    if proposed.shape != reference.shape:
        raise ValueError("actions and reference_actions must have identical shapes")
    trust_radius = _non_negative_finite("radius", radius)
    delta = clip_by_norm(proposed - reference, trust_radius)
    projected = reference + delta
    if not np.all(np.isfinite(projected)):
        raise FloatingPointError("trust-region projection produced non-finite actions")
    return projected


def _safe_reference(
    base_actions: Any,
    fallback_actions: Any | None,
    action_low: Any,
    action_high: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _action_chunk("base_actions", base_actions, finite=False)
    low, high = _bounds(action_low, action_high, base.shape)
    fallback = base if fallback_actions is None else _action_chunk("fallback_actions", fallback_actions)
    if fallback.shape != base.shape:
        raise ValueError("fallback_actions must have the same shape as base_actions")
    if not np.all(np.isfinite(fallback)):
        raise ValueError("a finite fallback is required when base_actions are non-finite")
    return np.clip(fallback, low, high), low, high


def apply_guidance_update(
    base_actions: Any,
    guidance: Any,
    *,
    step_size: float,
    action_low: Any,
    action_high: Any,
    trust_region_radius: float,
    max_guidance_norm: float,
    fallback_actions: Any | None = None,
) -> GuidanceResult:
    """Apply one bounded, norm-clipped trust-region guidance update.

    Invalid guidance or numerical divergence returns a finite bounded fallback.
    Configuration and shape errors raise ``ValueError`` because silently
    falling back cannot repair a malformed safety contract.
    """

    safe, low, high = _safe_reference(
        base_actions, fallback_actions, action_low, action_high
    )
    base = _action_chunk("base_actions", base_actions, finite=False)
    step = _non_negative_finite("step_size", step_size)
    radius = _non_negative_finite("trust_region_radius", trust_region_radius)
    max_norm = _non_negative_finite("max_guidance_norm", max_guidance_norm)

    direction = np.asarray(guidance, dtype=np.float64)
    if direction.shape != base.shape:
        raise ValueError(f"guidance must have shape {base.shape}, got {direction.shape}")
    if not np.all(np.isfinite(base)):
        return GuidanceResult(safe, True, "non_finite_base", float("nan"), 0.0, 0.0)
    if not np.all(np.isfinite(direction)):
        return GuidanceResult(safe, True, "non_finite_guidance", float("nan"), 0.0, 0.0)

    raw_norm = float(np.linalg.norm(direction.ravel()))
    clipped = clip_by_norm(direction, max_norm)
    applied_norm = float(np.linalg.norm(clipped.ravel()))
    with np.errstate(over="ignore", invalid="ignore"):
        proposed = safe + step * clipped
    if not np.all(np.isfinite(proposed)):
        return GuidanceResult(safe, True, "divergent_update", raw_norm, applied_norm, 0.0)

    projected = project_to_trust_region(proposed, safe, radius)
    bounded = np.clip(projected, low, high)
    # Convex box projection cannot increase distance from an in-box reference.
    update_norm = float(np.linalg.norm((bounded - safe).ravel()))
    if not np.all(np.isfinite(bounded)) or update_norm > radius + 1e-12:
        return GuidanceResult(safe, True, "unsafe_projection", raw_norm, applied_norm, 0.0)
    return GuidanceResult(
        actions=bounded,
        used_fallback=False,
        reason=None,
        raw_guidance_norm=raw_norm,
        applied_guidance_norm=applied_norm,
        update_norm=update_norm,
    )


def iterative_guidance(
    base_actions: Any,
    guidance_fn: Callable[[np.ndarray, int], Any],
    *,
    num_steps: int,
    step_size: float,
    action_low: Any,
    action_high: Any,
    trust_region_radius: float,
    max_guidance_norm: float,
) -> GuidanceResult:
    """Run deterministic callback guidance while staying near the original chunk."""

    steps = int(num_steps)
    if steps < 0 or steps != num_steps:
        raise ValueError("num_steps must be a non-negative integer")
    original = _action_chunk("base_actions", base_actions)
    low, high = _bounds(action_low, action_high, original.shape)
    reference = np.clip(original, low, high)
    current = reference.copy()
    last_raw_norm = 0.0
    last_applied_norm = 0.0

    for index in range(steps):
        try:
            direction = guidance_fn(current.copy(), index)
        except (FloatingPointError, OverflowError, ValueError):
            return GuidanceResult(reference, True, "guidance_callback_error", 0.0, 0.0, 0.0)
        result = apply_guidance_update(
            current,
            direction,
            step_size=step_size,
            action_low=low,
            action_high=high,
            trust_region_radius=trust_region_radius,
            max_guidance_norm=max_guidance_norm,
            fallback_actions=reference,
        )
        if result.used_fallback:
            return GuidanceResult(
                reference,
                True,
                result.reason,
                result.raw_guidance_norm,
                result.applied_guidance_norm,
                0.0,
            )
        current = project_to_trust_region(result.actions, reference, trust_region_radius)
        current = np.clip(current, low, high)
        last_raw_norm = result.raw_guidance_norm
        last_applied_norm = result.applied_guidance_norm

    return GuidanceResult(
        current,
        False,
        None,
        last_raw_norm,
        last_applied_norm,
        float(np.linalg.norm((current - reference).ravel())),
    )

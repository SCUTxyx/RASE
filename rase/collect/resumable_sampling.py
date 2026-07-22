"""Resumable adaptive sampling backed by ``DiskRolloutScheduler``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rase.collect.adaptive import (
    PROTOCOL_LEGACY_TWOSIDED,
    PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    WilsonEstimate,
    estimate,
    sequential_adaptive_sample,
    wilson_interval,
)
from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey


def outcomes_from_scheduler(
    scheduler: DiskRolloutScheduler,
    state_key: str,
    candidate: int,
    *,
    n_total: int,
) -> list[bool | None]:
    """Return length-``n_total`` list with ``None`` for unfinished rollouts."""
    values: list[bool | None] = []
    for index in range(n_total):
        record = scheduler.result(RolloutKey(state_key, candidate, index))
        if record is None:
            values.append(None)
            continue
        result = record.get("result") or {}
        if "success" not in result:
            raise ValueError(
                f"scheduler result missing success for "
                f"{state_key}/c{candidate}/r{index}"
            )
        values.append(bool(result["success"]))
    return values


def estimate_from_completed(outcomes: Sequence[bool | None]) -> WilsonEstimate | None:
    completed = [bool(value) for value in outcomes if value is not None]
    if not completed:
        return None
    return estimate(sum(completed), len(completed))


def adaptive_sample_resumable(
    scheduler: DiskRolloutScheduler,
    state_key: str,
    candidate: int,
    worker: str,
    rollout_fn: Callable[[int], Mapping[str, Any] | bool],
    *,
    threshold: float = 0.5,
    n_first: int = 6,
    n_total: int = 20,
    protocol_version: str = PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    alpha_first: float = 0.01,
    alpha_final: float = 0.04,
    sidedness: str = "one-sided",
) -> WilsonEstimate:
    """Resume-aware two-stage estimator with durable per-rollout records.

    ``rollout_fn(index)`` may return a bool or a mapping containing ``success``.
    Already-complete scheduler keys are never re-executed.
    """
    if n_first <= 0 or n_total < n_first:
        raise ValueError("require 0 < n_first <= n_total")

    def ensure(index: int) -> bool:
        key = RolloutKey(state_key, candidate, index)
        existing = scheduler.result(key)
        if existing is not None:
            return bool(existing["result"]["success"])
        claim = scheduler.claim(key, worker)
        if claim is None:
            # Another worker holds the lease or retries exhausted.
            existing = scheduler.result(key)
            if existing is not None:
                return bool(existing["result"]["success"])
            raise RuntimeError(
                f"unable to claim or complete rollout {key.state}/"
                f"c{key.candidate}/r{key.rollout}"
            )
        try:
            raw = rollout_fn(index)
            if isinstance(raw, Mapping):
                success = bool(raw["success"])
                payload = dict(raw)
            else:
                success = bool(raw)
                payload = {"success": success}
            payload.setdefault("protocol_version", protocol_version)
            scheduler.complete(key, payload, worker=worker)
            return success
        except Exception as exc:
            scheduler.fail(key, repr(exc), worker=worker)
            raise

    if protocol_version == PROTOCOL_LEGACY_TWOSIDED:
        return _legacy_resumable(
            ensure,
            threshold=threshold,
            n_first=n_first,
            n_total=n_total,
        )
    return sequential_adaptive_sample(
        ensure,
        threshold=threshold,
        n_first=n_first,
        n_total=n_total,
        alpha_first=alpha_first,
        alpha_final=alpha_final,
        sidedness=sidedness,
        protocol_version=protocol_version,
    )


def _legacy_resumable(
    ensure: Callable[[int], bool],
    *,
    threshold: float,
    n_first: int,
    n_total: int,
) -> WilsonEstimate:
    successes = sum(ensure(index) for index in range(n_first))
    lower, upper = wilson_interval(successes, n_first)
    current = WilsonEstimate(successes, n_first, successes / n_first, lower, upper)
    if current.upper < threshold or current.lower > threshold:
        return current
    for index in range(n_first, n_total):
        successes += ensure(index)
    lower, upper = wilson_interval(successes, n_total)
    return WilsonEstimate(successes, n_total, successes / n_total, lower, upper)

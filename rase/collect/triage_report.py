"""Rebuildable triage summaries from durable scheduler records."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rase.collect.adaptive import (
    PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    WilsonEstimate,
    estimate,
    triage,
    z_from_alpha,
)
from rase.collect.resumable_sampling import outcomes_from_scheduler
from rase.collect.scheduler import DiskRolloutScheduler


def estimate_from_trials(
    successes: int,
    trials: int,
    *,
    protocol_version: str,
    sidedness: str,
    alpha: float,
    stopped_early: bool,
) -> WilsonEstimate:
    z = z_from_alpha(alpha, sidedness=sidedness)
    return estimate(
        successes,
        trials,
        z=z,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha,
        stopped_early=stopped_early,
    )


def candidate_estimate_from_scheduler(
    scheduler: DiskRolloutScheduler,
    state_key: str,
    candidate: int,
    *,
    n_first: int = 6,
    n_total: int = 20,
    threshold: float = 0.5,
    alpha_first: float = 0.01,
    alpha_final: float = 0.04,
    sidedness: str = "one-sided",
    protocol_version: str = PROTOCOL_SEQUENTIAL_ONESIDED_V1,
) -> WilsonEstimate | None:
    """Recompute the adaptive estimate from completed outcomes only."""
    outcomes = outcomes_from_scheduler(
        scheduler, state_key, candidate, n_total=n_total
    )
    if any(value is None for value in outcomes[:n_first]):
        completed = [bool(v) for v in outcomes if v is not None]
        if not completed:
            return None
        return estimate_from_trials(
            sum(completed),
            len(completed),
            protocol_version=protocol_version,
            sidedness=sidedness,
            alpha=alpha_first,
            stopped_early=False,
        )

    successes = sum(bool(outcomes[i]) for i in range(n_first))
    stage1 = estimate_from_trials(
        successes,
        n_first,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha_first,
        stopped_early=True,
    )
    if stage1.upper < threshold or stage1.lower > threshold:
        return stage1

    if any(value is None for value in outcomes[n_first:n_total]):
        # Boundary case still collecting stage-2; report provisional bound.
        done = [bool(v) for v in outcomes[:n_total] if v is not None]
        return estimate_from_trials(
            sum(done),
            len(done),
            protocol_version=protocol_version,
            sidedness=sidedness,
            alpha=alpha_final,
            stopped_early=False,
        )

    successes = sum(bool(outcomes[i]) for i in range(n_total))
    return estimate_from_trials(
        successes,
        n_total,
        protocol_version=protocol_version,
        sidedness=sidedness,
        alpha=alpha_final,
        stopped_early=False,
    )


def triage_state(
    estimates: Sequence[WilsonEstimate],
    *,
    threshold: float = 0.5,
    set_a_min_good: int = 3,
) -> dict[str, Any]:
    label = triage(
        estimates, threshold=threshold, set_a_min_good=set_a_min_good
    )
    confidently_good = sum(item.lower > threshold for item in estimates)
    return {
        "set_label": label.value,
        "confidently_good_count": confidently_good,
        "max_r_hat": max(item.rate for item in estimates),
        "max_upper": max(item.upper for item in estimates),
        "min_lower": min(item.lower for item in estimates),
        "candidates": [
            {
                "successes": item.successes,
                "trials": item.trials,
                "rate": item.rate,
                "lower": item.lower,
                "upper": item.upper,
                "protocol_version": item.protocol_version,
                "sidedness": item.sidedness,
                "alpha": item.alpha,
                "stopped_early": item.stopped_early,
            }
            for item in estimates
        ],
    }


def summarize_run(
    scheduler: DiskRolloutScheduler,
    state_keys: Sequence[str],
    *,
    k: int = 8,
    n_first: int = 6,
    n_total: int = 20,
    threshold: float = 0.5,
    set_a_min_good: int = 3,
    alpha_first: float = 0.01,
    alpha_final: float = 0.04,
    sidedness: str = "one-sided",
    protocol_version: str = PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    cross_oracle: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-state triage and run-level budget diagnostics."""
    per_state: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    trials_per_candidate: list[int] = []
    elapsed: list[float] = []
    retries = 0

    for state_key in state_keys:
        estimates: list[WilsonEstimate] = []
        for candidate in range(k):
            est = candidate_estimate_from_scheduler(
                scheduler,
                state_key,
                candidate,
                n_first=n_first,
                n_total=n_total,
                threshold=threshold,
                alpha_first=alpha_first,
                alpha_final=alpha_final,
                sidedness=sidedness,
                protocol_version=protocol_version,
            )
            if est is None:
                continue
            estimates.append(est)
            trials_per_candidate.append(est.trials)
            for rollout in range(n_total):
                from rase.collect.scheduler import RolloutKey

                key = RolloutKey(state_key, candidate, rollout)
                retries = max(retries, scheduler.attempts(key))
                record = scheduler.result(key)
                if record and isinstance(record.get("result"), dict):
                    value = record["result"].get("elapsed_s")
                    if value is not None:
                        elapsed.append(float(value))
        if len(estimates) != k:
            per_state.append(
                {
                    "state_key": state_key,
                    "set_label": "incomplete",
                    "n_candidates_ready": len(estimates),
                }
            )
            labels["incomplete"] += 1
            continue
        report = triage_state(
            estimates, threshold=threshold, set_a_min_good=set_a_min_good
        )
        report["state_key"] = state_key
        if cross_oracle and state_key in cross_oracle:
            report["cross_oracle"] = dict(cross_oracle[state_key])
        per_state.append(report)
        labels[report["set_label"]] += 1

    def _percentile(values: Sequence[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(ordered[low])
        weight = position - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    return {
        "n_states": len(state_keys),
        "label_counts": dict(labels),
        "mean_trials_per_candidate": (
            float(statistics.mean(trials_per_candidate)) if trials_per_candidate else 0.0
        ),
        "total_rollouts_completed": sum(trials_per_candidate),
        "max_retries_observed": retries,
        "elapsed_s": {
            "n": len(elapsed),
            "mean": float(statistics.mean(elapsed)) if elapsed else None,
            "median": _percentile(elapsed, 0.5),
            "p90": _percentile(elapsed, 0.9),
        },
        "per_state": per_state,
        "protocol": {
            "version": protocol_version,
            "threshold": threshold,
            "n_first": n_first,
            "n_total": n_total,
            "alpha_first": alpha_first,
            "alpha_final": alpha_final,
            "sidedness": sidedness,
            "set_a_min_good": set_a_min_good,
            "k": k,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

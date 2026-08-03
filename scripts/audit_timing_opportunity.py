#!/usr/bin/env python3
"""Apply the preregistered independent timing-opportunity gate."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def audit(
    analysis: dict[str, Any],
    *,
    min_gap: float,
    min_tasks_per_timing: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = list((analysis.get("three_operator") or {}).get("per_state") or [])
    if not rows:
        raise ValueError("analysis requires three-operator per-state rows")
    tasks = [str(row["task_id"]) for row in rows]
    episodes = [str(row["episode_id"]) for row in rows]
    if len(tasks) != len(set(tasks)) or len(episodes) != len(set(episodes)):
        raise ValueError("independent gate requires one state per task and episode")

    def gap(sample: list[dict[str, Any]]) -> float:
        direct = sum(bool(row["direct_oft_success"]) for row in sample)
        deferred = sum(bool(row["decision_suffix_oft_success"]) for row in sample)
        oracle = sum(
            bool(row["direct_oft_success"] or row["decision_suffix_oft_success"])
            for row in sample
        )
        return (oracle - max(direct, deferred)) / len(sample)

    observed_gap = gap(rows)
    rng = random.Random(bootstrap_seed)
    boot = [
        gap([rows[rng.randrange(len(rows))] for _ in rows])
        for _ in range(bootstrap_replicates)
    ]
    direct_only_tasks = sorted(
        {
            str(row["task_id"])
            for row in rows
            if row["direct_oft_success"] and not row["decision_suffix_oft_success"]
        }
    )
    deferred_only_tasks = sorted(
        {
            str(row["task_id"])
            for row in rows
            if row["decision_suffix_oft_success"] and not row["direct_oft_success"]
        }
    )
    reasons = []
    if observed_gap < min_gap:
        reasons.append(f"oracle gap {observed_gap:.6f} is below {min_gap:.6f}")
    if len(direct_only_tasks) < min_tasks_per_timing:
        reasons.append("insufficient immediate-only task support")
    if len(deferred_only_tasks) < min_tasks_per_timing:
        reasons.append("insufficient deferred-only task support")
    classifications = Counter(str(row["classification"]) for row in rows)
    return {
        "schema_version": "rase-timing-opportunity-audit/v1",
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "n_states": len(rows),
        "n_tasks": len(set(tasks)),
        "n_episodes": len(set(episodes)),
        "classification_counts": dict(sorted(classifications.items())),
        "direct_only_tasks": direct_only_tasks,
        "deferred_only_tasks": deferred_only_tasks,
        "oracle_minus_best_fixed": observed_gap,
        "oracle_gap_bootstrap": {
            "unit": "task/episode (one state per cluster)",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        },
        "thresholds": {
            "min_oracle_gap": min_gap,
            "min_tasks_per_timing": min_tasks_per_timing,
        },
        "use_for": "screening gate only; not independent confirmation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--min-tasks-per-timing", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026081807)
    args = parser.parse_args()
    result = audit(
        _read(args.analysis.resolve()),
        min_gap=args.min_gap,
        min_tasks_per_timing=args.min_tasks_per_timing,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), **result}, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

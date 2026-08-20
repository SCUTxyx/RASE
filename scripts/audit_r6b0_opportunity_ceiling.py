#!/usr/bin/env python3
"""Compute model-free and constrained privileged ceilings for R6-B0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def metrics(data: dict[str, np.ndarray], idx: np.ndarray, cont: np.ndarray) -> dict[str, float]:
    source = data["source_seed_success"][idx].astype(bool)
    persistent = data["persistent_success"][idx].astype(bool)
    steps = data["persistent_teacher_steps"][idx].astype(float)
    decision = np.repeat(cont[:, None], 2, axis=1)
    success = np.where(decision, source, persistent[:, None])
    baseline = np.repeat(persistent[:, None], 2, axis=1)
    false = decision & (~source) & baseline
    teacher = np.where(decision, 0.0, steps[:, None])
    total = np.repeat(steps[:, None], 2, axis=1)
    return {
        "continued_states": int(cont.sum()),
        "success_gap": float((success.sum() - baseline.sum()) / success.size),
        "false_continue_rate": float(false.sum() / max(1, baseline.sum())),
        "savings": float(1 - teacher.sum() / max(1, total.sum())),
        "successes": int(success.sum()),
        "persistent_successes": int(baseline.sum()),
    }


def constrained_oracle(data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    persistent = data["persistent_success"][idx].astype(int)
    source_count = data["source_seed_success"][idx].sum(1).astype(int)
    steps = data["persistent_teacher_steps"][idx].astype(int)
    episode_count = len(idx) * 2
    baseline_success = int((persistent * 2).sum())
    minimum_delta = math.ceil(-0.05 * episode_count - 1e-12)
    maximum_false = math.floor(0.05 * baseline_success + 1e-12)
    # (success delta, false count) -> (saved teacher steps, selected local rows)
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for local in range(len(idx)):
        delta = int(source_count[local] - 2 * persistent[local])
        false = int((2 - source_count[local]) * persistent[local])
        saving = int(2 * steps[local])
        updated = dict(states)
        for (old_delta, old_false), (old_saving, selected) in states.items():
            key = (old_delta + delta, old_false + false)
            candidate = (old_saving + saving, selected + (local,))
            if candidate[0] > updated.get(key, (-1, ()))[0]:
                updated[key] = candidate
        states = updated
    feasible = [value for (delta, false), value in states.items()
                if delta >= minimum_delta and false <= maximum_false]
    if not feasible:
        raise RuntimeError("persistent baseline should make the program feasible")
    _, selected = max(feasible, key=lambda value: value[0])
    decision = np.zeros(len(idx), dtype=bool)
    decision[list(selected)] = True
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    results = {}
    for policy in sorted(set(data["policy_id"].tolist())):
        idx = np.where(data["policy_id"] == policy)[0]
        all_source = np.ones(len(idx), dtype=bool)
        robust = data["source_successes"][idx] == data["source_trials"][idx]
        oracle = constrained_oracle(data, idx)
        results[policy] = {
            "persistent": metrics(data, idx, np.zeros(len(idx), dtype=bool)),
            "always_source": metrics(data, idx, all_source),
            "both_seed_success_oracle": metrics(data, idx, robust),
            "gate_constrained_cost_oracle": metrics(data, idx, oracle),
        }
    result = {
        "schema_version": "rase-r6b0-opportunity-ceiling/v1",
        "status": "complete",
        "constraints": {"success_gap_min": -0.05, "false_continue_rate_max": 0.05},
        "policies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

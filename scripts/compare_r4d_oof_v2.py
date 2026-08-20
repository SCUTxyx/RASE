#!/usr/bin/env python3
"""Paired task-cluster comparison of two R4-D offline OOF controllers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(records: list[dict]) -> tuple[float, float, float]:
    n = max(len(records), 1)
    success = sum(r["policy_success"] for r in records) / n
    persistent_cost = sum(r["persistent_cost"] for r in records)
    policy_cost = sum(r["policy_cost"] for r in records)
    savings = (persistent_cost - policy_cost) / max(persistent_cost, 1e-9)
    harm = sum(r["harm"] for r in records) / n
    return success, savings, harm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    base_payload = json.loads(args.baseline.read_text())
    candidate_payload = json.loads(args.candidate.read_text())
    base = {r["state_key"]: r for r in base_payload["per_state"]}
    candidate = {r["state_key"]: r for r in candidate_payload["per_state"]}
    if set(base) != set(candidate):
        raise ValueError("state sets differ")
    by_task: dict[str, list[str]] = defaultdict(list)
    for key, row in base.items():
        by_task[str(row["task_id"])].append(key)
    tasks = sorted(by_task)
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.n_boot):
        sampled_tasks = rng.choice(tasks, len(tasks), replace=True)
        keys = [key for task in sampled_tasks for key in by_task[str(task)]]
        b = summarize([base[key] for key in keys])
        c = summarize([candidate[key] for key in keys])
        boot.append((c[0] - b[0], c[1] - b[1], c[2] - b[2]))
    boot_array = np.asarray(boot)
    base_point = summarize(list(base.values()))
    candidate_point = summarize(list(candidate.values()))
    diff = np.asarray(candidate_point) - np.asarray(base_point)
    report = {
        "schema_version": "rase-pre-c0-r4d-paired-model-comparison/v2",
        "cluster_unit": "task",
        "n_tasks": len(tasks),
        "n_states": len(base),
        "n_boot": args.n_boot,
        "baseline": dict(zip(("policy_success", "oft_savings", "harm_rate"), base_point)),
        "candidate": dict(zip(("policy_success", "oft_savings", "harm_rate"), candidate_point)),
        "candidate_minus_baseline": {
            "policy_success": float(diff[0]),
            "oft_savings": float(diff[1]),
            "harm_rate": float(diff[2]),
        },
        "ci95": {
            "policy_success": np.quantile(boot_array[:, 0], [0.025, 0.975]).tolist(),
            "oft_savings": np.quantile(boot_array[:, 1], [0.025, 0.975]).tolist(),
            "harm_rate": np.quantile(boot_array[:, 2], [0.025, 0.975]).tolist(),
        },
        "p_candidate_no_better_success": (
            1 + int(np.sum(boot_array[:, 0] <= 0.0))
        ) / (args.n_boot + 1),
        "p_candidate_no_better_savings": (
            1 + int(np.sum(boot_array[:, 1] <= 0.0))
        ) / (args.n_boot + 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

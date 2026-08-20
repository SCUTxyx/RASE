#!/usr/bin/env python3
"""Aggregate fixed five-seed shared multi-VLA source-risk reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase.risk.r7_source_protocol import TRAIN_SEEDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--per-vla-stability", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for seed in TRAIN_SEEDS:
        path = args.input_root / f"seed_{seed}" / f"{args.mode}.json"
        if not path.is_file():
            raise ValueError(f"missing fixed-seed report: {path}")
        row = json.loads(path.read_text())
        if int(row.get("seed", -1)) != seed or row.get("mode") != args.mode:
            raise ValueError(f"seed/mode mismatch: {path}")
        reports.append(row)
    hashes = {row.get("dataset_sha256") for row in reports}
    policy_sets = {tuple(row.get("policies") or []) for row in reports}
    if len(hashes) != 1 or len(policy_sets) != 1:
        raise ValueError("multi-VLA seeds do not bind one dataset/policy set")
    policies = list(next(iter(policy_sets)))
    per_vla = {}
    for path in args.per_vla_stability:
        row = json.loads(path.read_text())
        policy = str(row.get("policy_id"))
        if row.get("status") != "PASS" or row.get("decision") != "FULL_PASS":
            raise ValueError(f"per-VLA baseline is not FULL_PASS: {path}")
        per_vla[policy] = row
    if set(per_vla) != set(policies):
        raise ValueError("per-VLA baselines do not match shared policies")

    seeds_passed = {
        policy: sum(all(bool(value) for value in row["gate_by_policy"][policy].values())
                    for row in reports)
        for policy in policies
    }
    mean_auroc = {
        policy: float(np.mean([row["metrics_by_policy"][policy]["auroc"]
                               for row in reports]))
        for policy in policies
    }
    per_vla_mean = {
        policy: float(per_vla[policy]["aggregate_metrics"]["auroc"]["mean"])
        for policy in policies
    }
    auroc_gap = {policy: mean_auroc[policy] - per_vla_mean[policy] for policy in policies}
    gate = {
        "each_policy_at_least_4_of_5": all(value >= 4 for value in seeds_passed.values()),
        "each_policy_within_0p03_auroc_of_per_vla": all(value >= -0.03 for value in auroc_gap.values()),
    }
    result = {
        "schema_version": "rase-r7c-multivla-stability/v1",
        "status": "PASS" if all(gate.values()) else "FAIL",
        "mode": args.mode, "dataset_sha256": next(iter(hashes)),
        "policies": policies, "seeds_passed_by_policy": seeds_passed,
        "mean_auroc_by_policy": mean_auroc,
        "per_vla_mean_auroc": per_vla_mean,
        "shared_minus_per_vla_auroc": auroc_gap,
        "gate": gate,
        "zero_shot_is_gate": False,
        "unlocks_on_pass": ["leave-one-VLA-out and adaptation curves"],
        "remains_locked": ["selector", "world-model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

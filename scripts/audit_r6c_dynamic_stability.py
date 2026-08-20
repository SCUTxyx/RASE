#!/usr/bin/env python3
"""Aggregate R6-C per-VLA OOF seeds and enforce the 4/5-seed stage gate.

The R6-C stage gate is per qualified VLA: at least ``required_passing`` of the
``seed_count`` training seeds must simultaneously satisfy success gap >= -5pp,
false continue <= 5% and savings >= 20% for that VLA's per-VLA model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def gate(metrics: dict[str, float]) -> bool:
    return (metrics["success_gap"] >= -0.05
            and metrics["false_continue_rate"] <= 0.05
            and metrics["savings"] >= 0.20)


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--mode", default="per_vla")
    parser.add_argument("--required-passing-seeds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_reports: list[dict[str, Any]] = []
    for path_string in args.report:
        path = Path(path_string)
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise ValueError(f"incomplete result: {path}")
        if result.get("mode") != args.mode:
            raise ValueError(f"mode mismatch in {path}: {result.get('mode')}")
        by_policy = {name: value for name, value in result["metrics_by_policy"].items()}
        seed_reports.append({
            "seed": result["seed"],
            "path": str(path),
            "target_policy": result.get("target_policy"),
            "metrics": result["metrics"],
            "metrics_by_policy": by_policy,
        })

    policies: set[str] = set()
    for report in seed_reports:
        policies.update(report["metrics_by_policy"].keys())
    policies = sorted(policies)

    by_policy_seeds: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    for report in seed_reports:
        for policy in policies:
            if policy in report["metrics_by_policy"]:
                by_policy_seeds[policy].append(report)
    if len(seed_reports) < 5:
        raise ValueError(f"expected 5 training seeds, got {len(seed_reports)}")

    policy_results: dict[str, Any] = {}
    for policy in policies:
        member_reports = by_policy_seeds[policy]
        passing = sum(gate(report["metrics_by_policy"][policy]) for report in member_reports)
        policy_results[policy] = {
            "seed_count": len(member_reports),
            "passing_seed_count": passing,
            "required_passing_seed_count": args.required_passing_seeds,
            "policy_gate_passed": passing >= args.required_passing_seeds,
            "metric_across_seeds": {
                key: interval([report["metrics_by_policy"][policy][key] for report in member_reports])
                for key in ["success_gap", "false_continue_rate", "savings"]
            },
            "seed_details": [
                {"seed": report["seed"], "metrics": report["metrics_by_policy"][policy],
                 "seed_gate_passed": gate(report["metrics_by_policy"][policy])}
                for report in member_reports
            ],
        }

    stage_gate_passed = all(value["policy_gate_passed"] for value in policy_results.values())
    result = {
        "schema_version": "rase-r6c-dynamic-stability/v1",
        "status": "complete",
        "scientific_scope": "no-world-model dynamic risk; two-boundary dwell; per-VLA stage gate",
        "mode": args.mode,
        "seed_count": len(seed_reports),
        "required_passing_seed_count": args.required_passing_seeds,
        "policies": policies,
        "policy_results": policy_results,
        "stage_gate_passed": stage_gate_passed,
        "seed_reports": seed_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "policies": policies,
        "stage_gate_passed": stage_gate_passed,
        "policy_results": {key: {k: value[k] for k in ("seed_count", "passing_seed_count", "policy_gate_passed", "metric_across_seeds")}
                           for key, value in policy_results.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

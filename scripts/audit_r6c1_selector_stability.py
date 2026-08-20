#!/usr/bin/env python3
"""R6-C.1 stage-gate aggregation for the early-window stratified selector.

Aggregates the five training-seed OOF reports and enforces the R6-C.1 gate per
qualified VLA (>=4/5 seeds):

  - fold-correct success gap >= -5pp;
  - original-protocol false continue <= 5%;
  - absolute paired harm <= 5%;
  - teacher-step savings >= 20%;
  - no concentrated suite harm (every suite success gap >= -5pp and
    absolute paired harm <= 5%).

Conditional missed-rescue is reported as point estimate + task-cluster interval
only (no under-powered hard gate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def gate(metrics: dict[str, float], by_suite: dict[str, dict[str, float]]) -> bool:
    suite_ok = all(v["success_gap"] >= -0.05 and v["absolute_paired_harm"] <= 0.05
                   for v in by_suite.values())
    return (metrics["success_gap"] >= -0.05
            and metrics["false_continue_rate"] <= 0.05
            and metrics["absolute_paired_harm"] <= 0.05
            and metrics["savings"] >= 0.20
            and suite_ok)


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
    parser.add_argument("--mode", default="shared_calib")
    parser.add_argument("--required-passing-seeds", type=int, default=4)
    parser.add_argument("--expected-seed-count", type=int, default=5)
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
        seed_reports.append({
            "seed": result["seed"],
            "path": str(path),
            "target_policy": result.get("target_policy"),
            "metrics": result["metrics"],
            "metrics_by_policy": result["metrics_by_policy"],
            "metrics_by_suite": result["metrics_by_suite"],
            "metrics_by_policy_suite": result["metrics_by_policy_suite"],
            "metrics_bootstrap": result["metrics_bootstrap"],
            "metrics_bootstrap_by_policy": result.get("metrics_bootstrap_by_policy", {}),
            "seed_gate_passed": result["seed_gate_passed"],
        })

    # Group per seed by target policy.  In per-VLA/loo/zero-shot modes each
    # report covers one policy.  In shared* modes a single report covers every
    # policy in ``metrics_by_policy``; the per-policy gate must read that
    # policy's own fold-correct metrics (never the pooled aggregate).
    by_policy_seeds: dict[str, list[dict[str, Any]]] = {}
    for report in seed_reports:
        for policy in sorted(report["metrics_by_policy"]):
            by_policy_seeds.setdefault(policy, []).append({
                "seed": report["seed"],
                "path": report["path"],
                "metrics": report["metrics_by_policy"][policy],
                "metrics_by_suite": report["metrics_by_policy_suite"].get(policy, {}),
                "metrics_bootstrap": (report.get("metrics_bootstrap_by_policy", {}).get(policy)
                                      or report["metrics_bootstrap"]),
                "seed_gate_passed": report["seed_gate_passed"],
            })
    unique_seeds = sorted({int(report["seed"]) for report in seed_reports})
    if len(unique_seeds) != args.expected_seed_count:
        raise ValueError(
            f"expected exactly {args.expected_seed_count} training seeds, got {unique_seeds}")

    policy_results: dict[str, Any] = {}
    for policy, member_reports in sorted(by_policy_seeds.items()):
        policy_seeds = sorted({int(report["seed"]) for report in member_reports})
        if policy_seeds != unique_seeds or len(member_reports) != len(unique_seeds):
            raise ValueError(
                f"policy {policy} has incomplete/duplicate seed reports: "
                f"seeds={policy_seeds}, reports={len(member_reports)}")
        passing = sum(gate(report["metrics"], report["metrics_by_suite"])
                      for report in member_reports)
        keys = ["success_gap", "false_continue_rate", "absolute_paired_harm",
                "savings", "conditional_missed_rescue_rate"]
        policy_results[policy] = {
            "seed_count": len(member_reports),
            "passing_seed_count": passing,
            "required_passing_seed_count": args.required_passing_seeds,
            "policy_gate_passed": passing >= args.required_passing_seeds,
            "metric_across_seeds": {
                key: interval([report["metrics"][key] for report in member_reports])
                for key in keys
            },
            "missed_rescue_task_interval": {
                key: interval([report["metrics_bootstrap"][key]["mean"] for report in member_reports])
                for key in ["conditional_missed_rescue_rate", "absolute_paired_harm"]
            },
            "suites": sorted({suite for report in member_reports
                              for suite in report["metrics_by_suite"]}),
            "suite_metric_across_seeds": {
                suite: {key: interval([report["metrics_by_suite"].get(suite, {}).get(key, float("nan"))
                                       for report in member_reports])
                        for key in ["success_gap", "absolute_paired_harm", "savings"]}
                for suite in sorted({suite for report in member_reports
                                     for suite in report["metrics_by_suite"]})
            },
            "seed_details": [
                {"seed": report["seed"],
                 "metrics": report["metrics"],
                 "metrics_by_suite": report["metrics_by_suite"],
                 "seed_gate_passed": report["seed_gate_passed"]}
                for report in member_reports
            ],
        }

    stage_gate_passed = all(value["policy_gate_passed"] for value in policy_results.values())
    result = {
        "schema_version": "rase-r6c1-early-selector-stability/v1",
        "status": "complete",
        "scientific_scope": ("policy-conditioned early-window t={0,8,16} stratified selector; "
                             "fold-correct gate; no emergency trigger; "
                             "conditional missed-rescue reported with intervals only"),
        "mode": args.mode,
        "seed_count": len(unique_seeds),
        "training_seeds": unique_seeds,
        "required_passing_seed_count": args.required_passing_seeds,
        "policy_results": policy_results,
        "stage_gate_passed": stage_gate_passed,
        "seed_reports": seed_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "policies": sorted(policy_results),
        "stage_gate_passed": stage_gate_passed,
        "policy_results": {key: {k: value[k] for k in ("seed_count", "passing_seed_count", "policy_gate_passed", "metric_across_seeds")}
                           for key, value in policy_results.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

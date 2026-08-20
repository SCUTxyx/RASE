#!/usr/bin/env python3
"""Aggregate the frozen five-seed probabilistic handback OOF gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: list[dict[str, Any]] = [json.loads(path.read_text()) for path in args.report]
    if len(reports) != 5:
        raise ValueError(f"the frozen gate requires exactly five reports, got {len(reports)}")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != 5:
        raise ValueError("training seeds must be unique")
    split_seeds = {int(report["split_seed"]) for report in reports}
    if len(split_seeds) != 1:
        raise ValueError(f"task folds changed across seeds: {sorted(split_seeds)}")
    sources = {str(report["source"]) for report in reports}
    if len(sources) != 1:
        raise ValueError("seed reports do not use the same dataset")
    source_hashes = {str(report["source_sha256"]) for report in reports}
    if len(source_hashes) != 1:
        raise ValueError("dataset bytes changed across seeds")
    split_code = {
        (str(report["trainer_source_sha256"]), str(report["model_source_sha256"]))
        for report in reports
    }
    if len(split_code) != 1:
        raise ValueError("trainer/model code changed across seeds")
    identities = {
        (int(report["n_rows"]), int(report["n_states"]), int(report["n_tasks"]))
        for report in reports
    }
    if len(identities) != 1:
        raise ValueError("dataset cardinality changed across seeds")

    metrics = [report["oof_state_metrics"] for report in reports]
    keys = (
        "empirical_success_gap",
        "conditional_expected_false_handback",
        "oft_savings",
        "handback_rate",
    )
    per_seed = []
    for report in reports:
        per_seed.append({
            "seed": int(report["seed"]),
            "all_gates_passed": bool(report["all_controller_gates_passed"]),
            "gates": report["gates"],
            "metrics": report["oof_state_metrics"],
            "task_cluster_bootstrap_95": report.get("task_cluster_bootstrap_95"),
            "repeat_level_auc_diagnostic": report["repeat_level_auc_diagnostic"],
            "soft_brier": report["soft_brier"],
        })
    pass_count = sum(item["all_gates_passed"] for item in per_seed)

    decisions: dict[str, list[bool]] = defaultdict(list)
    boundaries: dict[str, list[int | None]] = defaultdict(list)
    for report in reports:
        for row in sorted(report["state_records"], key=lambda value: value["state_key"]):
            state = str(row["state_key"])
            decisions[state].append(bool(row["handback"]))
            boundaries[state].append(row["elapsed_oft_steps"])
    unstable_states = sorted(
        state for state, values in decisions.items() if len(set(values)) > 1
    )
    handback_all_seeds = sorted(
        state for state, values in decisions.items() if all(values)
    )
    no_handback_all_seeds = sorted(
        state for state, values in decisions.items() if not any(values)
    )

    persistent_degenerate = [
        bool(report["target_audit"]["persistent_target_degenerate"])
        for report in reports
    ]
    source_supported = [
        int(report["target_audit"]["source_success_trials"]) > 0
        and int(report["target_audit"]["source_failure_trials"]) > 0
        for report in reports
    ]
    allow_second_vla = pass_count >= 4
    result = {
        "schema_version": "rase-pre-c0-r5-probabilistic-oof-seed-gate/v1",
        "reports": [str(path.resolve()) for path in args.report],
        "report_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.report],
        "seeds": seeds,
        "split_seed": next(iter(split_seeds)),
        "dataset_identity": list(next(iter(identities))),
        "per_seed": per_seed,
        "metric_ranges": {
            key: describe([float(metric[key]) for metric in metrics]) for key in keys
        },
        "gates_passed": pass_count,
        "required_passes": 4,
        "second_vla_gate_status": "ready" if allow_second_vla else "not_ready",
        "allow_second_vla": allow_second_vla,
        "world_model_feature_search_allowed": allow_second_vla,
        "decision_stability": {
            "unstable_states": unstable_states,
            "unstable_state_fraction": len(unstable_states) / max(1, len(decisions)),
            "handback_all_seeds": handback_all_seeds,
            "no_handback_all_seeds": no_handback_all_seeds,
            "selected_boundaries_by_seed": boundaries,
        },
        "head_support": {
            "persistent_target_degenerate_all_seeds": all(persistent_degenerate),
            "source_risk_has_both_trial_outcomes_all_seeds": all(source_supported),
            "claim_rule": "unsupported heads remain architectural placeholders and cannot be claimed",
        },
        "interpretation": (
            "Proceed to the preregistered second-VLA matrix."
            if allow_second_vla else
            "Stop before second VLA and world-model features; expand independent development/calibration support."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if allow_second_vla else 2


if __name__ == "__main__":
    raise SystemExit(main())

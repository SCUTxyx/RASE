#!/usr/bin/env python3
"""Aggregate the five pre-registered R7-A source-risk OOF seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.r7_source_protocol import TRAIN_SEEDS

SEEDS = TRAIN_SEEDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    errors = []
    for seed in SEEDS:
        path = args.input_root / f"seed_{seed}" / "report.json"
        if not path.is_file():
            errors.append({"seed": seed, "reason": "missing_report", "path": str(path)})
            continue
        row = json.loads(path.read_text())
        if int(row.get("seed", -1)) != seed:
            errors.append({"seed": seed, "reason": "seed_mismatch"})
            continue
        reports.append(row)
    if errors:
        raise SystemExit(json.dumps({"status": "INCOMPLETE", "errors": errors}, indent=2))
    dataset_hashes = {row.get("dataset_sha256") for row in reports}
    if len(dataset_hashes) != 1:
        raise ValueError("five seeds do not bind the same frozen dataset")
    policy_ids = {row.get("policy_id") for row in reports}
    if len(policy_ids) != 1:
        raise ValueError("five seeds do not declare the same policy_id")
    passed = sum(row.get("status") == "PASS" for row in reports)
    near_signal = sum(
        float(row["metrics"]["auroc"]) >= 0.65
        and float(row["metrics"]["ap_above_prevalence"]) >= 0.05
        for row in reports
    )
    metric_names = ("auroc", "average_precision", "ap_above_prevalence",
                    "ece_10_equal_width", "brier")
    aggregate = {
        name: {
            "mean": float(np.mean([row["metrics"][name] for row in reports])),
            "min": float(np.min([row["metrics"][name] for row in reports])),
            "max": float(np.max([row["metrics"][name] for row in reports])),
        } for name in metric_names
    }
    full_pass = passed >= 4
    one_native_attempt = (
        not full_pass and near_signal >= 3 and aggregate["auroc"]["mean"] >= 0.65
    )
    decision = (
        "FULL_PASS" if full_pass else
        "ONE_POLICY_NATIVE_ATTEMPT" if one_native_attempt else
        "STOP_SOURCE_RISK_ESCALATION"
    )
    result = {
        "schema_version": "rase-r7a-source-risk-stability/v1",
        "status": "PASS" if full_pass else "FAIL",
        "policy_id": next(iter(policy_ids)),
        "decision": decision,
        "scientific_scope": "development five-seed task-held-out source-risk gate",
        "pre_registered_seeds": list(SEEDS),
        "dataset_sha256": next(iter(dataset_hashes)),
        "seeds_passed": passed, "seeds_required": 4,
        "near_signal_seeds": near_signal,
        "aggregate_metrics": aggregate,
        "per_seed": [{
            "seed": row["seed"], "status": row["status"],
            "metrics": row["metrics"], "gate": row["gate"],
            "bootstrap_auroc": row["task_bootstrap"]["auroc"],
        } for row in reports],
        "unlocks_on_full_pass": [
            "policy-native additive adapter probe",
            "new-cohort t0 persistent-OFT opportunity audit",
        ],
        "unlocks_on_one_policy_native_attempt": [
            "one pre-registered frozen-policy-native additive adapter probe only",
        ],
        "remains_locked": [
            "world-model features", "multi-VLA shared selector",
            "independent validation", "test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

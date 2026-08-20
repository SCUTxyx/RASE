#!/usr/bin/env python3
"""Aggregate the five pre-registered R8-B canonical OOF seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seeds = [int(value) for value in protocol["seeds"]]
    reports, errors = [], []
    for seed in seeds:
        path = args.input_root / f"seed_{seed}.json"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        report = json.loads(path.read_text())
        if (int(report.get("seed", -1)) != seed
                or report.get("policy_conditioning") != "id"
                or report.get("policy_filter") is not None
                or report.get("protocol_sha256") != sha256(args.protocol)):
            errors.append(f"contract mismatch {path}")
            continue
        reports.append(report)
    pass_count = sum(report["status"] == "PASS" for report in reports)
    complete = len(reports) == len(seeds) and not errors
    status = "PASS" if complete and pass_count >= 4 else "FAIL"
    decision = ("UNLOCK_R8B_COMPARATORS" if status == "PASS"
                else "STOP_LOCAL_HAZARD_ESCALATION")
    metric_names = [
        "auroc", "average_precision", "ap_above_prevalence",
        "ece_10_equal_width", "brier",
    ]
    summary = {
        name: {
            "mean": float(np.mean([row["hazard_metrics"][name] for row in reports])),
            "minimum": float(np.min([row["hazard_metrics"][name] for row in reports])),
            "maximum": float(np.max([row["hazard_metrics"][name] for row in reports])),
        }
        for name in metric_names
    } if reports else {}
    result = {
        "schema_version": "rase-r8b-local-hazard-stability/v1",
        "status": status, "decision": decision,
        "scientific_scope": "five-seed canonical no-world-model development stability gate",
        "protocol": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol),
        "expected_seeds": seeds, "completed_seeds": [row["seed"] for row in reports],
        "passing_seeds": [row["seed"] for row in reports if row["status"] == "PASS"],
        "pass_count": pass_count, "required_pass_count": 4,
        "metrics_across_seeds": summary,
        "seed_reports": [{
            "seed": row["seed"], "status": row["status"],
            "hazard_metrics": row["hazard_metrics"],
            "current_recoverable_metrics": row["current_recoverable_metrics"],
            "hazard_task_bootstrap_auroc": row["hazard_task_bootstrap_auroc"],
            "gate": row["gate"],
        } for row in reports],
        "errors": errors,
        "remains_locked": (["world-model", "validation", "test", "closed-loop"]
                            if status == "PASS" else
                            ["comparators", "selector", "world-model", "validation", "test"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "pass_count", "passing_seeds",
        "metrics_across_seeds", "errors",
    )}, indent=2, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze an outcome-balanced, task-distinct R6-B1 cross-suite pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = {
        "pi0fast_libero": [0],
        "pi05_libero": [0, 1],
    }
    records = []
    used_tasks: set[str] = set()
    for policy, seeds in policies.items():
        reports = [json.loads((args.atlas_root / policy / f"seed_{seed}" / "summary.json").read_text())
                   for seed in (0, 1)]
        indexed = [{str(row["state_key"]): row for row in report["per_state"]} for report in reports]
        suites = sorted({str(row["suite"]) for row in indexed[0].values()})
        for suite in suites:
            candidates = []
            for key, row0 in indexed[0].items():
                if str(row0["suite"]) != suite:
                    continue
                row1 = indexed[1][key]
                outcomes = (bool(row0["source_success"]), bool(row1["source_success"]))
                candidates.append((key, row0, row1, outcomes))
            for label in ("stable_success", "failure_support"):
                if label == "stable_success":
                    eligible = [item for item in candidates if item[3] == (True, True)]
                else:
                    eligible = [item for item in candidates if not all(item[3])]
                if not eligible:
                    raise ValueError(f"{policy}/{suite} lacks {label}")
                distinct = [item for item in eligible if str(item[1]["task_id"]) not in used_tasks]
                if not distinct:
                    raise ValueError(f"{policy}/{suite}/{label} lacks a task-distinct candidate")
                key, row0, row1, outcomes = min(distinct, key=lambda item: item[0])
                used_tasks.add(str(row0["task_id"]))
                records.append({
                    "policy_id": policy,
                    "suite": suite,
                    "selection_label": label,
                    "state_key": key,
                    "task_id": str(row0["task_id"]),
                    "seed_indices": seeds,
                    "r6a_seed0": {"success": outcomes[0], "env_steps": int(row0["result"]["env_steps"]),
                                  "rollout_seed": int(row0["rollout_seed"])},
                    "r6a_seed1": {"success": outcomes[1], "env_steps": int(row1["result"]["env_steps"]),
                                  "rollout_seed": int(row1["rollout_seed"])},
                })
    tasks = [record["task_id"] for record in records]
    if len(tasks) != len(set(tasks)):
        raise ValueError("pilot selections must be task-distinct")
    payload = {
        "schema_version": "rase-r6b1-pilot-manifest/v1",
        "status": "frozen",
        "scientific_scope": "development-only cross-suite dynamic collector pilot",
        "atlas_root": str(args.atlas_root.resolve()),
        "atlas_root_note": "source outcomes/seeds are frozen R6-A references",
        "records": records,
        "n_records": len(records),
        "n_true_tasks": len(set(tasks)),
        "boundaries": [0, 16, 32],
        "required_labels": ["stable_success", "failure_support"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["manifest_sha256"] = sha256(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

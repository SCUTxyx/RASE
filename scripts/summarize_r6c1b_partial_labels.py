#!/usr/bin/env python3
"""Summarize in-flight R6-C.1B labels without making a stage decision.

Only canonical rep0 files are counted.  The report is explicitly provisional:
rep1/rep2 reproducibility adjudication and exclusions have not yet been
applied, so it must never be consumed by a training or gate script.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def role_of(path: Path) -> str:
    return "train_enrichment" if "train_enrichment" in path.parts else "natural_development_eval"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    expected_total = int(plan["expected_trajectory_groups"])
    records = []
    seen = set()
    paths = sorted(glob.glob(str(args.input_root / "suite_*" / "*" / "**" /
                                  "rep0" / "*__seed*.json"), recursive=True))
    for value in paths:
        path = Path(value)
        data = json.loads(path.read_text())
        if not data.get("rows"):
            continue
        first = data["rows"][0]
        key = (str(first["policy_id"]), int(first["seed_index"]),
               str(first["state_key"]), role_of(path))
        if key in seen:
            raise ValueError(f"duplicate rep0 trajectory: {key}")
        seen.add(key)
        source_success = bool(data.get("source_success", first["source_final_success"]))
        boundaries = [row for row in data["rows"]
                      if int(row["elapsed_source_steps"]) in (0, 8, 16)]
        records.append({
            "policy_id": key[0],
            "role": key[3],
            "suite": str(first["suite"]),
            "task_id": str(first["task_id"]),
            "source_success": source_success,
            "early_rescuable": ((not source_success) and any(
                bool(row["persistent_success_if_enter_now"]) for row in boundaries)),
            "oft_harms_source_success": (source_success and any(
                not bool(row["persistent_success_if_enter_now"]) for row in boundaries)),
            "boundary_outcomes": {
                str(int(row["elapsed_source_steps"])): bool(
                    row["persistent_success_if_enter_now"])
                for row in boundaries
            },
        })

    by_policy = defaultdict(list)
    for row in records:
        by_policy[row["policy_id"]].append(row)
    policy_results = {}
    for policy, rows in sorted(by_policy.items()):
        role_results = {}
        for role in sorted({row["role"] for row in rows}):
            values = [row for row in rows if row["role"] == role]
            by_boundary = {}
            for elapsed in (0, 8, 16):
                key = str(elapsed)
                present = [row for row in values if key in row["boundary_outcomes"]]
                source_failures = [row for row in present if not row["source_success"]]
                source_successes = [row for row in present if row["source_success"]]
                by_boundary[key] = {
                    "groups": len(present),
                    "source_failure_groups": len(source_failures),
                    "failure_groups_rescued": sum(
                        row["boundary_outcomes"][key] for row in source_failures),
                    "source_success_groups": len(source_successes),
                    "source_success_groups_harmed": sum(
                        not row["boundary_outcomes"][key] for row in source_successes),
                }
            role_results[role] = {
                "rep0_groups": len(values),
                "tasks": len({row["task_id"] for row in values}),
                "source_failures": sum(not row["source_success"] for row in values),
                "early_rescuable": sum(row["early_rescuable"] for row in values),
                "source_success_groups_with_any_oft_harm": sum(
                    row["oft_harms_source_success"] for row in values),
                "by_boundary": by_boundary,
                "suites": sorted({row["suite"] for row in values}),
            }
        policy_results[policy] = {"roles": role_results}

    # The collection plan counts two replicas; partial monitoring counts rep0
    # only, so total files on disk are also reported separately.
    all_metadata = glob.glob(str(args.input_root / "suite_*" / "*" / "**" /
                                  "*__seed*.json"), recursive=True)
    result = {
        "schema_version": "rase-r6c1b-partial-label-summary/v1",
        "status": "provisional_do_not_gate",
        "input_root": str(args.input_root.resolve()),
        "expected_all_replica_groups": expected_total,
        "metadata_files_complete": len(all_metadata),
        "collection_fraction": len(all_metadata) / max(1, expected_total),
        "canonical_rep0_groups": len(records),
        "policy_results": policy_results,
        "warning": (
            "No reproducibility exclusions have been applied. Do not train, "
            "select thresholds, or make a stage decision from this report."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

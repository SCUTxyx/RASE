#!/usr/bin/env python3
"""Freeze a metadata-only four-suite B2 synchronous-capture smoke manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(salt: str, key: tuple[str, str, str, int]) -> str:
    return hashlib.sha256((salt + "\x1f" + "\x1f".join(map(str, key))).encode()).hexdigest()


def group_key(job: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(job["root_id"]), str(job["policy_id"]),
        str(job["decision_point"]["decision_point_id"]),
        int(job["seed_ledger"]["exact_repeat_replica"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", default="pi0fast.libero")
    parser.add_argument("--salt", default="rase-vnext-b2-capture-smoke-v1")
    args = parser.parse_args()
    parent_path = args.parent.resolve()
    parent = json.loads(parent_path.read_text())
    if parent.get("status") != "frozen_confirmation":
        raise SystemExit("parent must be a frozen confirmation manifest")
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in parent["jobs"]:
        if str(job["policy_id"]) != args.policy_id:
            continue
        if int(job["seed_ledger"]["exact_repeat_replica"]) != 0:
            continue
        grouped[group_key(job)].append(job)
    by_suite: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    for key, jobs in grouped.items():
        if len(jobs) != 5:
            raise SystemExit(f"group {key} does not contain five operators")
        by_suite[str(jobs[0]["suite"])].append(key)
    if set(by_suite) != {"Spatial", "Object", "Goal", "Long"}:
        raise SystemExit(f"unexpected suite coverage: {sorted(by_suite)}")
    selected = {
        min(keys, key=lambda key: (rank(args.salt, key), key))
        for suite, keys in sorted(by_suite.items())
    }
    jobs = [job for key in sorted(selected) for job in grouped[key]]
    root_ids = {str(key[0]) for key in selected}
    tasks = sorted({str(job["task_id"]) for job in jobs})
    manifest = {
        **parent,
        "schema_version": "rase-vnext-b2-capture-smoke-manifest/v1",
        "status": "frozen_confirmation",
        "scientific_scope": "B2_PARITY_ONLY_NOT_AN_EFFECT_COHORT",
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": sha256(parent_path),
        "selection_salt": args.salt,
        "selection_rule": (
            "one replica0 group per suite selected by sha256 over frozen metadata; "
            "outcomes not read"
        ),
        "fixed_repeats": 1,
        "roots_per_task": None,
        "roots": [row for row in parent["roots"] if str(row["root_id"]) in root_ids],
        "task_folds": {task: parent["task_folds"][task] for task in tasks},
        "tasks": tasks,
        "suites": sorted(by_suite),
        "expected_roots": len(root_ids),
        "expected_jobs": len(jobs),
        "expected_available_jobs": sum(job["available_by_contract"] for job in jobs),
        "jobs": jobs,
        "forbidden_adaptations": sorted(set(parent.get("forbidden_adaptations", [])) | {
            "effect_claim_from_b2_smoke", "outcome_dependent_selection",
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "groups": len(selected), "jobs": len(jobs), "tasks": tasks,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

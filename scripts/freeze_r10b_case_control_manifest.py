#!/usr/bin/env python3
"""Freeze a balanced R10-B t8->t16 recoverability-loss case-control pilot.

The case/control label is read only from the already-frozen K=2 replica
aggregate.  A case is certainly recoverable at t=8 and certainly
unrecoverable at t=16.  A control is certainly recoverable at both boundaries.
Ambiguous labels are ineligible.  This is a representation-development cohort,
not a prevalence, calibration, validation, or test cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SALT = "rase-r10b-case-control-t8-t16/v1/20260813"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value: str) -> str:
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def seed_from_group(group_id: str) -> int:
    match = re.search(r":seed(\d+)$", group_id)
    if not match:
        raise ValueError(f"group id has no seed index: {group_id}")
    return int(match.group(1))


def choose(candidates: list[dict], count: int) -> list[dict]:
    """Choose deterministically with both policies and task diversity."""
    selected: list[dict] = []
    policies = sorted({row["policy_id"] for row in candidates})
    for policy in policies:
        rows = sorted((row for row in candidates if row["policy_id"] == policy),
                      key=lambda row: rank(row["group_id"]))
        selected.extend(rows[:min(2, len(rows))])
    # The target is at least six in every suite/class for the frozen source.
    if len(selected) > count:
        selected = sorted(selected, key=lambda row: rank(row["group_id"]))[:count]
    selected_ids = {row["group_id"] for row in selected}
    while len(selected) < count:
        remaining = [row for row in candidates if row["group_id"] not in selected_ids]
        if not remaining:
            raise ValueError(f"not enough candidates: requested {count}")
        used_tasks = {row["task_id"] for row in selected}
        remaining.sort(key=lambda row: (
            row["task_id"] in used_tasks, rank(row["group_id"])))
        row = remaining[0]
        selected.append(row)
        selected_ids.add(row["group_id"])
    return selected


def assign_folds(records: list[dict], count: int = 5) -> dict[str, int]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task, key=lambda task: (
        -len(by_task[task]),
        -len({row["hazard_label_k2"] for row in by_task[task]}),
        rank(task),
    ))
    fold_rows: list[list[dict]] = [[] for _ in range(count)]
    assignment = {}
    for task in tasks:
        rows = by_task[task]
        pos = sum(row["hazard_label_k2"] for row in rows)
        neg = len(rows) - pos
        suite = rows[0]["suite"]
        scored = []
        for fold in range(count):
            current_pos = sum(row["hazard_label_k2"] for row in fold_rows[fold])
            current_neg = len(fold_rows[fold]) - current_pos
            current_suite = sum(row["suite"] == suite for row in fold_rows[fold])
            scored.append(((len(fold_rows[fold]), current_suite,
                            abs((current_pos + pos) - (current_neg + neg)),
                            rank(f"{task}:{fold}")), fold))
        fold = min(scored)[1]
        assignment[task] = fold
        fold_rows[fold].extend(rows)
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--r8-audit", type=Path, required=True)
    parser.add_argument("--parent-initial", type=Path, required=True)
    parser.add_argument("--enrichment-initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-per-class-suite", type=int, default=12)
    args = parser.parse_args()

    r8 = json.loads(args.r8_audit.read_text())
    if r8.get("status") != "PASS" or r8.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("R8-A audit is not PASS or is bound to another dataset")
    initial_sources = [json.loads(args.parent_initial.read_text()),
                       json.loads(args.enrichment_initial.read_text())]
    pools = {source["pool"] for source in initial_sources}
    if len(pools) != 1:
        raise ValueError(f"initial manifests use different pools: {pools}")
    initial_records = {}
    for source in initial_sources:
        for row in source["records"]:
            initial_records[row["state_key"]] = row

    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    groups: dict[str, dict[int, int]] = defaultdict(dict)
    for index, group_id in enumerate(data["group_id"]):
        groups[str(group_id)][int(data["elapsed_source_steps"][index])] = index
    candidates = []
    for group_id, by_elapsed in groups.items():
        if 8 not in by_elapsed or 16 not in by_elapsed:
            continue
        start, end = by_elapsed[8], by_elapsed[16]
        start_successes = float(data["persistent_successes"][start])
        start_trials = float(data["persistent_trials"][start])
        end_successes = float(data["persistent_successes"][end])
        end_trials = float(data["persistent_trials"][end])
        start_safe = start_trials >= 2 and start_successes == start_trials
        end_safe = end_trials >= 2 and end_successes == end_trials
        end_unsafe = end_trials >= 2 and end_successes == 0
        if not start_safe or not (end_safe or end_unsafe):
            continue
        state_key = str(data["state_key"][start])
        if state_key not in initial_records:
            raise ValueError(f"state missing from frozen initial manifests: {state_key}")
        candidates.append({
            "group_id": group_id,
            "state_key": state_key,
            "task_id": str(data["task_id"][start]),
            "suite": str(data["suite"][start]),
            "policy_id": str(data["policy_id"][start]),
            "seed_index": seed_from_group(group_id),
            "cohort_role_parent": str(data["cohort_role"][start]),
            "hazard_label_k2": int(end_unsafe),
            "t8_persistent_successes_k2": int(start_successes),
            "t8_persistent_trials_k2": int(start_trials),
            "t16_persistent_successes_k2": int(end_successes),
            "t16_persistent_trials_k2": int(end_trials),
            "selection_rank": rank(group_id),
        })

    selected = []
    suite_targets = {}
    for suite in ("Spatial", "Object", "Goal", "Long"):
        by_label = {label: [row for row in candidates
                            if row["suite"] == suite and row["hazard_label_k2"] == label]
                    for label in (0, 1)}
        target = min(args.maximum_per_class_suite, len(by_label[0]), len(by_label[1]))
        if target < 6:
            raise ValueError(f"insufficient balanced support for {suite}: "
                             f"negative={len(by_label[0])} positive={len(by_label[1])}")
        suite_targets[suite] = target
        selected.extend(choose(by_label[0], target))
        selected.extend(choose(by_label[1], target))

    selected.sort(key=lambda row: (row["suite"], row["task_id"], row["policy_id"],
                                   row["seed_index"], row["state_key"]))
    folds = assign_folds(selected)
    for row in selected:
        row["outer_fold"] = folds[row["task_id"]]
        row["initial_record"] = initial_records[row["state_key"]]

    fold_support = {}
    for fold in range(5):
        rows = [row for row in selected if row["outer_fold"] == fold]
        fold_support[str(fold)] = {
            "groups": len(rows), "tasks": len({row["task_id"] for row in rows}),
            "positives": sum(row["hazard_label_k2"] for row in rows),
            "negatives": sum(1 - row["hazard_label_k2"] for row in rows),
            "suites": dict(sorted(Counter(row["suite"] for row in rows).items())),
        }
    if any(row["positives"] < 1 or row["negatives"] < 1
           for row in fold_support.values()):
        raise ValueError(f"a frozen outer fold lacks both classes: {fold_support}")

    result = {
        "schema_version": "rase-r10b-case-control-manifest/v1",
        "status": "frozen",
        "scientific_scope": "development-only label-balanced representation pilot",
        "selection_uses_frozen_labels": True,
        "not_valid_for": ["prevalence", "calibration", "validation", "test",
                          "closed-loop selector claims"],
        "label_definition": "t8 K2 persistent all-success -> t16 K2 all-failure",
        "control_definition": "t8 K2 persistent all-success -> t16 K2 all-success",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "r8_audit": str(args.r8_audit.resolve()), "r8_audit_sha256": sha256(args.r8_audit),
        "parent_initial_sha256": sha256(args.parent_initial),
        "enrichment_initial_sha256": sha256(args.enrichment_initial),
        "pool": next(iter(pools)), "salt": SALT,
        "boundaries": [0, 4, 8, 12, 16], "temporal_history": 8,
        "replicas": 3, "records": selected, "expected_groups": len(selected),
        "expected_trajectories": len(selected) * 3,
        "suite_targets_per_class": suite_targets,
        "tasks": len({row["task_id"] for row in selected}),
        "policies": dict(sorted(Counter(row["policy_id"] for row in selected).items())),
        "labels": dict(sorted(Counter(row["hazard_label_k2"] for row in selected).items())),
        "fold_support": fold_support,
        "gate_after_collection": {
            "all_k3_labels_stable": True,
            "all_k3_labels_match_frozen_k2": True,
            "all_t8_causal_features_have_replica_parity": True,
            "each_outer_fold_has_both_classes": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "expected_groups", "expected_trajectories", "tasks", "policies",
        "labels", "suite_targets_per_class", "fold_support")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

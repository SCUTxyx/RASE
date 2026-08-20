#!/usr/bin/env python3
"""Freeze an outcome-independent R9-B temporal development manifest.

The manifest chooses six tasks per suite and one unused pool snapshot per task.
Selection is based only on frozen pool metadata and a hash salt; no source or
fallback outcome is read.  Existing R6/R7/R8 dataset states are excluded by
their state keys, so this pilot is development-only and cannot be called test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SALT = "rase-r9b-temporal-pilot/v1/20260813"
SUITES = ("Spatial", "Object", "Goal", "Long")
PERTURB_ORDER = {"clean": 0, "camera": 1, "robot": 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value: str) -> str:
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--exclude-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-per-suite", type=int, default=6)
    args = parser.parse_args()
    if args.tasks_per_suite < 1:
        raise ValueError("tasks-per-suite must be positive")
    import numpy as np

    with np.load(args.exclude_dataset, allow_pickle=False) as loaded:
        excluded_states = set(str(value) for value in loaded["state_key"])
    candidates: list[dict] = []
    for meta in sorted(args.pool.rglob("meta.json")):
        item = json.loads(meta.read_text())
        key = str(item["state_key"])
        if key in excluded_states:
            continue
        item = dict(item)
        item["state_key"] = key
        item["_rank"] = rank(key)
        item["_meta_path"] = str(meta.resolve())
        candidates.append(item)
    by_suite_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in candidates:
        suite = str(item["suite"])
        if suite in SUITES:
            by_suite_task[(suite, str(item["task_id"]))].append(item)
    records = []
    task_counts = {}
    for suite in SUITES:
        tasks = sorted({task for current_suite, task in by_suite_task if current_suite == suite})
        if len(tasks) < args.tasks_per_suite:
            raise ValueError(f"{suite}: only {len(tasks)} unused tasks")
        # Hash-ranked task selection avoids choosing by observed outcome.  A
        # perturbation-balanced ordering is applied only within each task.
        task_order = sorted(tasks, key=lambda task: rank(f"task:{suite}:{task}"))
        selected_tasks = task_order[:args.tasks_per_suite]
        task_counts[suite] = selected_tasks
        for task in selected_tasks:
            members = sorted(
                by_suite_task[(suite, task)],
                key=lambda item: (
                    PERTURB_ORDER.get(str(item.get("perturb_dim", "clean")), 9),
                    int(item.get("level", 0)), int(item.get("step", 0)), item["_rank"],
                ),
            )
            # Deterministically rotate clean/camera/robot targets.  If a task
            # lacks the requested perturbation, use the nearest available
            # metadata category; this remains outcome-independent.
            target = ("clean", "camera", "robot")[
                int(rank(f"perturb:{suite}:{task}")[:8], 16) % 3
            ]
            exact = [item for item in members if str(item.get("perturb_dim")) == target]
            chosen = sorted(exact or members, key=lambda item: item["_rank"])[0]
            records.append({
                "role": "r9b_temporal_development",
                "suite": suite, "task_id": task, "state_key": chosen["state_key"],
                "perturb_dim": str(chosen.get("perturb_dim", "clean")),
                "perturb_level": int(chosen.get("level", 0)),
                "step": int(chosen.get("step", 0)),
                "episode_id": str(chosen.get("episode_id", "")),
                "pool_seed": int(chosen.get("seed", 0)),
                "selection_rank": chosen["_rank"],
                "metadata_path": chosen["_meta_path"],
            })
    records.sort(key=lambda item: (SUITES.index(item["suite"]), item["task_id"]))
    state_keys = [item["state_key"] for item in records]
    payload = {
        "schema_version": "rase-r9b-temporal-manifest/v1",
        "status": "frozen",
        "scientific_scope": "development temporal observability pilot; not validation/test",
        "selection_salt": SALT, "selection_uses_outcomes": False,
        "pool": str(args.pool.resolve()), "pool_sha256": sha256(args.pool / "pool.json")
                         if (args.pool / "pool.json").is_file() else None,
        "exclude_dataset": str(args.exclude_dataset.resolve()),
        "exclude_dataset_sha256": sha256(args.exclude_dataset),
        "tasks_per_suite": args.tasks_per_suite, "expected_records": len(records),
        "task_counts": task_counts, "state_keys": state_keys,
        "state_keys_sha256": hashlib.sha256(json.dumps(state_keys).encode()).hexdigest(),
        "boundaries": [0, 4, 8, 12, 16], "temporal_history": 4,
        "replica_plan": {
            "replicas": [0, 1, 2], "same_seed_and_checkpoint": True,
            "t0_feature_parity_required": True,
            "full_late_action_trace_parity_required": False,
        },
        "source_policies": ["pi05_libero", "pi0fast_libero"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "records": len(records),
                      "task_counts": task_counts, "state_keys_sha256": payload["state_keys_sha256"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

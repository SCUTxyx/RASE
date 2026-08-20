#!/usr/bin/env python3
"""Freeze the immutable K5 selector dataset manifest (v2 plan §4).

Reads runs/rase_vnext/k5_collect_v1 (branches + captures), recomputes the
pre-registered feature groups with the alignment-audited loader, and writes:

  - selector_dataset_manifest.json : schema, hashes, fold/calibration splits,
    per-row metadata (task/root/operator/replica/fold/calib_split/success), and
    feature matrix hashes;
  - features.npz : state-only / raw-action / trace-only / semantic matrices.

No outcome is read to choose splits: fold comes from the frozen cohort
manifest's task_folds; within-fold calibration split (9 train + 3 calib) is
task-level, hash-ranked, frozen here for the first time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_rase_vnext_k3 import (  # noqa: E402
    atomic_json,
    load_collection,
    load_features_and_targets,
    sha256,
)


def stable_seed(*parts: object) -> int:
    token = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calib-per-fold", type=int, default=3)
    parser.add_argument("--salt", default="rase-vnext-selector-dataset-v1")
    args = parser.parse_args()

    cohort = json.loads(args.manifest.read_text())
    if cohort.get("status") != "frozen_confirmation":
        raise SystemExit("cohort manifest is not frozen")
    rows, bound = load_collection(args.output_dir)
    dataset = load_features_and_targets(rows, args.output_dir, bound)
    n = len(dataset["records"])
    if n == 0:
        raise SystemExit("no rows")

    # Fold (from frozen manifest) and within-fold calibration split (task-level,
    # hash-ranked; frozen here, never from outcome).
    fold_by_task = {task: int(fold) for task, fold in cohort["task_folds"].items()}
    tasks = sorted(set(dataset["tasks"]))
    tasks_by_fold: dict[int, list[str]] = {}
    for task in tasks:
        tasks_by_fold.setdefault(fold_by_task[task], []).append(task)
    calib_tasks: set[str] = set()
    for fold, fold_tasks in tasks_by_fold.items():
        if args.calib_per_fold <= 0:
            continue
        ordered = sorted(
            fold_tasks,
            key=lambda task: (stable_seed(args.salt, "calib", fold, task), task),
        )
        chosen = ordered[: args.calib_per_fold]
        if len(chosen) != args.calib_per_fold:
            raise SystemExit(f"fold {fold}: need {args.calib_per_fold} calib tasks, got {len(chosen)}")
        calib_tasks.update(chosen)

    rows_meta: list[dict[str, Any]] = []
    for record, label in zip(dataset["records"], dataset["labels"]):
        task = str(record["task_id"])
        rows_meta.append({
            "root_id": str(record["root_id"]),
            "task_id": task,
            "suite": str(record["suite"]),
            "operator": str(record["operator"]),
            "replica": int(record["replica"]),
            "fold": int(fold_by_task[task]),
            "calibration_split": bool(task in calib_tasks),
            "success": bool(label["success"]),
            "utility": float(label["utility"]),
            "progress": float(label["progress"]),
        })

    features_path = args.output.parent / f"{args.output.stem}_features.npz"
    temporary = features_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            state=dataset["state"],
            raw=dataset["raw"],
            trace=dataset["trace"],
            semantic=dataset["semantic"],
        )
    temporary.replace(features_path)

    manifest = {
        "schema_version": "rase-vnext-selector-dataset/v1",
        "status": "frozen",
        "cohort_manifest": str(args.manifest.resolve()),
        "cohort_manifest_sha256": sha256(args.manifest),
        "collection_output": str(args.output_dir.resolve()),
        "collection_manifest_sha256": sha256(args.output_dir / "manifest.bound.json"),
        "branches_sha256": sha256(args.output_dir / "branches.jsonl"),
        "feature_schema": {
            "state_only": "proprio[:16] padded",
            "raw_action": "raw_action_feature_vector(mean/std/min/max/last/len)",
            "trace_only": "trace_feature_vector(velocity/acc/jerk/path/reversal/gripper)",
            "semantic": "concat(raw, trace)",
        },
        "feature_version": "rase-vnext-selector-features/v1",
        "features_path": str(features_path.resolve()),
        "features_sha256": sha256(features_path),
        "n_rows": n,
        "n_roots": len(set(record["root_id"] for record in dataset["records"])),
        "n_tasks": len(tasks),
        "n_operators": len(set(record["operator"] for record in dataset["records"])),
        "splits": {
            "folds": {str(fold): len(fold_tasks) for fold, fold_tasks in sorted(tasks_by_fold.items())},
            "calib_tasks_per_fold": args.calib_per_fold,
            "calibration_split_rule": (
                "task-level, hash-ranked (stable_seed(salt, 'calib', fold, task)); "
                "9 train + 3 calib per fold; frozen before any selector metric"
            ),
            "calib_tasks": sorted(calib_tasks),
        },
        "labels": {"success": "binary outcome", "utility": "protocol utility (evaluation only)"},
        "selection_salt": args.salt,
        "rows": rows_meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "features_sha256": manifest["features_sha256"],
        "n_rows": n, "n_tasks": len(tasks), "n_roots": manifest["n_roots"],
        "calib_tasks": sorted(calib_tasks),
        "rows_per_fold": {str(f): sum(1 for r in rows_meta if r["fold"] == f) for f in sorted(tasks_by_fold)},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

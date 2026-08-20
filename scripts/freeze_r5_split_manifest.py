#!/usr/bin/env python3
"""Freeze the R5 train/calibration/test contract from PRE-A3 keys."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_states": len(records),
        "n_tasks": len({str(r["task_id"]) for r in records}),
        "state_keys": sorted(str(r["state_key"]) for r in records),
        "task_ids": sorted({str(r["task_id"]) for r in records}),
        "suite_state_counts": dict(sorted(Counter(str(r["suite"]) for r in records).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.state_keys.read_text())
    records = source["records"]
    by_split = {
        name: [record for record in records if record.get("split") == name]
        for name in ("train", "val", "test")
    }
    states = {name: {str(r["state_key"]) for r in rows} for name, rows in by_split.items()}
    tasks = {name: {str(r["task_id"]) for r in rows} for name, rows in by_split.items()}
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = {
            "state_overlap": len(states[left] & states[right]),
            "task_overlap": len(tasks[left] & tasks[right]),
        }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise ValueError(f"split overlap detected: {overlaps}")

    manifest = {
        "schema_version": "rase-pre-c0-r5-split-lock/v1",
        "status": "frozen",
        "source": str(args.state_keys.resolve()),
        "source_sha256": sha256(args.state_keys),
        "selection_uses_outcomes": False,
        "splits": {name: split_summary(rows) for name, rows in by_split.items()},
        "overlap_audit": overlaps,
        "roles": {
            "train": "model development and nested task-OOF only",
            "val": "threshold/conformal calibration and one pilot validation rebuild",
            "test": "untouched until all offline train+val gates pass",
        },
        "frozen_controller_gates": {
            "row_auroc_min": 0.70,
            "policy_success_gap_min": -0.05,
            "conditional_false_handback_max": 0.05,
            "oft_savings_min": 0.20,
            "task_cluster_confidence": 0.95,
        },
        "evaluation_semantics": {
            "decision_unit": "state",
            "stopping_rule": "earliest accepted boundary only",
            "missing_prediction": "hard error",
            "overlap": "hard error",
            "test_reuse": "forbidden",
        },
        "expansion_targets": {
            "development_states_min": 300,
            "persistent_rescuable_calibration_states_min": 100,
            "frozen_test_states_min": 100,
            "second_vla_required": True,
        },
        "invalidated_artifacts": [
            "runs/pre_c0_r4/heldout_m4.jsonl",
            "runs/pre_c0_r4/m4_validation.json",
            "runs/pre_c0_r4/m5_conference_eval.json",
        ],
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["manifest_content_sha256_without_self"] = hashlib.sha256(payload.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "source_sha256": manifest["source_sha256"],
        "overlap_audit": overlaps,
        "split_counts": {k: v["n_states"] for k, v in manifest["splits"].items()},
        "task_counts": {k: v["n_tasks"] for k, v in manifest["splits"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

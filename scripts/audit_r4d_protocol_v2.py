#!/usr/bin/env python3
"""Fail-closed audit for R4-D train/validation/evidence protocols."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["state_key"]), int(row.get("elapsed_oft_steps", 0))


def l2_difference(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, default=None)
    parser.add_argument("--teacher-evidence", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = read_jsonl(args.train)
    train_states = {str(row["state_key"]) for row in train}
    train_tasks = {str(row.get("task_id")) for row in train}
    train_keys = {row_key(row) for row in train}
    report: dict[str, Any] = {
        "schema_version": "rase-pre-c0-r4d-protocol-audit/v2",
        "train": {
            "rows": len(train), "states": len(train_states), "tasks": len(train_tasks),
            "duplicate_boundary_keys": len(train) - len(train_keys),
        },
    }
    gates: dict[str, bool] = {"unique_train_boundary_keys": len(train) == len(train_keys)}

    if args.heldout is not None:
        heldout = read_jsonl(args.heldout)
        heldout_states = {str(row["state_key"]) for row in heldout}
        heldout_tasks = {str(row.get("task_id")) for row in heldout}
        heldout_keys = {row_key(row) for row in heldout}
        report["heldout"] = {
            "rows": len(heldout), "states": len(heldout_states), "tasks": len(heldout_tasks),
            "state_overlap": len(train_states & heldout_states),
            "task_overlap": len(train_tasks & heldout_tasks),
            "boundary_key_overlap": len(train_keys & heldout_keys),
        }
        gates["state_disjoint_heldout"] = not bool(train_states & heldout_states)
        gates["task_disjoint_heldout"] = not bool(train_tasks & heldout_tasks)
        gates["boundary_key_disjoint_heldout"] = not bool(train_keys & heldout_keys)

    if args.teacher_evidence is not None:
        evidence = read_jsonl(args.teacher_evidence)
        evidence_keys = {
            (str(row["state_key"]), int(row["elapsed_oft_steps"]))
            for row in evidence if row.get("elapsed_oft_steps") is not None
        }
        distinct = sum(
            l2_difference(row.get("student_delta", []), row.get("oft_delta", [])) > 1e-8
            for row in evidence
        )
        report["teacher_evidence"] = {
            "rows": len(evidence),
            "exact_keys": len(evidence_keys),
            "matched_train_keys": len(train_keys & evidence_keys),
            "missing_train_keys": len(train_keys - evidence_keys),
            "distinct_student_oft_deltas": distinct,
            "distinct_student_oft_delta_fraction": distinct / max(len(evidence), 1),
        }
        gates["teacher_exact_key_coverage"] = train_keys <= evidence_keys
        # Identical predictions are legitimate when the two policies propose
        # effectively identical actions, so require broad rather than universal
        # branch sensitivity.
        gates["teacher_action_conditioned"] = distinct / max(len(evidence), 1) >= 0.5

    report["gates"] = gates
    report["all_pass"] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

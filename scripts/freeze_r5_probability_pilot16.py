#!/usr/bin/env python3
"""Freeze a fresh, suite/task-balanced 16-state probabilistic-label pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_excluded(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        str(json.loads(line)["state_key"])
        for line in path.read_text().splitlines()
        if line.strip()
    }


def finite_safe(row: dict[str, Any]) -> bool:
    return any(
        bool(success)
        for operator, success in row["operator_success"].items()
        if str(operator).startswith("OFT_H")
    )


def choose_two(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: str(row["state_key"]))
    finite = [row for row in rows if finite_safe(row)]
    persistent_only = [row for row in rows if not finite_safe(row)]
    if finite and persistent_only:
        return [finite[0], persistent_only[0]]
    if len(rows) < 2:
        raise ValueError("each task needs at least two fresh states")
    return rows[:2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exclude-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    excluded = read_excluded(args.exclude_dataset)
    candidates = [
        dict(row) for row in audit["per_state"]
        if str(row["state_key"]) not in excluded and str(row.get("split")) == "val"
    ]
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_suite_task[str(row["suite"])][str(row["task_id"])].append(row)

    selected: list[dict[str, Any]] = []
    for suite in sorted(by_suite_task):
        tasks = by_suite_task[suite]
        if len(tasks) != 2:
            raise ValueError(f"expected two val tasks in {suite}, got {sorted(tasks)}")
        suite_selected = [row for task in sorted(tasks) for row in choose_two(tasks[task])]
        if len(suite_selected) != 4:
            raise ValueError(f"expected four states in {suite}")
        selected.extend(suite_selected)

    if len(selected) != 16 or len({row["state_key"] for row in selected}) != 16:
        raise ValueError("pilot must contain 16 unique states")
    records = [
        {
            "state_key": str(row["state_key"]),
            "task_id": str(row["task_id"]),
            "suite": str(row["suite"]),
            "historical_finite_safe": finite_safe(row),
            "historical_persistent_success": bool(row["operator_success"]["OFT_PERSISTENT"]),
        }
        for row in selected
    ]
    output = {
        "schema_version": "rase-pre-c0-r5-probability-pilot16/v1",
        "role": "development_label_entropy_only",
        "selection_rule": "four fresh states per suite; two per true task; maximize finite-safe/persistent-only diversity within task",
        "source_audit": str(args.audit.resolve()),
        "source_audit_sha256": sha256(args.audit),
        "excluded_dataset": str(args.exclude_dataset.resolve()) if args.exclude_dataset else None,
        "excluded_dataset_sha256": sha256(args.exclude_dataset) if args.exclude_dataset else None,
        "n_states": len(records),
        "n_tasks": len({row["task_id"] for row in records}),
        "n_suites": len({row["suite"] for row in records}),
        "historical_finite_safe_states": sum(row["historical_finite_safe"] for row in records),
        "records": records,
        "boundaries": [0, 16, 64, 128],
        "handback_repeats": 5,
        "formal_claim_scope": "none; protocol and label-entropy pilot",
        "frozen_test_access": "forbidden",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

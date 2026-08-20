#!/usr/bin/env python3
"""Freeze one train state per true task for the paired-repeat R5-B24 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_safe(row: dict[str, Any]) -> bool:
    return any(
        bool(value) for key, value in row["operator_success"].items()
        if str(key).startswith("OFT_H")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    if audit.get("safe_handback_status") != "ready":
        raise ValueError(f"source safe-handback gate is closed: {audit.get('safe_handback_reasons')}")
    if audit.get("deterministic_prefix_consistency_status") != "ready":
        raise ValueError("source audit has prefix inconsistency")
    rows = [dict(row) for row in audit["per_state"] if str(row.get("split")) == "train"]
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_suite_task[str(row["suite"])][str(row["task_id"])].append(row)

    negative_task_by_suite: dict[str, str] = {}
    for suite, tasks in sorted(by_suite_task.items()):
        candidates = sorted(
            task for task, task_rows in tasks.items()
            if any(not bool(row["operator_success"]["OFT_PERSISTENT"]) for row in task_rows)
        )
        if candidates:
            negative_task_by_suite[suite] = candidates[0]

    selected = []
    for suite, tasks in sorted(by_suite_task.items()):
        if len(tasks) != 6:
            raise ValueError(f"expected six train tasks in {suite}, got {len(tasks)}")
        for task, task_rows in sorted(tasks.items()):
            task_rows = sorted(task_rows, key=lambda row: str(row["state_key"]))
            if negative_task_by_suite.get(suite) == task:
                candidates = [
                    row for row in task_rows
                    if not bool(row["operator_success"]["OFT_PERSISTENT"])
                ]
                reason = "suite_persistent_failure_support"
            else:
                candidates = [row for row in task_rows if finite_safe(row)]
                reason = "task_finite_safe_lexicographic"
            if not candidates:
                raise ValueError(f"no candidate for {suite}/{task}/{reason}")
            row = candidates[0]
            selected.append({
                "state_key": str(row["state_key"]),
                "task_id": str(row["task_id"]),
                "suite": str(row["suite"]),
                "selection_reason": reason,
                "historical_finite_safe": finite_safe(row),
                "historical_persistent_success": bool(
                    row["operator_success"]["OFT_PERSISTENT"]
                ),
            })

    if len(selected) != 24 or len({row["task_id"] for row in selected}) != 24:
        raise ValueError("B24 must contain one state for each of 24 train tasks")
    manifest = {
        "schema_version": "rase-pre-c0-r5-b24-opportunity/v1",
        "role": "outcome_enriched_development_opportunity_screen",
        "formal_claim_scope": "none",
        "selection_uses_historical_development_outcomes": True,
        "selection_rule": (
            "one state per train task; one persistent failure in each suite with support; "
            "otherwise lexicographically first historical finite-safe state"
        ),
        "source_audit": str(args.audit.resolve()),
        "source_audit_sha256": sha256(args.audit),
        "qc_excluded_state_keys": audit.get("qc_excluded_state_keys", []),
        "n_states": len(selected),
        "n_tasks": len({row["task_id"] for row in selected}),
        "n_suites": len({row["suite"] for row in selected}),
        "historical_finite_safe_states": sum(row["historical_finite_safe"] for row in selected),
        "historical_persistent_success_states": sum(row["historical_persistent_success"] for row in selected),
        "records": selected,
        "boundaries": [0, 16, 32, 64, 96, 128],
        "handback_repeats": 5,
        "paired_repeat_seeds_across_boundaries": True,
        "opportunity_min_finite_states": 20,
        "opportunity_min_finite_tasks": 3,
        "opportunity_min_populated_bins": 2,
        "opportunity_min_savings": 0.25,
        "frozen_test_access": "forbidden",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

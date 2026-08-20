"""Contract and label-density audit for fixed-K discovery branches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def audit_discovery_feasibility(
    rows: list[dict[str, Any]], *, manifest: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    jobs = manifest.get("jobs", [])
    expected = {str(job["job_id"]): job for job in jobs}
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for row in rows:
        job_id = str(row.get("job_id", ""))
        if job_id not in expected:
            unknown.append(job_id)
            continue
        if job_id in seen:
            duplicates.append(job_id)
        else:
            seen[job_id] = row
    missing = sorted(set(expected) - set(seen))
    completed = [row for row in seen.values() if row.get("completed") is True]
    available = [row for row in completed if row.get("available") is True]
    completion_fraction = len(completed) / len(expected) if expected else 0.0
    unmasked_fraction = len(available) / len(completed) if completed else 0.0

    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_operator[str(row["operator_id"])].append(row)
        if row.get("available") is True:
            by_cell[(
                str(row["root_id"]), str(row["policy_id"]),
                str(row["decision_point_id"]),
            )].append(row)
    nondegenerate = 0
    eligible_cells = 0
    for values in by_cell.values():
        by_op: dict[str, list[float]] = defaultdict(list)
        for row in values:
            by_op[str(row["operator_id"])].append(float(row["success"]))
        if len(by_op) < 2:
            continue
        eligible_cells += 1
        means = [float(np.mean(scores)) for scores in by_op.values()]
        if max(means) - min(means) > 0:
            nondegenerate += 1
    nondegenerate_fraction = nondegenerate / eligible_cells if eligible_cells else 0.0
    operator_coverage = {
        operator: {
            "scheduled": len(values),
            "available": sum(row.get("available") is True for row in values),
            "available_fraction": (
                sum(row.get("available") is True for row in values) / len(values)
            ),
        }
        for operator, values in sorted(by_operator.items())
    }
    checks = {
        "unique_known_jobs": not duplicates and not unknown,
        "complete_schedule": not missing and completion_fraction >= float(
            gate["minimum_completed_branch_fraction"]
        ),
        "operator_mask_coverage": unmasked_fraction >= float(
            gate["minimum_unmasked_operator_fraction"]
        ),
        "nondegenerate_outcomes": nondegenerate_fraction >= float(
            gate["minimum_nondegenerate_outcome_fraction"]
        ),
    }
    return {
        "schema_version": "rase-vnext-discovery-feasibility-audit/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_jobs": len(expected),
        "observed_unique_jobs": len(seen),
        "completed_jobs": len(completed),
        "available_jobs": len(available),
        "completion_fraction": completion_fraction,
        "unmasked_operator_fraction": unmasked_fraction,
        "eligible_outcome_cells": eligible_cells,
        "nondegenerate_outcome_cells": nondegenerate,
        "nondegenerate_outcome_fraction": nondegenerate_fraction,
        "operator_coverage": operator_coverage,
        "missing_job_ids": missing,
        "duplicate_job_ids": sorted(set(duplicates)),
        "unknown_job_ids": sorted(set(unknown)),
        "checks": checks,
        "next_step": (
            "FREEZE_CONFIRMATION_COHORT" if all(checks.values())
            else "STOP_AND_REVISE_COLLECTION_OR_OPERATOR_PROTOCOL"
        ),
    }

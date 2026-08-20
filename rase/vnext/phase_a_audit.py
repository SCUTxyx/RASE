"""Strict, capability-aware Phase-A audits for a frozen RASE confirmation.

This module is deliberately read-only with respect to collection artifacts.  It
does not change the frozen opportunity gate implemented in ``opportunity.py``;
instead, it adds the integrity, non-abort, tie-aware, and partial-policy verdicts
required before any learned action-semantic model is unlocked.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .opportunity import RAW_FIELDS, aggregate_confirmation, nested_task_root_bootstrap, utility


ABORT_OPERATOR = "abort.safe"
REQUIRED_ID_FIELDS = (
    "root_id",
    "task_id",
    "suite",
    "policy_id",
    "decision_point_id",
    "operator_id",
    "exact_repeat_replica",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _job_seed(job: Mapping[str, Any], name: str) -> int | None:
    ledger = job.get("seed_ledger")
    if isinstance(ledger, Mapping) and name in ledger:
        return int(ledger[name])
    if name in job:
        return int(job[name])
    return None


def _row_seed(row: Mapping[str, Any], name: str) -> int | None:
    ledger = row.get("seed_ledger")
    if isinstance(ledger, Mapping) and name in ledger:
        return int(ledger[name])
    if name in row:
        return int(row[name])
    return None


def _row_completed(row: Mapping[str, Any]) -> bool:
    # Confirmation JSONL historically omitted ``completed`` for successful
    # atomic rows.  Presence in the file is completion unless explicitly false.
    return row.get("completed", True) is not False


def _row_available(row: Mapping[str, Any]) -> bool:
    return row.get("available", True) is True


def _finite_raw_fields(row: Mapping[str, Any]) -> list[str]:
    invalid: list[str] = []
    for field in RAW_FIELDS:
        if field not in row:
            invalid.append(f"missing:{field}")
            continue
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            invalid.append(f"non_numeric:{field}")
            continue
        if not math.isfinite(value):
            invalid.append(f"non_finite:{field}")
    return invalid


def audit_confirmation_integrity(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    repeats: int,
) -> dict[str, Any]:
    """Audit a frozen manifest against trial rows without reading outcomes for selection.

    The manifest is authoritative for job identity and contract availability.
    Masked jobs may either be present as unavailable rows or absent, but they can
    never appear as available.  Every available job must be present exactly once.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    jobs = list(manifest.get("jobs", []))
    if not jobs:
        raise ValueError("confirmation manifest contains no jobs")

    expected: dict[str, Mapping[str, Any]] = {}
    duplicate_manifest_jobs: list[str] = []
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        if not job_id:
            raise ValueError("manifest job is missing job_id")
        if job_id in expected:
            duplicate_manifest_jobs.append(job_id)
        expected[job_id] = job

    seen: dict[str, Mapping[str, Any]] = {}
    duplicate_rows: list[str] = []
    unknown_rows: list[str] = []
    row_contract_errors: list[dict[str, Any]] = []
    invalid_numeric_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        job_id = str(row.get("job_id", ""))
        if job_id not in expected:
            unknown_rows.append(job_id or f"<missing>@{index}")
            continue
        if job_id in seen:
            duplicate_rows.append(job_id)
            continue
        seen[job_id] = row
        job = expected[job_id]

        errors: list[str] = []
        for field in REQUIRED_ID_FIELDS[:-1]:
            expected_value = job.get(field)
            if field == "decision_point_id" and expected_value is None:
                point = job.get("decision_point")
                if isinstance(point, Mapping):
                    expected_value = point.get("decision_point_id")
            if str(row.get(field, "")) != str(expected_value):
                errors.append(f"{field}:{row.get(field)!r}!={expected_value!r}")
        expected_replica = _job_seed(job, "exact_repeat_replica")
        observed_replica = _row_seed(row, "exact_repeat_replica")
        if observed_replica != expected_replica:
            errors.append(
                f"exact_repeat_replica:{observed_replica!r}!={expected_replica!r}"
            )

        expected_available = job.get("available_by_contract", True) is True
        observed_available = _row_available(row)
        if observed_available and not expected_available:
            errors.append("contract-masked job appeared available")
        if expected_available and not observed_available:
            errors.append("contract-available job appeared unavailable")
        if not _row_completed(row):
            errors.append("row explicitly marked incomplete")
        if errors:
            row_contract_errors.append({"job_id": job_id, "errors": errors})
        if observed_available:
            invalid = _finite_raw_fields(row)
            if invalid:
                invalid_numeric_rows.append({"job_id": job_id, "errors": invalid})

    expected_available_ids = {
        job_id for job_id, job in expected.items()
        if job.get("available_by_contract", True) is True
    }
    expected_masked_ids = set(expected) - expected_available_ids
    observed_ids = set(seen)
    missing_available = sorted(expected_available_ids - observed_ids)
    absent_masked = sorted(expected_masked_ids - observed_ids)
    observed_masked = sorted(expected_masked_ids & observed_ids)

    # K/replica coverage is checked from the observed available rows, independent
    # of job ordering in the JSONL.
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for row in seen.values():
        if not _row_available(row):
            continue
        key = (
            str(row.get("root_id")),
            str(row.get("policy_id")),
            str(row.get("decision_point_id")),
            str(row.get("operator_id")),
        )
        replica = _row_seed(row, "exact_repeat_replica")
        if replica is not None:
            groups[key].append(replica)
    invalid_replica_groups = [
        {"key": list(key), "replicas": sorted(values)}
        for key, values in sorted(groups.items())
        if sorted(values) != list(range(repeats))
    ]

    expected_dimensions = {
        "roots": sorted({str(job.get("root_id")) for job in jobs}),
        "tasks": sorted({str(job.get("task_id")) for job in jobs}),
        "suites": sorted({str(job.get("suite")) for job in jobs}),
        "policies": sorted({str(job.get("policy_id")) for job in jobs}),
        "decision_points": sorted({
            str(
                job.get("decision_point_id")
                or (
                    job.get("decision_point", {}).get("decision_point_id")
                    if isinstance(job.get("decision_point"), Mapping) else ""
                )
            )
            for job in jobs
        }),
        "operators": sorted({str(job.get("operator_id")) for job in jobs}),
    }
    available_rows = [row for row in seen.values() if _row_available(row)]
    observed_dimensions = {
        "roots": sorted({str(row.get("root_id")) for row in available_rows}),
        "tasks": sorted({str(row.get("task_id")) for row in available_rows}),
        "suites": sorted({str(row.get("suite")) for row in available_rows}),
        "policies": sorted({str(row.get("policy_id")) for row in available_rows}),
        "decision_points": sorted({
            str(row.get("decision_point_id")) for row in available_rows
        }),
        "operators": sorted({str(row.get("operator_id")) for row in available_rows}),
    }

    checks = {
        "manifest_job_ids_unique": not duplicate_manifest_jobs,
        "row_job_ids_unique_known": not duplicate_rows and not unknown_rows,
        "all_available_jobs_present": not missing_available,
        "row_contract_matches_manifest": not row_contract_errors,
        "available_raw_fields_finite": not invalid_numeric_rows,
        "exact_k_replica_coverage": not invalid_replica_groups,
        "declared_expected_jobs_match": int(manifest.get("expected_jobs", len(jobs))) == len(jobs),
    }
    return {
        "schema_version": "rase-vnext-phase-a-integrity/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "manifest_jobs": len(jobs),
        "expected_available_jobs": len(expected_available_ids),
        "expected_masked_jobs": len(expected_masked_ids),
        "observed_unique_jobs": len(seen),
        "observed_available_jobs": len(available_rows),
        "observed_masked_jobs": len(observed_masked),
        "absent_masked_jobs_allowed": len(absent_masked),
        "missing_available_job_ids": missing_available,
        "duplicate_manifest_job_ids": sorted(set(duplicate_manifest_jobs)),
        "duplicate_row_job_ids": sorted(set(duplicate_rows)),
        "unknown_row_job_ids": sorted(set(unknown_rows)),
        "row_contract_errors": row_contract_errors,
        "invalid_numeric_rows": invalid_numeric_rows,
        "invalid_replica_groups": invalid_replica_groups,
        "expected_dimensions": expected_dimensions,
        "observed_available_dimensions": observed_dimensions,
        "checks": checks,
    }


def _gap(rows: Sequence[Mapping[str, Any]]) -> float:
    by_root_policy: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_operator: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_root_policy[(str(row["root_id"]), str(row["policy_id"]))].append(row)
        by_operator[str(row["operator_id"])].append(float(row["utility"]))
    if not by_root_policy or not by_operator:
        raise ValueError("opportunity gap requires roots and operators")
    oracle = float(np.mean([
        max(float(item["utility"]) for item in values)
        for values in by_root_policy.values()
    ]))
    best_fixed = max(float(np.mean(values)) for values in by_operator.values())
    return oracle - best_fixed


def paired_trial_operator_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, Any],
    excluded_operators: Iterable[str] = (),
    tie_margin: float = 0.0,
) -> dict[str, Any]:
    """Create tie-aware paired-trial evidence for each operator pair.

    Pairing uses root/policy/decision-point/replica.  The result is a diagnostic
    and a future soft-target source; it is not a replacement for the root-level
    preregistered gate.
    """
    if tie_margin < 0 or not math.isfinite(tie_margin):
        raise ValueError("tie_margin must be finite and non-negative")
    excluded = {str(value) for value in excluded_operators}
    cells: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        operator = str(row.get("operator_id"))
        if operator in excluded or not _row_available(row):
            continue
        invalid = _finite_raw_fields(row)
        if invalid:
            continue
        replica = _row_seed(row, "exact_repeat_replica")
        if replica is None:
            continue
        key = (
            str(row.get("root_id")),
            str(row.get("policy_id")),
            str(row.get("decision_point_id")),
            replica,
        )
        cells[key][operator] = utility(dict(row), dict(weights))

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"left_wins": 0, "right_wins": 0, "ties": 0, "pairs": 0}
    )
    for values in cells.values():
        for left, right in itertools.combinations(sorted(values), 2):
            diff = values[left] - values[right]
            record = counts[(left, right)]
            record["pairs"] += 1
            if diff > tie_margin:
                record["left_wins"] += 1
            elif diff < -tie_margin:
                record["right_wins"] += 1
            else:
                record["ties"] += 1

    return {
        f"{left}__vs__{right}": {
            **record,
            "soft_preference_left": (
                (record["left_wins"] + 0.5 * record["ties"]) / record["pairs"]
                if record["pairs"] else float("nan")
            ),
        }
        for (left, right), record in sorted(counts.items())
    }


def strict_opportunity_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    weights: Mapping[str, Any],
    gate: Mapping[str, Any],
    excluded_operators: Iterable[str] = (),
    tie_margin: float = 0.0,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 202708,
) -> dict[str, Any]:
    """Run a root-denominator, winner-support-aware opportunity audit."""
    excluded = {str(value) for value in excluded_operators}
    selected = [
        dict(row) for row in rows
        if str(row.get("operator_id")) not in excluded
    ]
    aggregated = aggregate_confirmation(
        selected, repeats=repeats, weights=dict(weights),
    )
    roots = sorted({str(row["root_id"]) for row in aggregated})
    tasks = sorted({str(row["task_id"]) for row in aggregated})
    suites = sorted({str(row["suite"]) for row in aggregated})
    policies = sorted({str(row["policy_id"]) for row in aggregated})
    gap = _gap(aggregated)
    lower, upper = nested_task_root_bootstrap(
        aggregated, samples=bootstrap_samples, seed=bootstrap_seed,
    )

    root_metadata: dict[str, tuple[str, str]] = {}
    by_root_operator: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in aggregated:
        root = str(row["root_id"])
        metadata = (str(row["task_id"]), str(row["suite"]))
        if root in root_metadata and root_metadata[root] != metadata:
            raise ValueError(f"root {root} maps to multiple task/suite identities")
        root_metadata[root] = metadata
        by_root_operator[(root, str(row["operator_id"]))].append(float(row["utility"]))

    by_root: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (root, operator), values in by_root_operator.items():
        by_root[root].append((operator, float(np.mean(values))))

    winner_roots: dict[str, set[str]] = defaultdict(set)
    unique_roots: dict[str, set[str]] = defaultdict(set)
    co_best_roots: dict[str, set[str]] = defaultdict(set)
    for root, values in by_root.items():
        best = max(value for _, value in values)
        winners = [operator for operator, value in values if best - value <= tie_margin]
        for operator in winners:
            winner_roots[operator].add(root)
            (unique_roots if len(winners) == 1 else co_best_roots)[operator].add(root)

    all_operators = sorted({str(row["operator_id"]) for row in aggregated})
    winner_summary: dict[str, dict[str, Any]] = {}
    for operator in all_operators:
        wins = winner_roots.get(operator, set())
        task_support = {root_metadata[root][0] for root in wins}
        suite_support = {root_metadata[root][1] for root in wins}
        winner_summary[operator] = {
            "roots": len(wins),
            "root_fraction": len(wins) / len(roots) if roots else 0.0,
            "unique_best_roots": len(unique_roots.get(operator, set())),
            "co_best_roots": len(co_best_roots.get(operator, set())),
            "tasks": len(task_support),
            "suites": sorted(suite_support),
        }

    minimum_fraction = float(gate["minimum_root_winner_fraction"])
    qualifying = [
        operator for operator in all_operators
        if winner_summary[operator]["root_fraction"] >= minimum_fraction
    ]
    qualifying_roots = set().union(
        *(winner_roots.get(operator, set()) for operator in qualifying),
    ) if qualifying else set()
    qualifying_tasks = {root_metadata[root][0] for root in qualifying_roots}
    qualifying_suites = {root_metadata[root][1] for root in qualifying_roots}

    policy_results: dict[str, dict[str, Any]] = {}
    meaningful = float(gate["minimum_oracle_minus_best_fixed"])
    for policy in policies:
        subset = [row for row in aggregated if row["policy_id"] == policy]
        policy_gap = _gap(subset)
        policy_results[policy] = {
            "oracle_minus_best_fixed": policy_gap,
            "passes_effect": policy_gap >= meaningful,
        }
    required_policies = {str(value) for value in gate["required_policies"]}
    checks = {
        "G_O1_mean_effect": gap >= meaningful,
        "G_O1_bootstrap_lower": lower > 0,
        "G_O2_operator_diversity": len(qualifying) >= int(gate["minimum_winner_operators"]),
        "G_O2_winner_task_coverage": len(qualifying_tasks) >= int(gate["minimum_tasks"]),
        "G_O2_winner_suite_coverage": len(qualifying_suites) >= int(gate["minimum_suites"]),
        "G_O3_policy_coverage": required_policies <= set(policies),
        "G_O3_policy_opportunity": all(
            policy_results.get(policy, {}).get("passes_effect", False)
            for policy in required_policies
        ),
    }
    raw_summary = {
        operator: {
            field: float(np.mean([
                row[field] for row in aggregated if row["operator_id"] == operator
            ]))
            for field in RAW_FIELDS
        }
        for operator in all_operators
    }
    return {
        "schema_version": "rase-vnext-strict-opportunity/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "excluded_operators": sorted(excluded),
        "tie_margin": tie_margin,
        "independence_unit": "physical_root",
        "root_policy_aggregation": "average_policies_before_root_winner",
        "roots": len(roots),
        "tasks": len(tasks),
        "suites": suites,
        "policies": policies,
        "oracle_minus_best_fixed": gap,
        "bootstrap_95_ci": [lower, upper],
        "winner_summary": winner_summary,
        "qualifying_winner_operators": qualifying,
        "qualifying_winner_task_coverage": len(qualifying_tasks),
        "qualifying_winner_suite_coverage": sorted(qualifying_suites),
        "policy_results": policy_results,
        "raw_metrics_by_operator": raw_summary,
        "paired_trial_operator_evidence": paired_trial_operator_evidence(
            selected, weights=weights, excluded_operators=(), tie_margin=tie_margin,
        ),
        "checks": checks,
    }


def audit_phase_a(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    repeats: int,
    weights: Mapping[str, Any],
    gate: Mapping[str, Any],
    abort_operator: str = ABORT_OPERATOR,
    tie_margin: float = 0.0,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 202708,
) -> dict[str, Any]:
    """Return integrity, full/non-abort opportunity, and an actionable verdict."""
    integrity = audit_confirmation_integrity(
        rows, manifest=manifest, repeats=repeats,
    )
    manifest_hash = canonical_json_sha256(manifest)
    if integrity["status"] != "PASS":
        return {
            "schema_version": "rase-vnext-phase-a-audit/v1",
            "status": "INTEGRITY_FAIL",
            "verdict": "REPAIR_OR_RESUME_FROZEN_COLLECTION",
            "manifest_sha256": manifest_hash,
            "integrity": integrity,
            "full_opportunity": None,
            "non_abort_opportunity": None,
            "unlocks": [],
        }

    full = strict_opportunity_audit(
        rows, repeats=repeats, weights=weights, gate=gate,
        tie_margin=tie_margin, bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    non_abort = strict_opportunity_audit(
        rows, repeats=repeats, weights=weights, gate=gate,
        excluded_operators={abort_operator}, tie_margin=tie_margin,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    required = [str(value) for value in gate["required_policies"]]
    passing_policies = [
        policy for policy in required
        if non_abort["policy_results"].get(policy, {}).get("passes_effect", False)
    ]
    if full["status"] == "PASS" and non_abort["status"] == "PASS":
        status = "A_PASS"
        verdict = "UNLOCK_PHASE_B_C"
        unlocks = ["canonical_motion_parity", "low_cost_action_sensitivity"]
    elif passing_policies or (
        non_abort["checks"]["G_O1_mean_effect"]
        and non_abort["checks"]["G_O2_operator_diversity"]
    ):
        status = "A_PARTIAL"
        verdict = "SINGLE_POLICY_PILOT_AND_INDEPENDENT_CHALLENGE_COHORT"
        unlocks = ["labeled_single_policy_pilot"]
    else:
        status = "A_FAIL"
        verdict = "REVISE_OPERATOR_OR_FAILURE_COVERAGE"
        unlocks = []
    return {
        "schema_version": "rase-vnext-phase-a-audit/v1",
        "status": status,
        "verdict": verdict,
        "manifest_sha256": manifest_hash,
        "integrity": integrity,
        "full_opportunity": full,
        "non_abort_opportunity": non_abort,
        "passing_non_abort_policies": passing_policies,
        "unlocks": unlocks,
        "remains_locked": ["semantic_pretraining", "world_model", "rl", "opd"],
    }

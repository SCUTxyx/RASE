"""Outcome-independent confirmation cohort and fixed-K schedule."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rase.vnext.discovery import (
    assign_task_folds,
    stable_hex,
    stable_seed,
    validate_root_catalog,
)


def select_confirmation_roots(
    records: list[dict[str, Any]], *, excluded_root_ids: set[str],
    roots_per_task: int, salt: str,
) -> list[dict[str, Any]]:
    """Select every task and rank roots using metadata-only SHA256 ordering."""
    validate_root_catalog(records)
    if roots_per_task < 1:
        raise ValueError("confirmation requires at least one root per task")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_suites: dict[str, set[str]] = defaultdict(set)
    for row in records:
        task = str(row["task_id"])
        task_suites[task].add(str(row["suite"]))
        if str(row["root_id"]) not in excluded_root_ids:
            by_task[task].append(row)
    if any(len(suites) != 1 for suites in task_suites.values()):
        raise ValueError("each task_id must belong to exactly one suite")
    selected = []
    for task in sorted(task_suites):
        candidates = by_task.get(task, [])
        if len(candidates) < roots_per_task:
            raise ValueError(
                f"task {task} has {len(candidates)} non-discovery roots; need {roots_per_task}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: stable_hex(salt, "confirmation-root", task, row["root_id"]),
        )
        selected.extend(ranked[:roots_per_task])
    return selected


def build_confirmation_manifest(
    root_catalog: list[dict[str, Any]], protocol: dict[str, Any], *,
    discovery_root_ids: set[str], roots_per_task: int, salt: str,
    operator_masks: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    collection = protocol["collection"]
    roots = select_confirmation_roots(
        root_catalog, excluded_root_ids=discovery_root_ids,
        roots_per_task=roots_per_task, salt=salt,
    )
    folds = assign_task_folds(
        (str(row["task_id"]) for row in roots), folds=5, salt=salt,
    )
    masks = operator_masks or {}
    repeats = int(collection["confirmation_repeats"])
    jobs = []
    for root in sorted(roots, key=lambda row: str(row["root_id"])):
        for policy in collection["source_policies"]:
            source_seed = stable_seed(salt, "source", root["root_id"], policy)
            for point in collection["decision_points"]:
                for operator in protocol["operators"]:
                    operator_id = str(operator["operator_id"])
                    mask_reason = masks.get((str(policy), operator_id))
                    operator_seed = stable_seed(
                        salt, "operator", root["root_id"], policy,
                        point["decision_point_id"], operator_id,
                    )
                    for replica in range(repeats):
                        job_id = stable_hex(
                            salt, "job", root["root_id"], policy,
                            point["decision_point_id"], operator_id, replica,
                        )[:24]
                        jobs.append({
                            "collection_phase": "confirmation",
                            "job_id": job_id,
                            "root_id": root["root_id"],
                            "task_id": root["task_id"],
                            "suite": root["suite"],
                            "outer_fold": folds[str(root["task_id"])],
                            "state_key": root["state_key"],
                            "restore_state_ref": root["restore_state_ref"],
                            "policy_id": policy,
                            "decision_point": point,
                            "operator_id": operator_id,
                            "operator_kind": operator["kind"],
                            "candidate_ids": operator["candidate_ids"],
                            "available_by_contract": mask_reason is None,
                            "contract_mask_reason": mask_reason,
                            "seed_ledger": {
                                "init_state_id": root["init_state_id"],
                                "environment_seed": root["environment_seed"],
                                "source_sampling_seed": source_seed,
                                "operator_seed": operator_seed,
                                "exact_repeat_replica": replica,
                            },
                        })
    suites = sorted({str(row["suite"]) for row in roots})
    tasks = sorted({str(row["task_id"]) for row in roots})
    return {
        "schema_version": "rase-vnext-confirmation-manifest/v1",
        "status": "frozen_confirmation",
        "selection_salt": salt,
        "selection_rule": (
            "all catalog tasks; exclude discovery physical roots; "
            "sha256 metadata-only root ranking"
        ),
        "fixed_repeats": repeats,
        "roots_per_task": roots_per_task,
        "excluded_discovery_root_ids": sorted(discovery_root_ids),
        "operator_masks": [
            {"policy_id": policy, "operator_id": operator, "reason": reason}
            for (policy, operator), reason in sorted(masks.items())
        ],
        "roots": roots,
        "task_folds": folds,
        "tasks": tasks,
        "suites": suites,
        "expected_roots": len(roots),
        "expected_jobs": len(jobs),
        "expected_available_jobs": sum(job["available_by_contract"] for job in jobs),
        "jobs": jobs,
        "forbidden_adaptations": [
            "outcome_dependent_k", "root_replacement", "preferred_repeat",
            "post_outcome_threshold_change", "unmask_after_outcome",
        ],
    }


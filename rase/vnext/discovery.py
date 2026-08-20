"""Outcome-independent discovery cohort and branch schedule construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable


FORBIDDEN_OUTCOME_KEYS = {
    "success", "is_success", "outcome", "label", "reward", "return", "harm",
    "utility", "cost", "failure", "failed", "oracle_value", "risk_score",
}
REQUIRED_ROOT_KEYS = {
    "root_id", "state_key", "task_id", "suite", "init_state_id",
    "environment_seed", "restore_state_ref",
}


def stable_hex(salt: str, *parts: object) -> str:
    payload = "\x1f".join([salt, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_seed(salt: str, *parts: object) -> int:
    return int(stable_hex(salt, *parts)[:8], 16) & 0x7FFFFFFF


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                result.append(path)
            result.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return result


def validate_root_catalog(records: list[dict]) -> None:
    if not records:
        raise ValueError("root catalog is empty")
    root_ids: set[str] = set()
    for index, row in enumerate(records):
        missing = REQUIRED_ROOT_KEYS - row.keys()
        if missing:
            raise ValueError(f"root[{index}] missing keys: {sorted(missing)}")
        forbidden = _forbidden_paths(row)
        if forbidden:
            raise ValueError(f"root[{index}] contains outcome-derived fields: {forbidden}")
        root_id = str(row["root_id"])
        if root_id in root_ids:
            raise ValueError(f"duplicate root_id: {root_id}")
        root_ids.add(root_id)
        if isinstance(row["init_state_id"], bool) or not isinstance(row["init_state_id"], int):
            raise ValueError(f"root[{index}] init_state_id must be an integer")
        if isinstance(row["environment_seed"], bool) or not isinstance(row["environment_seed"], int):
            raise ValueError(f"root[{index}] environment_seed must be an integer")


def select_discovery_roots(
    records: list[dict], *, tasks_per_suite: int, roots_per_task: int, salt: str
) -> list[dict]:
    validate_root_catalog(records)
    by_suite_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_suite_task[str(row["suite"])][str(row["task_id"])].append(row)
    selected: list[dict] = []
    for suite in sorted(by_suite_task):
        tasks = by_suite_task[suite]
        eligible = [task for task, roots in tasks.items() if len(roots) >= roots_per_task]
        if len(eligible) < tasks_per_suite:
            raise ValueError(
                f"suite {suite} has {len(eligible)} eligible tasks; need {tasks_per_suite}"
            )
        ranked_tasks = sorted(eligible, key=lambda task: stable_hex(salt, "task", suite, task))
        for task in ranked_tasks[:tasks_per_suite]:
            ranked_roots = sorted(
                tasks[task], key=lambda row: stable_hex(salt, "root", row["root_id"])
            )
            selected.extend(ranked_roots[:roots_per_task])
    return selected


def assign_task_folds(tasks: Iterable[str], *, folds: int, salt: str) -> dict[str, int]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    return {task: stable_seed(salt, "fold", task) % folds for task in sorted(set(tasks))}


def build_discovery_manifest(root_catalog: list[dict], protocol: dict, *, salt: str) -> dict:
    collection = protocol["collection"]
    roots = select_discovery_roots(
        root_catalog,
        tasks_per_suite=int(collection["discovery_tasks_per_suite"]),
        roots_per_task=int(collection["discovery_roots_per_task"]),
        salt=salt,
    )
    folds = assign_task_folds((str(row["task_id"]) for row in roots), folds=5, salt=salt)
    operators = protocol["operators"]
    policies = collection["source_policies"]
    points = collection["decision_points"]
    repeats = int(collection["discovery_repeats"])
    jobs: list[dict] = []
    for root in sorted(roots, key=lambda row: str(row["root_id"])):
        for policy in policies:
            source_seed = stable_seed(salt, "source", root["root_id"], policy)
            for point in points:
                for operator in operators:
                    operator_seed = stable_seed(
                        salt, "operator", root["root_id"], policy,
                        point["decision_point_id"], operator["operator_id"],
                    )
                    for replica in range(repeats):
                        job_id = stable_hex(
                            salt, "job", root["root_id"], policy,
                            point["decision_point_id"], operator["operator_id"], replica,
                        )[:24]
                        jobs.append({
                            "job_id": job_id,
                            "root_id": root["root_id"],
                            "task_id": root["task_id"],
                            "suite": root["suite"],
                            "outer_fold": folds[str(root["task_id"])],
                            "state_key": root["state_key"],
                            "restore_state_ref": root["restore_state_ref"],
                            "policy_id": policy,
                            "decision_point": point,
                            "operator_id": operator["operator_id"],
                            "operator_kind": operator["kind"],
                            "candidate_ids": operator["candidate_ids"],
                            "seed_ledger": {
                                "init_state_id": root["init_state_id"],
                                "environment_seed": root["environment_seed"],
                                "source_sampling_seed": source_seed,
                                "operator_seed": operator_seed,
                                "exact_repeat_replica": replica,
                            },
                        })
    return {
        "schema_version": "rase-vnext-discovery-manifest/v1",
        "status": "frozen_discovery",
        "selection_salt": salt,
        "selection_rule": "sha256 outcome-independent task then root ranking",
        "fixed_repeats": repeats,
        "roots": roots,
        "task_folds": folds,
        "expected_roots": len(roots),
        "expected_jobs": len(jobs),
        "jobs": jobs,
        "forbidden_adaptations": [
            "outcome_dependent_k", "root_replacement", "preferred_repeat",
            "post_outcome_threshold_change",
        ],
    }

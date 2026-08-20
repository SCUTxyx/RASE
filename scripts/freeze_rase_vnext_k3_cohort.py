#!/usr/bin/env python3
"""Freeze the formal K3 cohort: 8 tasks x 3 roots, 4 folds x 2 tasks, 432 slots.

Selection is strictly metadata-only (suite/task/root metadata and frozen hash
ranking); no outcome, capability or baseline result is read.  Near-duplicate
exclusion uses frozen proprio (qpos) L-infinity thresholds against the state
pool bundles; any violation replaces the root with the next hash-ranked
candidate from the same task (still metadata-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(salt: str, key: tuple[str, ...]) -> str:
    token = (salt + "\x1f" + "\x1f".join(map(str, key))).encode()
    return hashlib.sha256(token).hexdigest()


def stable_seed(salt: str, *parts: object) -> int:
    token = (salt + "\x1f" + "\x1f".join(str(part) for part in parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


K3_OPERATORS = (
    "continue.source", "requery.source",
    "resample.source/candidate.0", "resample.source/candidate.1",
    "fallback.persistent", "abort.safe",
)


def load_proprio(pool_path: Path, state_key: str) -> np.ndarray | None:
    """Read the frozen proprio (qpos) vector for a pool state, or None."""
    import numpy as np

    from rase.collect.state_pool import StatePool

    pool = StatePool(pool_path)
    try:
        loaded = pool.read_state(state_key, load_observations=False)
        return np.asarray(loaded.proprio, dtype=np.float64)
    except Exception:
        return None


def near_duplicate_violations(
    selected: list[dict[str, Any]], pool_path: Path, *, threshold: float,
) -> list[tuple[str, str, float]]:
    """Return (root_a, root_b, max_abs_diff) pairs violating the L-inf gate.

    Only same-task pairs are considered: cross-task states execute different
    task instructions and are not duplicate sampling even when proprio is close.
    """
    import numpy as np

    vectors: dict[str, np.ndarray | None] = {}
    for record in selected:
        key = str(record["state_key"])
        if key not in vectors:
            vectors[key] = load_proprio(pool_path, key)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_task[str(record["task_id"])].append(record)
    violations: list[tuple[str, str, float]] = []
    for task_records in by_task.values():
        for i in range(len(task_records)):
            for j in range(i + 1, len(task_records)):
                left = vectors.get(str(task_records[i]["state_key"]))
                right = vectors.get(str(task_records[j]["state_key"]))
                if left is None or right is None:
                    continue
                if left.shape != right.shape:
                    continue
                diff = float(np.max(np.abs(left - right)))
                if diff <= threshold:
                    violations.append(
                        (str(task_records[i]["root_id"]), str(task_records[j]["root_id"]), diff)
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", default="pi0fast.libero")
    parser.add_argument("--salt", default="rase-vnext-k3-cohort-v1")
    parser.add_argument("--decision-point", default="source.step.8")
    parser.add_argument("--tasks-per-suite", type=int, default=2)
    parser.add_argument("--roots-per-task", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    parent = json.loads(args.parent.read_text())
    protocol = json.loads(args.protocol.read_text())
    if catalog.get("status") not in {
        "frozen", "frozen_catalog", "frozen_outcome_independent",
    }:
        raise SystemExit(f"catalog status is {catalog.get('status')!r}")
    if parent.get("status") != "frozen_confirmation":
        raise SystemExit("parent must be a frozen confirmation manifest")
    if parent.get("protocol_sha256") != sha256(args.protocol):
        raise SystemExit("parent protocol hash does not match --protocol")

    # Suite -> task -> list of records (metadata only).
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in catalog["records"]:
        if str(record.get("suite")) in {"Spatial", "Object", "Goal", "Long"}:
            by_suite_task[str(record["suite"])][str(record["task_id"])].append(record)

    # Exclude B2/D0/E0 roots and any bundle-hash duplicates.
    excluded_root_ids = {
        str(root.get("root_id")) for root in parent.get("roots", [])
        if root.get("root_id", "").startswith("root.")
    }
    b2_path = args.catalog.parent / "b2_capture_smoke_manifest_v1.json"
    if b2_path.exists():
        b2 = json.loads(b2_path.read_text())
        excluded_root_ids.update(str(root["root_id"]) for root in b2.get("roots", []))
    e0_path = args.catalog.parent / "k3_e0_native_capture_smoke_manifest_v1.json"
    if e0_path.exists():
        e0 = json.loads(e0_path.read_text())
        excluded_root_ids.update(str(job["root_id"]) for job in e0.get("jobs", []))

    suites = sorted(by_suite_task)
    if len(suites) != 4:
        raise SystemExit(f"expected four suites, got {suites}")
    pool_path = Path(str(catalog["pool"])).resolve()

    # Per-suite task selection with root-level near-duplicate replacement; if a
    # task cannot field enough roots it is skipped and the next hash-ranked task
    # is tried (all metadata-only, frozen rule).
    selected: list[dict[str, Any]] = []
    selected_tasks: list[str] = []
    replacement_notes: list[str] = []
    for suite in suites:
        tasks = sorted(
            by_suite_task[suite],
            key=lambda task: (rank(args.salt, ("task", suite, task)), task),
        )
        task_choice: list[str] = []
        suite_roots: list[dict[str, Any]] = []
        for task in tasks:
            records = by_suite_task[suite][task]
            ordered = sorted(
                records,
                key=lambda rec: (rank(args.salt, ("root", suite, task, rec["root_id"])), rec["root_id"]),
            )
            usable = [
                rec for rec in ordered
                if str(rec["root_id"]) not in excluded_root_ids
            ]
            if len(usable) < args.roots_per_task:
                continue
            # Root-level near-duplicate replacement within this task.
            chosen: list[dict[str, Any]] = list(usable[: args.roots_per_task])
            for _round in range(8):
                if args.dry_run:
                    break
                violations = near_duplicate_violations(
                    chosen, pool_path, threshold=args.near_duplicate_threshold,
                )
                if not violations:
                    break
                left, right, diff = violations[0]
                chosen_ids = {str(rec["root_id"]) for rec in chosen}
                replacement = next(
                    (rec for rec in ordered
                     if str(rec["root_id"]) not in chosen_ids
                     and str(rec["root_id"]) not in excluded_root_ids),
                    None,
                )
                if replacement is None:
                    chosen = []
                    break
                if rank(args.salt, ("root", suite, task, left)) > rank(
                    args.salt, ("root", suite, task, right)
                ):
                    replace_key = left
                else:
                    replace_key = right
                chosen = [rec for rec in chosen if str(rec["root_id"]) != replace_key]
                chosen.append(replacement)
                replacement_notes.append(
                    f"{replace_key} replaced by {replacement['root_id']} "
                    f"(near-dup diff {diff:.4f}, {task})"
                )
            if len(chosen) == args.roots_per_task:
                task_choice.append(task)
                suite_roots.extend(chosen)
                if len(task_choice) == args.tasks_per_suite:
                    break
        if len(task_choice) != args.tasks_per_suite:
            raise SystemExit(
                f"{suite}: only {len(task_choice)} tasks with "
                f"{args.roots_per_task} non-near-duplicate roots, need {args.tasks_per_suite}"
            )
        selected_tasks.extend(task_choice)
        selected.extend(suite_roots)

    # Final cross-task near-duplicate check over the whole cohort.
    if not args.dry_run:
        remaining = near_duplicate_violations(selected, pool_path, threshold=args.near_duplicate_threshold)
        if remaining:
            raise SystemExit(f"near-duplicate violations remain after replacement: {remaining}")

    root_ids = {str(rec["root_id"]) for rec in selected}
    tasks = sorted({str(rec["task_id"]) for rec in selected})
    if len(root_ids) != len(tasks) * args.roots_per_task:
        raise SystemExit("root count mismatch after selection")
    if len(tasks) != len(suites) * args.tasks_per_suite:
        raise SystemExit("task count mismatch after selection")

    # Frozen folds: 4 folds x 2 tasks (hash-ranked, metadata only).
    task_ranks = {
        task: rank(args.salt, ("fold", task)) for task in tasks
    }
    task_folds: dict[str, int] = {}
    for index, task in enumerate(sorted(tasks, key=lambda t: (task_ranks[t], t))):
        task_folds[task] = index % 4
    fold_counts: dict[int, int] = defaultdict(int)
    for fold in task_folds.values():
        fold_counts[fold] += 1
    if any(count != args.tasks_per_suite for fold, count in fold_counts.items()):
        raise SystemExit(f"unbalanced folds: {dict(fold_counts)}")

    # Build jobs: 24 roots x 6 operators x 3 repeats (single decision point).
    jobs: list[dict[str, Any]] = []
    for record in sorted(selected, key=lambda r: (str(r["suite"]), str(r["task_id"]), str(r["root_id"]))):
        for operator in K3_OPERATORS:
            for replica in range(args.repeats):
                job_id = stable_seed(
                    args.salt, record["root_id"], operator, replica, "job",
                )
                jobs.append({
                    "available_by_contract": True,
                    "candidate_ids": [],
                    "collection_phase": "k3",
                    "contract_mask_reason": None,
                    "decision_point": {
                        "decision_point_id": args.decision_point,
                        "rule": "source_elapsed_step",
                        "value": int(args.decision_point.split(".")[-1]),
                    },
                    "job_id": f"{job_id:012x}",
                    "operator_id": operator,
                    "operator_kind": operator.split("/")[0],
                    "outer_fold": task_folds[str(record["task_id"])],
                    "policy_id": args.policy_id,
                    "restore_state_ref": str(record["restore_state_ref"]),
                    "root_id": str(record["root_id"]),
                    "seed_ledger": {
                        "environment_seed": int(record["environment_seed"]),
                        "exact_repeat_replica": int(replica),
                        "execution_seed": stable_seed(
                            args.salt, record["root_id"], "exec", replica,
                        ),
                        "init_state_id": int(record["init_state_id"]),
                        "operator_seed": stable_seed(
                            args.salt, record["root_id"], operator, replica,
                        ),
                        "source_sampling_seed": stable_seed(
                            args.salt, record["root_id"], "prefix", replica,
                        ),
                    },
                    "state_key": str(record["state_key"]),
                    "suite": str(record["suite"]),
                    "task_id": str(record["task_id"]),
                })

    expected_jobs = len(tasks) * args.roots_per_task * len(K3_OPERATORS) * args.repeats
    if len(jobs) != expected_jobs:
        raise SystemExit(f"job count {len(jobs)} != expected {expected_jobs}")
    if len(jobs) not in {432, 864, 1728}:
        raise SystemExit(
            f"job count {len(jobs)} is not a known cohort size (432/864/1728)"
        )

    manifest = {
        "schema_version": "rase-vnext-k3-cohort-manifest/v1",
        "status": "frozen_confirmation",
        "scientific_scope": (
            "FORMAL_K3_INDEPENDENT_ACTION_SIGNAL_PILOT: single-policy pi0fast "
            "development pilot; outcome-independent selection; gates K3-CAPTURE/"
            "SIGNAL/UTILITY"
        ),
        "parent_manifest": str(args.parent.resolve()),
        "parent_manifest_sha256": sha256(args.parent),
        "protocol_sha256": sha256(args.protocol),
        "root_catalog_pool": str(pool_path),
        "root_catalog_sha256": sha256(args.catalog),
        "selection_salt": args.salt,
        "selection_rule": (
            "metadata-only hash-ranked selection: 2 tasks/suite, 3 roots/task; "
            "B2/D0/E0 roots and bundle-hash duplicates excluded; near-duplicate "
            f"L-inf threshold {args.near_duplicate_threshold} on frozen proprio; "
            "no outcome/capability/baseline read"
        ),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "near_duplicate_replacements": replacement_notes,
        "decision_point": {
            "decision_point_id": args.decision_point,
            "rule": "source_elapsed_step",
            "value": int(args.decision_point.split(".")[-1]),
        },
        "fixed_repeats": args.repeats,
        "roots_per_task": args.roots_per_task,
        "tasks_per_suite": args.tasks_per_suite,
        "operators": list(K3_OPERATORS),
        "expected_roots": len(root_ids),
        "expected_jobs": expected_jobs,
        "expected_simulator_executions": (
            len(tasks) * args.roots_per_task
            * (len(K3_OPERATORS) - 1) * args.repeats
        ),
        "roots": selected,
        "tasks": tasks,
        "suites": suites,
        "task_folds": task_folds,
        "jobs": jobs,
        "forbidden_adaptations": sorted(set(parent.get("forbidden_adaptations", [])) | {
            "effect_claim_from_k3_capture_only",
            "outcome_dependent_selection",
            "pooled_multi_vla_claim_from_k3_pilot",
            "real_time_closed_loop_claim_from_k3_pilot",
            "resample_failure_as_ordinary_failure",
            "incapable_removal_from_denominator",
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        "output": str(args.output.resolve()),
        "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
        "dry_run": bool(args.dry_run),
        "roots": len(root_ids), "tasks": tasks,
        "task_folds": task_folds,
        "jobs": len(jobs),
        "simulator_executions": manifest["expected_simulator_executions"],
        "replacements": replacement_notes,
    }
    if not args.dry_run:
        temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

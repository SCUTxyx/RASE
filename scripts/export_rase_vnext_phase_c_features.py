#!/usr/bin/env python3
"""Replay frozen roots to export hash-aligned π0-fast Phase C pilot features.

This script never executes a post-decision outcome rollout.  It replays only
the frozen source prefix, exports the boundary observation and native source
candidate action chunks, then joins them to immutable confirmation labels by
job id and first-action SHA256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.libero import LiberoBenchmarkAdapter
from rase.vnext.phase_c_pilot import SOURCE_OPERATORS, choose_tasks, pad_action_chunk, stable_seed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype=np.float32).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def group_key(job: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(job["root_id"]), str(job["policy_id"]),
        str(job["decision_point"]["decision_point_id"]),
        int(job["seed_ledger"]["exact_repeat_replica"]),
    )


def group_digest(key: Sequence[object]) -> str:
    return hashlib.sha256("\x1f".join(map(str, key)).encode()).hexdigest()[:24]


def _policy_action(policy: Any, observation: Any, instruction: str) -> np.ndarray:
    return np.asarray(policy.act(observation, task=instruction), dtype=np.float32).reshape(-1, 7)[0]


def _queue_snapshot(continuation: Any) -> dict[Any, Any]:
    queues = getattr(continuation.policy_bundle["policy"], "_queues", None)
    if not isinstance(queues, dict):
        raise RuntimeError("policy does not expose a dictionary action queue")
    snapshot: dict[Any, Any] = {}
    for key, value in queues.items():
        if isinstance(value, deque):
            snapshot[key] = deque(value)
        else:
            snapshot[key] = copy.copy(value)
    return snapshot


def _restore_queues(continuation: Any, snapshot: Mapping[Any, Any]) -> None:
    restored: dict[Any, Any] = {}
    for key, value in snapshot.items():
        restored[key] = deque(value) if isinstance(value, deque) else copy.copy(value)
    continuation.policy_bundle["policy"]._queues = restored


def cached_chunk(
    continuation: Any,
    *,
    first_action: np.ndarray,
    observation: Any,
    instruction: str,
    horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Peek only the residual native queue and restore it exactly afterwards."""
    snapshot = _queue_snapshot(continuation)
    actions = [np.asarray(first_action, dtype=np.float32).reshape(7)]
    try:
        while len(actions) < horizon and not continuation._action_queue_empty():
            actions.append(_policy_action(continuation, observation, instruction))
    finally:
        _restore_queues(continuation, snapshot)
    return pad_action_chunk(actions, horizon=horizon)


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        job_id = str(row["job_id"])
        if job_id in rows:
            raise ValueError(f"duplicate branch job_id {job_id} at line {line_number}")
        rows[job_id] = row
    return rows


def expected_hash(value: Any, label: str) -> str:
    token = str(value or "")
    if len(token) != 64:
        raise ValueError(f"missing or invalid {label} hash")
    return token


def export_group(
    *,
    pool: Any,
    bundle: Mapping[str, Any],
    jobs: Sequence[dict[str, Any]],
    rows_by_job: Mapping[str, dict[str, Any]],
    output_dir: Path,
    libero_plus_root: str | None,
    operators: Sequence[str],
    alignment_attempt: int = 1,
) -> dict[str, Any]:
    from rase.collect.forked_rollout import InProcessLeRobotContinuation, restore_pool_state
    from scripts.collect_rase_vnext_discovery import prefix_to_decision, restore_boundary

    by_operator = {str(job["operator_id"]): job for job in jobs}
    operators = tuple(operators)
    missing = set(operators) - set(by_operator)
    if missing:
        raise ValueError(f"group is missing source operators: {sorted(missing)}")
    exemplar = jobs[0]
    key = group_key(exemplar)
    digest = group_digest(key)
    meta_path = output_dir / "groups" / f"{digest}.json"
    npz_path = output_dir / "features" / f"{digest}.npz"
    if meta_path.exists() and npz_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior.get("group_key") != list(key) or prior.get("features_sha256") != sha256(npz_path):
            raise RuntimeError(f"existing feature group failed provenance check: {digest}")
        return {"status": "skip", "group_key": list(key)}

    main = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
    source = InProcessLeRobotContinuation(
        bundle, seed=int(exemplar["seed_ledger"]["source_sampling_seed"]),
    )
    try:
        prefix = prefix_to_decision(
            main, source, decision_step=int(exemplar["decision_point"]["value"]),
        )
        if not prefix["available"]:
            payload = {
                "schema_version": "rase-vnext-phase-c-feature-group/v1",
                "status": "UNAVAILABLE", "group_key": list(key),
                "reason": prefix.get("reason"),
            }
            atomic_json(meta_path, payload)
            return payload

        instruction = str(prefix["instruction"])
        observation = prefix["observation"]
        benchmark = LiberoBenchmarkAdapter(vector_env=main.handle.vector_env, forkable=main.forkable)
        canonical = benchmark.observation_to_canonical(
            observation,
            task_text=instruction,
            timestamp_s=float(exemplar["decision_point"]["value"]) / 10.0,
        )

        chunks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        first_hashes: dict[str, str] = {}
        continue_first = np.asarray(prefix["boundary_action"], dtype=np.float32)
        chunks["continue.source"] = cached_chunk(
            source, first_action=continue_first, observation=observation,
            instruction=instruction,
        )
        first_hashes["continue.source"] = action_hash(continue_first)

        boundary_snapshot = prefix["snapshot"]
        requery_job = by_operator["requery.source"]
        requery_observation = restore_boundary(main, boundary_snapshot)
        source.seed = int(requery_job["seed_ledger"]["operator_seed"])
        source.reset()
        requery_first = _policy_action(source, requery_observation, instruction)
        chunks["requery.source"] = cached_chunk(
            source, first_action=requery_first, observation=requery_observation,
            instruction=instruction,
        )
        first_hashes["requery.source"] = action_hash(requery_first)

        candidate_chunks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        candidate_hashes: dict[str, str] = {}
        if "resample.source" in operators:
            resample_job = by_operator["resample.source"]
            operator_seed = int(resample_job["seed_ledger"]["operator_seed"])
            for candidate_id in ("candidate.0", "candidate.1"):
                candidate_observation = restore_boundary(main, boundary_snapshot)
                source.seed = stable_seed("rase-vnext-resample-v1", operator_seed, candidate_id)
                source.reset()
                first = _policy_action(source, candidate_observation, instruction)
                candidate_chunks[candidate_id] = cached_chunk(
                    source, first_action=first, observation=candidate_observation,
                    instruction=instruction,
                )
                candidate_hashes[candidate_id] = action_hash(first)

        outcome_rows = {
            operator: rows_by_job[str(by_operator[operator]["job_id"])]
            for operator in operators
        }
        selected_candidate: str | None = None
        if "resample.source" in operators:
            resample_row = outcome_rows["resample.source"]
            if resample_row.get("available") is not True or resample_row.get("utility") is None:
                raise RuntimeError("requested resample operator is unavailable in frozen outcomes")
            selected_candidate = str(resample_row["selected_candidate_id"])
            if selected_candidate not in candidate_chunks:
                raise RuntimeError(f"unknown selected resample candidate {selected_candidate}")
            chunks["resample.source"] = candidate_chunks[selected_candidate]
            first_hashes["resample.source"] = candidate_hashes[selected_candidate]

        expected = {
            "continue.source": expected_hash(
                outcome_rows["continue.source"].get("boundary_action_sha256"), "continue",
            ),
            "requery.source": expected_hash(
                outcome_rows["requery.source"].get("requery_first_action_sha256"), "requery",
            ),
        }
        checks = {
            "continue_first_action_hash": first_hashes["continue.source"] == expected["continue.source"],
            "requery_first_action_hash": first_hashes["requery.source"] == expected["requery.source"],
            "source_boundary_hash": first_hashes["continue.source"] == str(prefix["boundary_action_sha256"]),
        }
        if "resample.source" in operators:
            expected_candidates = {
                str(item["candidate_id"]): expected_hash(item["first_action_sha256"], "resample")
                for item in outcome_rows["resample.source"].get("candidates", [])
            }
            checks["resample_candidate_hashes"] = candidate_hashes == expected_candidates
        if not all(checks.values()):
            diagnostic = {
                "group_key": list(key), "checks": checks,
                "observed_first_action_sha256": first_hashes,
                "expected_first_action_sha256": expected,
                "observed_resample_candidate_sha256": candidate_hashes,
                "expected_resample_candidate_sha256": (
                    expected_candidates if "resample.source" in operators else {}
                ),
            }
            raise RuntimeError(
                "feature/outcome hash alignment failed: "
                + json.dumps(diagnostic, sort_keys=True)
            )

        actions = np.stack([chunks[operator][0] for operator in operators])
        masks = np.stack([chunks[operator][1] for operator in operators])
        arrays: dict[str, np.ndarray] = {
            "actions": actions,
            "action_step_mask": masks,
            "proprio": canonical.proprio,
            "proprio_mask": canonical.proprio_mask,
        }
        if candidate_chunks:
            arrays["resample_candidate_actions"] = np.stack([
                candidate_chunks[candidate][0] for candidate in ("candidate.0", "candidate.1")
            ])
            arrays["resample_candidate_step_mask"] = np.stack([
                candidate_chunks[candidate][1] for candidate in ("candidate.0", "candidate.1")
            ])
        for role, image in canonical.images.items():
            arrays[f"image_{role}"] = np.asarray(image, dtype=np.uint8)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = npz_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(npz_path)

        payload = {
            "schema_version": "rase-vnext-phase-c-feature-group/v1",
            "status": "COMPLETE", "group_key": list(key),
            "root_id": str(exemplar["root_id"]),
            "task_id": str(exemplar["task_id"]), "suite": str(exemplar["suite"]),
            "policy_id": str(exemplar["policy_id"]),
            "decision_point_id": str(exemplar["decision_point"]["decision_point_id"]),
            "decision_step": int(exemplar["decision_point"]["value"]),
            "exact_repeat_replica": int(exemplar["seed_ledger"]["exact_repeat_replica"]),
            "selection_scope": "A_PARTIAL_labeled_single_policy_pilot",
            "alignment_attempt": int(alignment_attempt),
            "operator_order": list(operators),
            "selected_resample_candidate": selected_candidate,
            "first_action_sha256": first_hashes,
            "alignment_checks": checks,
            "outcomes": {
                operator: {
                    key: outcome_rows[operator].get(key)
                    for key in ("job_id", "success", "harm", "query_cost", "latency_cost", "utility")
                }
                for operator in operators
            },
            "features_path": str(npz_path.resolve()),
            "features_sha256": sha256(npz_path),
        }
        atomic_json(meta_path, payload)
        return payload
    finally:
        main.close()


def build_contract(
    *, manifest: Mapping[str, Any], manifest_path: Path, branches_path: Path,
    policy_id: str, tasks: Sequence[str], max_replicas: int,
    operators: Sequence[str],
) -> dict[str, Any]:
    selected_jobs = [
        job for job in manifest["jobs"]
        if str(job["policy_id"]) == policy_id
        and str(job["task_id"]) in set(tasks)
        and int(job["seed_ledger"]["exact_repeat_replica"]) < max_replicas
    ]
    groups = sorted({group_key(job) for job in selected_jobs})
    return {
        "schema_version": "rase-vnext-phase-c-feature-contract/v1",
        "status": "FROZEN_BEFORE_FEATURE_REPLAY",
        "phase_a_scope": "A_PARTIAL_labeled_single_policy_pilot",
        "policy_id": policy_id,
        "task_selection_rule": "lexicographic_first_N_per_suite_from_frozen_manifest; outcomes_not_read",
        "tasks": list(tasks), "max_replicas": max_replicas,
        "operators": list(operators), "groups": [list(key) for key in groups],
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_branches": str(branches_path.resolve()),
        "source_branches_sha256": sha256(branches_path),
    }


def summarize(output_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = {group_digest(key) for key in contract["groups"]}
    complete: set[str] = set()
    unavailable: set[str] = set()
    unreproducible: set[str] = set()
    corrupt: list[str] = []
    alignment_attempts: list[int] = []
    for path in sorted((output_dir / "groups").glob("*.json")):
        try:
            data = json.loads(path.read_text())
            digest = path.stem
            if digest not in expected:
                corrupt.append(f"{path}: unexpected group")
            elif data.get("status") == "COMPLETE":
                feature_path = Path(str(data["features_path"]))
                if not feature_path.exists() or sha256(feature_path) != data["features_sha256"]:
                    corrupt.append(f"{path}: feature hash mismatch")
                else:
                    complete.add(digest)
                    alignment_attempts.append(int(data.get("alignment_attempt", 1)))
            elif data.get("status") == "UNAVAILABLE":
                unavailable.add(digest)
            elif data.get("status") == "UNREPRODUCIBLE":
                unreproducible.add(digest)
        except Exception as exc:
            corrupt.append(f"{path}: {type(exc).__name__}: {exc}")
    missing = sorted(expected - complete - unavailable - unreproducible)
    if missing or unavailable or corrupt:
        status = "INCOMPLETE"
    elif unreproducible:
        status = "PARTIAL_REPRODUCIBLE"
    else:
        status = "COMPLETE"
    report = {
        "schema_version": "rase-vnext-phase-c-feature-report/v1",
        "status": status,
        "expected_groups": len(expected), "complete_groups": len(complete),
        "unavailable_groups": len(unavailable), "missing_groups": len(missing),
        "unreproducible_groups": len(unreproducible),
        "corrupt_groups": corrupt,
        "alignment_retry_groups": sum(attempt > 1 for attempt in alignment_attempts),
        "maximum_alignment_attempt": max(alignment_attempts, default=0),
        "contract_sha256": sha256(output_dir / "EXPORT_CONTRACT.json"),
    }
    atomic_json(output_dir / "collection_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--policy-id", default="pi0fast.libero")
    parser.add_argument("--suite", choices=("Spatial", "Object", "Goal", "Long"))
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks-per-suite", type=int, default=0)
    parser.add_argument("--max-replicas", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--hash-alignment-retries", type=int, default=5,
        help="Retry prefix replay using action hashes only; outcomes are never consulted.",
    )
    parser.add_argument(
        "--operators", nargs="+",
        default=("continue.source", "requery.source"),
        choices=SOURCE_OPERATORS,
        help="Capability-aware source operators; π0-fast resample is frozen masked.",
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    branches_path = args.branches.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen_confirmation":
        raise SystemExit("Phase C feature replay requires the frozen confirmation manifest")
    if args.max_replicas <= 0:
        raise SystemExit("max-replicas must be positive")
    if args.hash_alignment_retries <= 0:
        raise SystemExit("hash-alignment-retries must be positive")
    policy_jobs = [job for job in manifest["jobs"] if str(job["policy_id"]) == args.policy_id]
    tasks = choose_tasks(policy_jobs, tasks_per_suite=args.tasks_per_suite)
    operators = tuple(args.operators)
    if not {"continue.source", "requery.source"}.issubset(operators):
        raise SystemExit("pilot requires continue.source and requery.source")
    contract = build_contract(
        manifest=manifest, manifest_path=manifest_path, branches_path=branches_path,
        policy_id=args.policy_id, tasks=tasks, max_replicas=args.max_replicas,
        operators=operators,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "EXPORT_CONTRACT.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise SystemExit("output directory is bound to a different export contract")
    else:
        atomic_json(contract_path, contract)
    if args.summarize:
        report = summarize(args.output_dir, contract)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "COMPLETE" else 3
    if not args.policy_path or not args.suite:
        raise SystemExit("feature replay requires --policy-path and --suite")

    rows_by_job = load_rows(branches_path)
    selected_task_set = set(tasks)
    selected_jobs = [
        job for job in policy_jobs
        if str(job["suite"]) == args.suite
        and str(job["task_id"]) in selected_task_set
        and int(job["seed_ledger"]["exact_repeat_replica"]) < args.max_replicas
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in selected_jobs:
        groups[group_key(job)].append(job)
    ordered = sorted(groups.items())
    if args.max_groups:
        ordered = ordered[: args.max_groups]

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import load_lerobot_policy_bundle
    from rase.collect.state_pool import StatePool

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()
    pool = StatePool(Path(str(manifest["root_catalog_pool"])).resolve())
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device=args.device, num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    for position, (key, jobs) in enumerate(ordered, 1):
        result = None
        last_alignment_error: RuntimeError | None = None
        for attempt in range(1, args.hash_alignment_retries + 1):
            try:
                result = export_group(
                    pool=pool, bundle=bundle, jobs=jobs, rows_by_job=rows_by_job,
                    output_dir=args.output_dir,
                    libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
                    operators=operators, alignment_attempt=attempt,
                )
                break
            except RuntimeError as exc:
                if "feature/outcome hash alignment failed" not in str(exc):
                    raise
                print(
                    f"PHASE_C_FEATURE_RETRY {args.suite} {position}/{len(ordered)} "
                    f"{key} attempt={attempt} reason=hash_mismatch",
                    flush=True,
                )
                if attempt == args.hash_alignment_retries:
                    last_alignment_error = exc
                    break
        if result is None and last_alignment_error is not None:
            digest = group_digest(key)
            exemplar = jobs[0]
            result = {
                "schema_version": "rase-vnext-phase-c-feature-group/v1",
                "status": "UNREPRODUCIBLE", "group_key": list(key),
                "root_id": str(exemplar["root_id"]),
                "task_id": str(exemplar["task_id"]),
                "suite": str(exemplar["suite"]),
                "policy_id": str(exemplar["policy_id"]),
                "decision_point_id": str(exemplar["decision_point"]["decision_point_id"]),
                "exact_repeat_replica": int(exemplar["seed_ledger"]["exact_repeat_replica"]),
                "alignment_attempts": args.hash_alignment_retries,
                "reason": "action_hash_mismatch_after_bounded_replay",
                "last_error": str(last_alignment_error),
            }
            atomic_json(args.output_dir / "groups" / f"{digest}.json", result)
        assert result is not None
        print(
            f"PHASE_C_FEATURE {args.suite} {position}/{len(ordered)} "
            f"{key} status={result['status']}", flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

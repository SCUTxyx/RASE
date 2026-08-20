#!/usr/bin/env python3
"""Resume-safe paired five-operator vNext discovery collection on LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OPERATORS = (
    "continue.source", "requery.source", "resample.source",
    "fallback.persistent", "abort.safe",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts: object) -> int:
    token = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def terminal_values(term: Any, trunc: Any, info: Any) -> tuple[bool, bool]:
    from rase.collect.policy_step import success_from_info

    terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    return terminal, bool(success_from_info(info)) if terminal else False


def action_hash(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype=np.float32).tobytes()).hexdigest()


def _policy_action(policy: Any, observation: Any, instruction: str) -> np.ndarray:
    value = policy.act(observation, task=instruction)
    return np.asarray(value, dtype=np.float32).reshape(-1, 7)[0]


def prefix_to_decision(restored: Any, policy: Any, *, decision_step: int) -> dict[str, Any]:
    """Execute the source prefix and stop after proposing, but before executing, a boundary action."""
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    restored.forkable.restore(
        restored.snapshot, check_task_fingerprint=restored.check_task_fingerprint,
    )
    single = restored.handle.vector_env.envs[0]
    vector_env = restored.handle.vector_env
    instruction = str(
        getattr(single, "task_description", "") or restored.loaded.metadata.instruction
    )
    observation = observation_from_libero_env(single)
    horizon = int(getattr(single, "_max_episode_steps", 600))
    policy.reset_metrics()
    policy.reset()
    elapsed = 0
    action_trace: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        while elapsed < decision_step:
            if current_timestep(restored.handle.control_env) >= horizon:
                return {"available": False, "reason": "horizon_before_decision", "elapsed": elapsed}
            action = _policy_action(policy, observation, instruction)
            action_trace.append(action.copy())
            observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
            elapsed += 1
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                return {
                    "available": False,
                    "reason": "terminal_before_decision",
                    "terminal_success": success,
                    "elapsed": elapsed,
                }
        boundary_action = _policy_action(policy, observation, instruction)
    except Exception as exc:
        return {
            "available": False,
            "reason": "source_policy_inference_error",
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:1000],
            "elapsed": elapsed,
        }
    return {
        "available": True,
        "instruction": instruction,
        "observation": observation,
        "snapshot": restored.forkable.snapshot(),
        "boundary_action": boundary_action,
        "boundary_action_sha256": action_hash(boundary_action),
        "source_prefix_action_sha256": hashlib.sha256(
            np.asarray(action_trace, dtype=np.float32).tobytes()
        ).hexdigest(),
        "source_prefix_steps": elapsed,
        "source_prefix_wall_s": time.perf_counter() - started,
        "simulator_timestep": int(current_timestep(restored.handle.control_env)),
    }


def rollout_policy(
    restored: Any, policy: Any, *, observation: Any, instruction: str,
    first_action: np.ndarray,
) -> dict[str, Any]:
    """Execute a proposed first action and then the policy queue to terminal/horizon."""
    from rase.collect.policy_step import as_batched_action, current_timestep

    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    horizon = int(getattr(single, "_max_episode_steps", 600))
    steps = 0
    success = False
    stop_reason = "horizon"
    action = np.asarray(first_action, dtype=np.float32)
    started = time.perf_counter()
    inference_error: dict[str, str] | None = None
    while current_timestep(restored.handle.control_env) < horizon:
        observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            stop_reason = "success" if success else "terminal_failure"
            break
        try:
            action = _policy_action(policy, observation, instruction)
        except Exception as exc:
            stop_reason = "policy_inference_error"
            inference_error = {
                "exception_type": type(exc).__name__, "exception": str(exc)[:1000],
            }
            success = False
            break
    return {
        "success": bool(success),
        "post_decision_env_steps": steps,
        "stop_reason": stop_reason,
        "policy_inference_error": inference_error,
        "branch_wall_s": time.perf_counter() - started,
        "policy_metrics": policy.metrics(),
    }


def restore_boundary(restored: Any, snapshot: Any) -> Any:
    from rase.collect.pool_candidates import observation_from_libero_env

    restored.forkable.restore(
        snapshot, check_task_fingerprint=restored.check_task_fingerprint,
    )
    return observation_from_libero_env(restored.handle.vector_env.envs[0])


def base_row(job: dict[str, Any], prefix: dict[str, Any]) -> dict[str, Any]:
    seed = job["seed_ledger"]
    return {
        "schema_version": "rase-vnext-discovery-branch/v1",
        "job_id": job["job_id"],
        "completed": True,
        "root_id": job["root_id"],
        "state_key": job["state_key"],
        "task_id": job["task_id"],
        "suite": job["suite"],
        "policy_id": job["policy_id"],
        "decision_point_id": job["decision_point"]["decision_point_id"],
        "decision_step": int(job["decision_point"]["value"]),
        "operator_id": job["operator_id"],
        "exact_repeat_replica": int(seed["exact_repeat_replica"]),
        "seed_ledger": seed,
        "source_prefix_steps": int(prefix.get("source_prefix_steps", prefix.get("elapsed", 0))),
        "source_prefix_action_sha256": prefix.get("source_prefix_action_sha256"),
        "boundary_action_sha256": prefix.get("boundary_action_sha256"),
    }


def finalize_group_rows(rows: list[dict[str, Any]], *, utility: dict[str, Any]) -> None:
    """Add paired harm, normalized costs, latency, and frozen utility in place."""
    available = {str(row["operator_id"]): row for row in rows if row.get("available") is True}
    continue_row = available.get("continue.source")
    if continue_row is None:
        return
    continue_success = bool(continue_row["success"])
    continue_latency = float(continue_row["branch_wall_s"])
    horizon = float(utility["normalization"]["max_episode_steps"])
    nominal_seconds = horizon / float(utility["normalization"]["control_hz"])
    for row in available.values():
        row["harm"] = float(continue_success and not bool(row["success"]))
        row["query_cost"] = float(row["intervention_query_count"]) / horizon
        row["fallback_cost"] = float(row["fallback_steps"]) / horizon
        row["latency_cost"] = max(0.0, float(row["branch_wall_s"]) - continue_latency) / nominal_seconds
        row["utility"] = (
            float(utility["success_reward"]) * float(row["success"])
            - float(utility["harm_weight"]) * row["harm"]
            - float(utility["query_weight"]) * row["query_cost"]
            - float(utility["fallback_weight"]) * row["fallback_cost"]
            - float(utility["latency_weight"]) * row["latency_cost"]
        )


def collect_group(
    *, pool: Any, bundle: Any, jobs: list[dict[str, Any]], client: Any,
    utility: dict[str, Any], libero_plus_root: str | None,
) -> dict[str, Any]:
    """Collect all five matched branches for one root-policy-point-replica cell."""
    from rase.collect.forked_rollout import InProcessLeRobotContinuation, restore_pool_state
    from scripts.collect_r6b1_dynamic_boundaries import persistent_branch, preserve_rng_state

    by_operator = {str(job["operator_id"]): job for job in jobs}
    if set(by_operator) != set(OPERATORS):
        raise ValueError(f"group does not contain the frozen five operators: {sorted(by_operator)}")
    exemplar = jobs[0]
    seed_ledger = exemplar["seed_ledger"]
    decision_step = int(exemplar["decision_point"]["value"])
    source_seed = int(seed_ledger["source_sampling_seed"])
    main = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
    source = InProcessLeRobotContinuation(bundle, seed=source_seed)
    started = time.perf_counter()
    try:
        prefix = prefix_to_decision(main, source, decision_step=decision_step)
        if not prefix["available"]:
            rows = []
            for operator in OPERATORS:
                row = base_row(by_operator[operator], prefix)
                row.update({
                    "available": False,
                    "mask_reason": prefix["reason"],
                    "success": None, "harm": None, "query_cost": None,
                    "fallback_cost": None, "latency_cost": None, "utility": None,
                    "source_prefix_diagnostic": {
                        key: value for key, value in prefix.items()
                        if key not in {"observation", "snapshot", "boundary_action"}
                    },
                })
                rows.append(row)
            return {"rows": rows, "prefix_available": False, "wall_s": time.perf_counter() - started}

        snapshot = prefix["snapshot"]
        instruction = str(prefix["instruction"])
        rows_by_operator: dict[str, dict[str, Any]] = {}

        # Continue preserves the source policy's exact action queue and the
        # already proposed boundary action.
        source.reset_metrics()
        result = rollout_policy(
            main, source, observation=prefix["observation"], instruction=instruction,
            first_action=prefix["boundary_action"],
        )
        row = base_row(by_operator["continue.source"], prefix)
        row.update(result)
        row.update({
            "available": True, "mask_reason": None,
            "intervention_query_count": 0, "fallback_steps": 0,
        })
        rows_by_operator["continue.source"] = row

        # Requery clears the cached source chunk and performs one intervention
        # query under its separately frozen operator seed.
        branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
        observation = restore_boundary(branch, snapshot)
        source.seed = int(by_operator["requery.source"]["seed_ledger"]["operator_seed"])
        source.reset_metrics()
        source.reset()
        requery_started = time.perf_counter()
        try:
            first_action = _policy_action(source, observation, instruction)
            result = rollout_policy(
                branch, source, observation=observation, instruction=instruction,
                first_action=first_action,
            )
            result["branch_wall_s"] = time.perf_counter() - requery_started
            row = base_row(by_operator["requery.source"], prefix)
            row.update(result)
            row.update({
                "available": True, "mask_reason": None,
                "intervention_query_count": 1, "fallback_steps": 0,
                "requery_first_action_sha256": action_hash(first_action),
            })
        except Exception as exc:
            row = base_row(by_operator["requery.source"], prefix)
            row.update({
                "available": True, "mask_reason": None, "success": False,
                "post_decision_env_steps": 0, "stop_reason": "policy_inference_error",
                "policy_inference_error": {
                    "exception_type": type(exc).__name__, "exception": str(exc)[:1000],
                },
                "branch_wall_s": time.perf_counter() - requery_started,
                "policy_metrics": source.metrics(),
                "intervention_query_count": 1, "fallback_steps": 0,
            })
        rows_by_operator["requery.source"] = row
        branch.close()

        # Resample creates two native candidates and uses the preregistered
        # minimum first-action L2 verifier. The selected candidate is regenerated
        # under the same seed because LeRobot policy queues are mutable.
        branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
        observation = restore_boundary(branch, snapshot)
        operator_seed = int(by_operator["resample.source"]["seed_ledger"]["operator_seed"])
        candidates = []
        resample_started = time.perf_counter()
        resample_error: Exception | None = None
        source.reset_metrics()
        for candidate_id in ("candidate.0", "candidate.1"):
            candidate_seed = stable_seed("rase-vnext-resample-v1", operator_seed, candidate_id)
            source.seed = candidate_seed
            source.reset()
            try:
                action = _policy_action(source, observation, instruction)
            except Exception as exc:
                resample_error = exc
                break
            candidates.append({
                "candidate_id": candidate_id, "seed": candidate_seed,
                "first_action": action,
                "first_action_l2": float(np.linalg.norm(action)),
                "first_action_sha256": action_hash(action),
            })
        if resample_error is None:
            selected = min(
                candidates, key=lambda item: (item["first_action_l2"], item["candidate_id"]),
            )
            source.seed = int(selected["seed"])
            source.reset()
            regenerated = _policy_action(source, observation, instruction)
            if action_hash(regenerated) != selected["first_action_sha256"]:
                raise RuntimeError("resample candidate regeneration was not exact under the frozen seed")
            result = rollout_policy(
                branch, source, observation=observation, instruction=instruction,
                first_action=regenerated,
            )
            result["branch_wall_s"] = time.perf_counter() - resample_started
            row = base_row(by_operator["resample.source"], prefix)
            row.update(result)
            row.update({
                "available": True, "mask_reason": None,
                "intervention_query_count": 3, "fallback_steps": 0,
                "candidate_verifier": "minimum_first_action_l2_then_candidate_id",
                "selected_candidate_id": selected["candidate_id"],
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "first_action"}
                    for candidate in candidates
                ],
            })
        else:
            row = base_row(by_operator["resample.source"], prefix)
            row.update({
                "available": True, "mask_reason": None, "success": False,
                "post_decision_env_steps": 0, "stop_reason": "policy_inference_error",
                "policy_inference_error": {
                    "exception_type": type(resample_error).__name__,
                    "exception": str(resample_error)[:1000],
                },
                "branch_wall_s": time.perf_counter() - resample_started,
                "policy_metrics": source.metrics(),
                "intervention_query_count": len(candidates) + 1, "fallback_steps": 0,
                "candidate_verifier": "minimum_first_action_l2_then_candidate_id",
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "first_action"}
                    for candidate in candidates
                ],
            })
        rows_by_operator["resample.source"] = row
        branch.close()

        # Persistent OFT uses only the frozen boundary snapshot and records the
        # number of native chunk queries for explicit compute cost.
        branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
        fallback_started = time.perf_counter()
        with preserve_rng_state():
            fallback = persistent_branch(
                branch, snapshot, client, instruction, record_chunk_trace=True,
            )
        row = base_row(by_operator["fallback.persistent"], prefix)
        row.update({
            "available": True, "mask_reason": None,
            "success": bool(fallback["success"]),
            "post_decision_env_steps": int(fallback["steps"]),
            "stop_reason": "success" if fallback["success"] else "terminal_or_horizon_failure",
            "branch_wall_s": time.perf_counter() - fallback_started,
            "intervention_query_count": len(fallback.get("chunk_query_records", [])),
            "fallback_steps": int(fallback["steps"]),
            "fallback_action_trace_sha256": fallback["action_trace_sha256"],
            "fallback_action_trace_shape": fallback["action_trace_shape"],
        })
        rows_by_operator["fallback.persistent"] = row
        branch.close()

        row = base_row(by_operator["abort.safe"], prefix)
        row.update({
            "available": True, "mask_reason": None, "success": False,
            "post_decision_env_steps": 0, "stop_reason": "safe_abort",
            "branch_wall_s": 0.0, "intervention_query_count": 0,
            "fallback_steps": 0,
        })
        rows_by_operator["abort.safe"] = row

        rows = [rows_by_operator[operator] for operator in OPERATORS]
        finalize_group_rows(rows, utility=utility)
        return {"rows": rows, "prefix_available": True, "wall_s": time.perf_counter() - started}
    finally:
        main.close()


def group_key(job: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(job["root_id"]), str(job["policy_id"]),
        str(job["decision_point"]["decision_point_id"]),
        int(job["seed_ledger"]["exact_repeat_replica"]),
    )


def group_path(output_dir: Path, key: tuple[str, str, str, int]) -> Path:
    digest = hashlib.sha256("\x1f".join(map(str, key)).encode()).hexdigest()[:24]
    return output_dir / "groups" / f"{digest}.json"


def summarize(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    expected = {str(job["job_id"]) for job in manifest["jobs"]}
    rows: list[dict[str, Any]] = []
    corrupt: list[str] = []
    for path in sorted((output_dir / "groups").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            if payload.get("manifest_sha256") != sha256(Path(str(output_dir / "manifest.bound.json"))):
                corrupt.append(f"{path}: bound manifest hash mismatch")
                continue
            rows.extend(payload["rows"])
        except Exception as exc:
            corrupt.append(f"{path}: {type(exc).__name__}: {exc}")
    observed_ids = [str(row.get("job_id", "")) for row in rows]
    duplicates = sorted({job_id for job_id in observed_ids if observed_ids.count(job_id) > 1})
    unknown = sorted(set(observed_ids) - expected)
    missing = sorted(expected - set(observed_ids))
    branches = output_dir / "branches.jsonl"
    temporary = branches.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(branches)
    report = {
        "schema_version": "rase-vnext-discovery-collection-report/v1",
        "status": "COMPLETE" if not (corrupt or duplicates or unknown or missing) else "INCOMPLETE",
        "expected_jobs": len(expected), "observed_rows": len(rows),
        "available_rows": sum(row.get("available") is True for row in rows),
        "masked_rows": sum(row.get("available") is False for row in rows),
        "success_rows": sum(row.get("available") is True and bool(row.get("success")) for row in rows),
        "missing_job_ids": missing, "duplicate_job_ids": duplicates,
        "unknown_job_ids": unknown, "corrupt_group_files": corrupt,
        "branches": str(branches.resolve()),
        "branches_sha256": sha256(branches),
    }
    atomic_json(output_dir / "collection_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--policy-id")
    parser.add_argument("--suite")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    protocol_path = args.protocol.resolve()
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if manifest.get("status") != "frozen_discovery":
        raise SystemExit("manifest is not frozen_discovery")
    if manifest.get("protocol_sha256") != sha256(protocol_path):
        raise SystemExit("manifest protocol hash does not match the supplied frozen protocol")
    for point in protocol["collection"]["decision_points"]:
        if point.get("rule") != "source_elapsed_step":
            raise SystemExit("noncausal decision-point rule is forbidden")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bound_manifest = args.output_dir / "manifest.bound.json"
    if bound_manifest.exists() and sha256(bound_manifest) != sha256(manifest_path):
        raise SystemExit("output directory is already bound to a different manifest")
    if not bound_manifest.exists():
        bound_manifest.write_bytes(manifest_path.read_bytes())
    if args.summarize:
        report = summarize(manifest, args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "COMPLETE" else 3
    if not args.policy_path or not args.policy_id or not args.suite:
        raise SystemExit("collection requires --policy-path, --policy-id, and --suite")

    selected_jobs = [
        job for job in manifest["jobs"]
        if str(job["policy_id"]) == args.policy_id and str(job["suite"]) == args.suite
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in selected_jobs:
        groups[group_key(job)].append(job)
    ordered = sorted(groups.items())
    if args.max_groups:
        ordered = ordered[:args.max_groups]
    if not ordered:
        raise SystemExit("no manifest groups match the requested suite and policy")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import load_lerobot_policy_bundle
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()
    pool = StatePool(Path(str(manifest["root_catalog_pool"])).resolve())
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device=args.device, num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    actual_suite = {
        "Spatial": "libero_spatial", "Object": "libero_object",
        "Goal": "libero_goal", "Long": "libero_10",
    }[args.suite]
    if model_info.get("suite") not in {None, actual_suite}:
        raise SystemExit(f"oracle suite mismatch: {model_info.get('suite')} != {actual_suite}")

    manifest_hash = sha256(bound_manifest)
    collector_hash = sha256(Path(__file__).resolve())
    completed = 0
    for position, (key, jobs) in enumerate(ordered, 1):
        path = group_path(args.output_dir, key)
        expected_ids = {str(job["job_id"]) for job in jobs}
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("manifest_sha256") != manifest_hash:
                raise SystemExit(f"existing group has a different manifest hash: {path}")
            if {str(row["job_id"]) for row in prior.get("rows", [])} != expected_ids:
                raise SystemExit(f"existing group has a different job set: {path}")
            print(f"VNEXT skip {args.suite}/{args.policy_id} {position}/{len(ordered)} {key}", flush=True)
            completed += 1
            continue
        result = collect_group(
            pool=pool, bundle=bundle, jobs=jobs, client=client,
            utility=protocol["utility"],
            libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
        )
        payload = {
            "schema_version": "rase-vnext-discovery-group/v1",
            "status": "complete", "group_key": list(key),
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "collector_sha256": collector_hash,
            "oracle_model_info": model_info,
            **result,
        }
        atomic_json(path, payload)
        completed += 1
        successes = sum(row.get("available") is True and bool(row.get("success")) for row in result["rows"])
        print(
            f"VNEXT done {args.suite}/{args.policy_id} {position}/{len(ordered)} "
            f"available={result['prefix_available']} successes={successes}/5 wall_s={result['wall_s']:.1f}",
            flush=True,
        )
    print(json.dumps({
        "status": "batch_complete", "suite": args.suite, "policy_id": args.policy_id,
        "completed_groups": completed, "scheduled_groups": len(ordered),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resume-safe paired five-operator vNext discovery collection on LIBERO."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OPERATORS = (
    "continue.source", "requery.source",
    "resample.source/candidate.0", "resample.source/candidate.1",
    "fallback.persistent", "abort.safe",
)

# Pre-K3 manifests encode resample as a single operator job.
OPERATORS_LEGACY = (
    "continue.source", "requery.source", "resample.source",
    "fallback.persistent", "abort.safe",
)

CAPTURE_HORIZON = 10


@dataclass(frozen=True)
class CaptureChunk:
    """One operator's frozen candidate-capture payload (K3-E0 contract v2).

    ``actions`` is the candidate chunk in env space (may be a suffix of
    ``full_env_chunk`` for continue when the boundary falls inside a partially
    consumed inference event); ``capability`` follows the frozen schema
    (executable / incapable_* / control_only_abort / execution_error).
    """

    capability: str
    chunk_origin: str
    actions: np.ndarray | None = None
    full_env_chunk: np.ndarray | None = None
    native_chunk_sha256: str | None = None
    inference_event_id: str | None = None
    queue_cursor_at_boundary: int | None = None
    candidate_generation_seed: int | None = None
    boundary_action: np.ndarray | None = None
    mask_reason: str | None = None


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


# --- K3-E0 native capture helpers -------------------------------------------
#
# The K3 capture contract forbids reconstructing candidate chunks from
# LeRobot's mutable action queue.  Instead every native ``predict_action_chunk``
# forward is recorded as an immutable InferenceEvent at inference time; the
# queue cursor at the decision boundary is bookkept on the continuation.
# Requery/resample run on an isolated policy state (queue + RNG) so they can
# never pollute the continue branch's boundary state.


def isolated_force_inference(
    bundle: Any,
    observation: Any,
    *,
    task: str,
    boundary_step: int,
    generation_seed: int | None,
    horizon: int = 10,
) -> tuple[np.ndarray, Any]:
    """Clear the queue and run exactly one native inference on an isolated state.

    The caller's policy queue/RNG are snapshotted before and restored after, so
    the continue branch can later resume its exact boundary state.
    """
    from rase.collect.candidates import seed_everything
    from rase.collect.policy_step import (
        capture_inference_event,
        clear_policy_queues,
        policy_state_restore,
        policy_state_snapshot,
    )

    snapshot = policy_state_snapshot(bundle)
    try:
        bundle["policy"].reset()
        clear_policy_queues(bundle["policy"])
        if generation_seed is not None:
            seed_everything(int(generation_seed))
        return capture_inference_event(
            bundle, observation, task=task,
            boundary_step=boundary_step,
            generation_seed=generation_seed,
            horizon=horizon,
        )
    finally:
        policy_state_restore(bundle, snapshot)


def isolated_forward(
    bundle: Any,
    observation: Any,
    *,
    task: str,
    boundary_step: int,
    generation_seed: int | None,
    horizon: int = 10,
) -> tuple[np.ndarray, Any]:
    """Run one native inference with the *current* RNG state, fully isolated.

    Used when the source queue is exactly exhausted at the boundary: the
    natural next act() would trigger a fresh inference, so we pre-run it
    (same RNG state => identical chunk) and restore the policy state so the
    continue rollout reproduces the same chunk itself.
    """
    from rase.collect.policy_step import capture_inference_event, policy_state_restore, policy_state_snapshot

    snapshot = policy_state_snapshot(bundle)
    try:
        return capture_inference_event(
            bundle, observation, task=task,
            boundary_step=boundary_step,
            generation_seed=generation_seed,
            horizon=horizon,
        )
    finally:
        policy_state_restore(bundle, snapshot)


def _extract_proprio(observation: Any) -> np.ndarray | None:
    """Best-effort proprio vector from a (possibly nested) gym observation.

    LeRobot gym observations wrap proprio as ``robot_state`` dict with
    ``joints.pos``, ``gripper.qpos``, ``eef.pos/quat`` arrays.
    """
    rs = observation.get("robot_state")
    if isinstance(rs, dict):
        parts: list[np.ndarray] = []
        for group in ("joints", "gripper"):
            sub = rs.get(group)
            if isinstance(sub, dict):
                for subkey in ("pos", "qpos"):
                    if subkey in sub:
                        try:
                            parts.append(np.asarray(sub[subkey], dtype=np.float64).reshape(-1))
                        except Exception:
                            pass
                        break
        if parts:
            try:
                array = np.concatenate(parts)
                if 4 <= array.size <= 64 and np.isfinite(array).all():
                    return array
            except Exception:
                pass
        for key in ("eef",):
            sub = rs.get(key)
            if isinstance(sub, dict) and "pos" in sub:
                try:
                    array = np.asarray(sub["pos"], dtype=np.float64).reshape(-1)
                    if 4 <= array.size <= 64 and np.isfinite(array).all():
                        return array
                except Exception:
                    pass
    candidates: list[Any] = []
    value = observation.get("robot_state")
    if value is not None and not isinstance(value, dict):
        candidates.append(value)
    for key in ("proprio", "agent_pos", "state"):
        if key in observation:
            candidates.append(observation[key])
    for value in candidates:
        if isinstance(value, dict):
            for sub in ("robot_state", "state", "proprio", "agent_pos"):
                if sub in value:
                    value = value[sub]
                    break
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if 4 <= array.size <= 64 and array.dtype != object and np.isfinite(array).all():
                return array
        except Exception:
            continue
    # fallback: first small non-image array-like value
    for key, value in observation.items():
        if key in ("pixels", "task", "instruction", "robot_state"):
            continue
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if 4 <= array.size <= 64 and array.dtype != object and np.isfinite(array).all():
                return array
        except Exception:
            continue
    return None


def prefix_to_decision(
    restored: Any, policy: Any, *,
    decision_step: int | None = None,
    detector: Any = None,
    max_steps: int = 80,
) -> dict[str, Any]:
    """Execute the source prefix and stop after proposing, but before executing,
    a boundary action.

    Static mode (decision_step given) keeps the legacy fixed-step boundary.
    Detector mode (decision_step=None, detector given) executes the source
    rollout until the causal detector fires (or max_steps), returning full
    trigger provenance plus decomposed prefix timing.
    """
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    if decision_step is None and detector is None:
        raise ValueError("prefix_to_decision requires decision_step or detector")
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
    if hasattr(policy, "note_boundary_step"):
        policy.note_boundary_step(decision_step if decision_step is not None else max_steps)
    elapsed = 0
    action_trace: list[np.ndarray] = []
    started = time.perf_counter()
    inference_wall = 0.0
    env_wall = 0.0
    trigger_provenance: Any = None
    try:
        while True:
            if decision_step is not None:
                if elapsed >= decision_step:
                    break
            else:
                if trigger_provenance is not None or elapsed >= max_steps:
                    break
            if current_timestep(restored.handle.control_env) >= horizon:
                return {"available": False, "reason": "horizon_before_decision", "elapsed": elapsed}
            infer_started = time.perf_counter()
            action = _policy_action(policy, observation, instruction)
            inference_wall += time.perf_counter() - infer_started
            action_trace.append(action.copy())
            if detector is not None:
                detector.update(
                    action, proprio=_extract_proprio(observation),
                )
            env_started = time.perf_counter()
            observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
            env_wall += time.perf_counter() - env_started
            elapsed += 1
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                return {
                    "available": False,
                    "reason": "terminal_before_decision",
                    "terminal_success": success,
                    "elapsed": elapsed,
                }
            if detector is not None and decision_step is None:
                trigger_provenance = detector.evaluate(elapsed, time.perf_counter())
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
        "trigger_provenance": (
            trigger_provenance.to_dict() if trigger_provenance is not None else None
        ),
        "boundary_rule": (
            str(trigger_provenance.rule) if trigger_provenance is not None else "static"
        ),
        "source_prefix_wall_s": time.perf_counter() - started,
        "source_prefix_inference_wall_s": inference_wall,
        "source_prefix_env_wall_s": env_wall,
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
    phase = str(job.get("collection_phase", "discovery"))
    return {
        "schema_version": f"rase-vnext-{phase}-branch/v1",
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
        "boundary_rule": prefix.get("boundary_rule", "static"),
        "trigger_provenance": prefix.get("trigger_provenance"),
        "source_prefix_wall_s": prefix.get("source_prefix_wall_s"),
        "source_prefix_inference_wall_s": prefix.get("source_prefix_inference_wall_s"),
        "source_prefix_env_wall_s": prefix.get("source_prefix_env_wall_s"),
    }


def finalize_group_rows(rows: list[dict[str, Any]], *, utility: dict[str, Any]) -> None:
    """Add paired harm, normalized costs, latency, and frozen utility in place."""
    available = {
        str(row["operator_id"]): row for row in rows
        if row.get("available") is True and row.get("execution_status") != "not_selected"
    }
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
    candidate_capture_dir: Path | None = None,
) -> dict[str, Any]:
    """Collect all five matched branches for one root-policy-point-replica cell."""
    from rase.collect.forked_rollout import InProcessLeRobotContinuation, restore_pool_state
    from scripts.collect_r6b1_dynamic_boundaries import persistent_branch, preserve_rng_state

    by_operator = {str(job["operator_id"]): job for job in jobs}
    operator_ids = set(by_operator)
    if operator_ids != set(OPERATORS) and operator_ids != set(OPERATORS_LEGACY):
        raise ValueError(
            f"group does not contain the frozen operators: {sorted(operator_ids)}"
        )
    legacy_resample = "resample.source" in operator_ids
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
        candidate_chunks: dict[str, Any] = {}
        executed_first_hashes: dict[str, str] = {}
        fallback_full_action_trace: np.ndarray | None = None

        # === continue.source: read boundary inference-event provenance (never queue).
        source.reset_metrics()
        cont_event = source.current_inference_event()
        cont_cursor = source.consumed_in_current_event()
        if cont_event is None:
            continue_cap = "incapable_missing"
            continue_origin = "no_inference_event"
            continue_chunk = None
            continue_full = None
            continue_event_id = None
            continue_cursor = None
            continue_reason = "no_inference_event_at_boundary"
        else:
            continue_event_id = cont_event.inference_event_id
            continue_cursor = int(cont_cursor)
            if cont_cursor >= cont_event.env_chunk.shape[0]:
                # Queue exactly exhausted at the boundary: pre-run the natural
                # next inference (same RNG state) so the chunk is frozen before
                # execution; the continue rollout reproduces the same chunk.
                _first2, cont_event2 = isolated_forward(
                    bundle, prefix["observation"], task=instruction,
                    boundary_step=decision_step,
                    generation_seed=cont_event.candidate_generation_seed,
                    horizon=CAPTURE_HORIZON,
                )
                continue_chunk = cont_event2.env_chunk
                continue_full = cont_event2.env_chunk
                continue_event_id = cont_event2.inference_event_id
                continue_cursor = 0
            else:
                continue_chunk = cont_event.env_chunk[cont_cursor:]
                continue_full = cont_event.env_chunk
            continue_cap = "executable"
            continue_origin = "inference_event"
            continue_reason = None
        if candidate_capture_dir is not None:
            from rase.vnext.candidate_capture import array_sha256

            candidate_chunks["continue.source"] = CaptureChunk(
                capability=continue_cap, chunk_origin=continue_origin,
                actions=continue_chunk, full_env_chunk=continue_full,
                native_chunk_sha256=(
                    array_sha256(cont_event.native_chunk)
                    if cont_event is not None and continue_cursor is not None
                    else None
                ),
                inference_event_id=continue_event_id,
                queue_cursor_at_boundary=continue_cursor,
                candidate_generation_seed=(
                    cont_event.candidate_generation_seed if cont_event is not None else None
                ),
                boundary_action=prefix["boundary_action"],
                mask_reason=continue_reason,
            )
        executed_first_hashes["continue.source"] = action_hash(prefix["boundary_action"])
        result = rollout_policy(
            main, source, observation=prefix["observation"], instruction=instruction,
            first_action=prefix["boundary_action"],
        )
        row = base_row(by_operator["continue.source"], prefix)
        row.update(result)
        row.update({
            "available": continue_cap == "executable", "mask_reason": continue_reason,
            "capability_status": continue_cap, "chunk_origin": continue_origin,
            "inference_event_id": continue_event_id,
            "queue_cursor_at_boundary": continue_cursor,
            "intervention_query_count": 0, "fallback_steps": 0,
        })
        rows_by_operator["continue.source"] = row

        # === requery.source: isolated forced inference (queue cleared, own
        # === operator seed), captured AND executed before the continue branch
        # === runs; the policy state is snapshotted first and restored after, so
        # === the continue branch resumes its exact boundary state.
        from rase.collect.policy_step import (
            capture_inference_event,
            policy_state_restore,
            policy_state_snapshot,
        )

        requery_job = by_operator["requery.source"]
        requery_seed = int(requery_job["seed_ledger"]["operator_seed"])
        requery_cont = InProcessLeRobotContinuation(
            bundle, seed=requery_seed, capture=True, capture_horizon=CAPTURE_HORIZON,
        )
        requery_cont.note_boundary_step(decision_step)
        requery_snapshot = policy_state_snapshot(bundle)
        branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
        observation = restore_boundary(branch, snapshot)
        requery_started = time.perf_counter()
        requery_row: dict[str, Any] | None = None
        try:
            requery_cont.reset()
            first_action, requery_event = capture_inference_event(
                bundle, observation, task=instruction,
                boundary_step=decision_step, generation_seed=requery_seed,
                horizon=CAPTURE_HORIZON,
            )
            if candidate_capture_dir is not None:
                from rase.vnext.candidate_capture import array_sha256

                candidate_chunks["requery.source"] = CaptureChunk(
                    capability="executable", chunk_origin="forced_inference",
                    actions=requery_event.env_chunk,
                    full_env_chunk=requery_event.env_chunk,
                    native_chunk_sha256=array_sha256(requery_event.native_chunk),
                    inference_event_id=requery_event.inference_event_id,
                    queue_cursor_at_boundary=0,
                    candidate_generation_seed=requery_seed,
                    boundary_action=first_action,
                    mask_reason=None,
                )
            executed_first_hashes["requery.source"] = action_hash(first_action)
            result = rollout_policy(
                branch, requery_cont, observation=observation, instruction=instruction,
                first_action=first_action,
            )
            result["branch_wall_s"] = time.perf_counter() - requery_started
            requery_row = base_row(requery_job, prefix)
            requery_row.update(result)
            requery_row.update({
                "available": True, "mask_reason": None,
                "capability_status": "executable", "chunk_origin": "forced_inference",
                "inference_event_id": requery_event.inference_event_id,
                "queue_cursor_at_boundary": 0,
                "intervention_query_count": 1, "fallback_steps": 0,
                "requery_first_action_sha256": action_hash(first_action),
            })
        except Exception as exc:
            requery_row = base_row(requery_job, prefix)
            requery_row.update({
                "available": True, "mask_reason": None, "success": False,
                "capability_status": "execution_error", "chunk_origin": "forced_inference",
                "post_decision_env_steps": 0, "stop_reason": "policy_inference_error",
                "policy_inference_error": {
                    "exception_type": type(exc).__name__, "exception": str(exc)[:1000],
                },
                "branch_wall_s": time.perf_counter() - requery_started,
                "policy_metrics": requery_cont.metrics(),
                "intervention_query_count": 1, "fallback_steps": 0,
            })
        finally:
            policy_state_restore(bundle, requery_snapshot)
        rows_by_operator["requery.source"] = requery_row
        branch.close()

        # Resample creates two native candidates and uses the preregistered
        # minimum first-action L2 verifier. The selected candidate is regenerated
        # under the same seed because LeRobot policy queues are mutable.
        # === resample.source: two native candidates (candidate.0 / candidate.1),
        # === each captured under its own frozen operator seed via isolated
        # === forced inference.  Capability is data-driven: if both first
        # === actions are bitwise identical there is no native diversity and the
        # === slots are recorded incapable (never ordinary failure).  The
        # === preregistered verifier (min first-action L2, then candidate id)
        # === selects which candidate is executed when diversity exists.
        if legacy_resample:
            resample_candidates = [by_operator["resample.source"]]
        else:
            resample_candidates = [
                by_operator["resample.source/candidate.0"],
                by_operator["resample.source/candidate.1"],
            ]
        if any(str(job.get("available_by_contract", True)) != "True" and not job.get("available_by_contract", True) for job in resample_candidates):
            for job in resample_candidates:
                operator = str(job["operator_id"])
                row = base_row(job, prefix)
                row.update({
                    "available": False,
                    "mask_reason": str(job.get("contract_mask_reason") or "contract_masked"),
                    "capability_status": "incapable_missing",
                    "chunk_origin": "contract_mask",
                    "execution_status": "not_executed",
                    "success": None, "harm": None, "query_cost": None,
                    "fallback_cost": None, "latency_cost": None, "utility": None,
                    "intervention_query_count": 0, "fallback_steps": 0,
                })
                if candidate_capture_dir is not None:
                    candidate_chunks[operator] = CaptureChunk(
                        capability="incapable_missing", chunk_origin="contract_mask",
                        actions=None, mask_reason=str(
                            job.get("contract_mask_reason") or "contract_masked",
                        ),
                    )
                rows_by_operator[operator] = row
        else:
            branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
            observation = restore_boundary(branch, snapshot)
            resample_started = time.perf_counter()
            candidates: list[dict[str, Any]] = []
            resample_error: Exception | None = None
            if legacy_resample:
                base_seed = int(resample_candidates[0]["seed_ledger"]["operator_seed"])
                candidate_specs = [
                    (resample_candidates[0], "candidate.0",
                     stable_seed("rase-vnext-resample-v1", base_seed, "candidate.0")),
                    (resample_candidates[0], "candidate.1",
                     stable_seed("rase-vnext-resample-v1", base_seed, "candidate.1")),
                ]
            else:
                candidate_specs = [
                    (resample_candidates[0], "candidate.0",
                     int(resample_candidates[0]["seed_ledger"]["operator_seed"])),
                    (resample_candidates[1], "candidate.1",
                     int(resample_candidates[1]["seed_ledger"]["operator_seed"])),
                ]
            for job, candidate_id, candidate_seed in candidate_specs:
                try:
                    action, event = isolated_force_inference(
                        bundle, observation, task=instruction,
                        boundary_step=decision_step,
                        generation_seed=int(candidate_seed),
                        horizon=CAPTURE_HORIZON,
                    )
                except Exception as exc:
                    resample_error = exc
                    break
                candidates.append({
                    "candidate_id": candidate_id,
                    "job": job,
                    "seed": int(candidate_seed),
                    "event": event,
                    "first_action": action,
                    "first_action_l2": float(np.linalg.norm(action)),
                    "first_action_sha256": action_hash(action),
                })
            if resample_error is None and len(candidates) == len(candidate_specs):
                distinct_first = {candidate["first_action_sha256"] for candidate in candidates}
                if len(distinct_first) < len(candidates):
                    # No native resample diversity: deterministic capability
                    # record for every slot; nothing is executed or failed.
                    emitted: set[str] = set()
                    for candidate in candidates:
                        job = candidate["job"]
                        event = candidate["event"]
                        operator = str(job["operator_id"])
                        if operator in emitted:
                            continue
                        emitted.add(operator)
                        row = base_row(job, prefix)
                        row.update({
                            "available": False,
                            "mask_reason": "no_native_resample_diversity",
                            "capability_status": "incapable_missing",
                            "chunk_origin": "forced_inference",
                            "execution_status": "not_executed",
                            "inference_event_id": event.inference_event_id,
                            "queue_cursor_at_boundary": 0,
                            "success": None, "harm": None, "query_cost": None,
                            "fallback_cost": None, "latency_cost": None, "utility": None,
                            "intervention_query_count": 0, "fallback_steps": 0,
                            "candidate_verifier": "minimum_first_action_l2_then_candidate_id",
                        })
                        if candidate_capture_dir is not None:
                            from rase.vnext.candidate_capture import array_sha256

                            candidate_chunks[operator] = CaptureChunk(
                                capability="incapable_missing",
                                chunk_origin="forced_inference",
                                actions=None,
                                full_env_chunk=event.env_chunk,
                                native_chunk_sha256=array_sha256(event.native_chunk),
                                inference_event_id=event.inference_event_id,
                                queue_cursor_at_boundary=0,
                                candidate_generation_seed=int(candidate["seed"]),
                                boundary_action=candidate["first_action"],
                                mask_reason="no_native_resample_diversity",
                            )
                        rows_by_operator[operator] = row
                else:
                    selected = min(
                        candidates,
                        key=lambda item: (item["first_action_l2"], item["candidate_id"]),
                    )
                    regenerated, regen_event = isolated_force_inference(
                        bundle, observation, task=instruction,
                        boundary_step=decision_step,
                        generation_seed=int(selected["seed"]),
                        horizon=CAPTURE_HORIZON,
                    )
                    if action_hash(regenerated) != selected["first_action_sha256"]:
                        raise RuntimeError("resample candidate regeneration was not exact under the frozen seed")
                    resample_cont = InProcessLeRobotContinuation(
                        bundle, seed=int(selected["seed"]),
                        capture=True, capture_horizon=CAPTURE_HORIZON,
                    )
                    resample_cont.note_boundary_step(decision_step)
                    if candidate_capture_dir is not None:
                        from rase.vnext.candidate_capture import array_sha256

                        for candidate in candidates:
                            if legacy_resample and candidate is not selected:
                                continue
                            operator = str(candidate["job"]["operator_id"])
                            event = candidate["event"]
                            candidate_chunks[operator] = CaptureChunk(
                                capability="executable", chunk_origin="forced_inference",
                                actions=event.env_chunk,
                                full_env_chunk=event.env_chunk,
                                native_chunk_sha256=array_sha256(event.native_chunk),
                                inference_event_id=event.inference_event_id,
                                queue_cursor_at_boundary=0,
                                candidate_generation_seed=int(candidate["seed"]),
                                boundary_action=candidate["first_action"],
                                mask_reason=None,
                            )
                    executed_first_hashes[str(selected["job"]["operator_id"])] = action_hash(regenerated)
                    resample_cont.reset()
                    result = rollout_policy(
                        branch, resample_cont, observation=observation, instruction=instruction,
                        first_action=regenerated,
                    )
                    result["branch_wall_s"] = time.perf_counter() - resample_started
                    for candidate in candidates:
                        if legacy_resample and candidate is not selected:
                            # Legacy manifests carry a single resample job: only
                            # the executed (selected) candidate gets a row.
                            continue
                        job = candidate["job"]
                        operator = str(job["operator_id"])
                        executed = candidate is selected
                        row = base_row(job, prefix)
                        row.update({
                            "available": True, "mask_reason": None,
                            "capability_status": "executable",
                            "chunk_origin": "forced_inference",
                            "execution_status": "executed" if executed else "not_selected",
                            "inference_event_id": candidate["event"].inference_event_id,
                            "queue_cursor_at_boundary": 0,
                            "intervention_query_count": 3, "fallback_steps": 0,
                            "candidate_verifier": "minimum_first_action_l2_then_candidate_id",
                            "selected_candidate_id": selected["candidate_id"],
                            "candidates": [
                                {key: value for key, value in cand.items()
                                 if key not in {"first_action", "event", "job"}}
                                for cand in candidates
                            ],
                        })
                        if executed:
                            row.update(result)
                        else:
                            row.update({
                                "success": None, "harm": None, "query_cost": None,
                                "fallback_cost": None, "latency_cost": None,
                                "utility": None, "post_decision_env_steps": None,
                                "stop_reason": "not_selected",
                                "policy_metrics": {},
                            })
                        rows_by_operator[operator] = row
            else:
                for job in resample_candidates:
                    operator = str(job["operator_id"])
                    row = base_row(job, prefix)
                    row.update({
                        "available": True, "mask_reason": None, "success": False,
                        "capability_status": "execution_error",
                        "chunk_origin": "forced_inference",
                        "execution_status": "executed",
                        "post_decision_env_steps": 0, "stop_reason": "policy_inference_error",
                        "policy_inference_error": {
                            "exception_type": type(resample_error).__name__ if resample_error else "unknown",
                            "exception": str(resample_error)[:1000] if resample_error else "",
                        },
                        "branch_wall_s": time.perf_counter() - resample_started,
                        "intervention_query_count": len(candidates) + 1, "fallback_steps": 0,
                        "candidate_verifier": "minimum_first_action_l2_then_candidate_id",
                    })
                    if candidate_capture_dir is not None:
                        candidate_chunks[operator] = CaptureChunk(
                            capability="execution_error", chunk_origin="forced_inference",
                            actions=None,
                            mask_reason=(
                                f"{type(resample_error).__name__}: {resample_error}"
                                if resample_error else "unknown_error"
                            ),
                        )
                    rows_by_operator[operator] = row
            branch.close()

        # Persistent OFT uses only the frozen boundary snapshot and records the
        # number of native chunk queries for explicit compute cost.
        branch = restore_pool_state(pool, exemplar["state_key"], libero_plus_root=libero_plus_root)
        fallback_started = time.perf_counter()
        with preserve_rng_state():
            fallback = persistent_branch(
                branch, snapshot, client, instruction, record_chunk_trace=True,
                return_action_trace=candidate_capture_dir is not None,
            )
        if candidate_capture_dir is not None:
            fallback_full_action_trace = np.asarray(
                fallback.pop("action_trace"), dtype=np.float32,
            )
            if not len(fallback_full_action_trace):
                raise RuntimeError("fallback produced no action for synchronous capture")
            candidate_chunks["fallback.persistent"] = CaptureChunk(
                capability="executable", chunk_origin="executed_trace",
                actions=fallback_full_action_trace[:CAPTURE_HORIZON],
                full_env_chunk=fallback_full_action_trace[:CAPTURE_HORIZON],
                native_chunk_sha256=None,
                inference_event_id=None,
                queue_cursor_at_boundary=0,
                candidate_generation_seed=None,
                boundary_action=fallback_full_action_trace[0],
                mask_reason=None,
            )
            executed_first_hashes["fallback.persistent"] = action_hash(
                fallback_full_action_trace[0],
            )
        row = base_row(by_operator["fallback.persistent"], prefix)
        row.update({
            "available": True, "mask_reason": None,
            "capability_status": "executable", "chunk_origin": "executed_trace",
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
            "available": True, "mask_reason": None,
            "capability_status": "control_only_abort", "chunk_origin": "control_only",
            "success": False,
            "post_decision_env_steps": 0, "stop_reason": "safe_abort",
            "branch_wall_s": 0.0, "intervention_query_count": 0,
            "fallback_steps": 0,
        })
        if candidate_capture_dir is not None:
            candidate_chunks["abort.safe"] = CaptureChunk(
                capability="control_only_abort", chunk_origin="control_only",
                actions=None, mask_reason="safe_abort_control_event",
            )
        rows_by_operator["abort.safe"] = row

        rows = [
            rows_by_operator[operator]
            for operator in (OPERATORS if not legacy_resample else OPERATORS_LEGACY)
        ]
        finalize_group_rows(rows, utility=utility)
        capture: dict[str, Any] | None = None
        if candidate_capture_dir is not None:
            from rase.vnext.candidate_capture import write_candidate_capture
            from rase.vnext.libero import LiberoBenchmarkAdapter

            benchmark = LiberoBenchmarkAdapter(
                vector_env=main.handle.vector_env, forkable=main.forkable,
            )
            canonical = benchmark.observation_to_canonical(
                prefix["observation"], task_text=instruction,
                timestamp_s=float(decision_step) / 10.0,
            )
            capture = write_candidate_capture(
                candidate_capture_dir,
                group_key=group_key(exemplar),
                operator_chunks=candidate_chunks,
                instruction=instruction,
                task_id=str(exemplar["task_id"]),
                suite=str(exemplar["suite"]),
                policy_id=str(exemplar["policy_id"]),
                decision_point_id=str(exemplar["decision_point"]["decision_point_id"]),
                replica=int(exemplar["seed_ledger"]["exact_repeat_replica"]),
                seed_ledger=exemplar["seed_ledger"],
                proprio=canonical.proprio,
                proprio_mask=canonical.proprio_mask,
                images=canonical.images,
                executed_first_action_sha256=executed_first_hashes,
                fallback_full_action_trace=fallback_full_action_trace,
            )
            for row in rows:
                row["candidate_capture_arrays_path"] = capture["arrays_path"]
                row["candidate_capture_arrays_sha256"] = capture["arrays_sha256"]
                row["candidate_capture_metadata_path"] = capture["metadata_path"]
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
    phase = "confirmation" if manifest.get("status") == "frozen_confirmation" else "discovery"
    report = {
        "schema_version": f"rase-vnext-{phase}-collection-report/v1",
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
    parser.add_argument(
        "--candidate-capture-dir", type=Path,
        help="Synchronously persist candidate chunks actually used by each branch.",
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    protocol_path = args.protocol.resolve()
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if manifest.get("status") not in {"frozen_discovery", "frozen_confirmation"}:
        raise SystemExit("manifest is neither frozen_discovery nor frozen_confirmation")
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
            candidate_capture_dir=(
                args.candidate_capture_dir.resolve()
                if args.candidate_capture_dir is not None else None
            ),
        )
        phase = "confirmation" if manifest.get("status") == "frozen_confirmation" else "discovery"
        payload = {
            "schema_version": f"rase-vnext-{phase}-group/v1",
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

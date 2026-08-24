#!/usr/bin/env python3
"""Collect task-isolated local C/R counterfactual data for V6 RASE.

At one late native-chunk decision point, the frozen source VLA supplies two
fully frozen candidates: C is the remaining stale suffix and R is one fresh
re-query.  Both candidates are evaluated from the *same perturbed snapshot*
under the same two deterministic downstream seeds.  The target is therefore
an estimate of A_R = Q(R) - Q(C), not a best-of-K label and not a trajectory
schedule/RL target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_v6_stage0_direct_position import (  # noqa: E402
    apply_position_perturbation,
    array_sha256,
    atomic_json,
    atomic_npz,
    bddl_goal_manipulated_objects,
    choose_perturbation_targets,
    clean_bddl_directory,
    exact_bddl_objects,
    flatten_numeric_observation,
    force_fresh_inference,
    qpos_sha256,
    stable_seed,
    terminal_values,
)


ROOT_SCHEMA = "rase-v6-local-cf-root/v1"
BRANCH_SCHEMA = "rase-v6-local-cf-branch/v1"
PLAN_SCHEMA = "rase-v6-local-cf-plan/v1"


def parse_temperature(value: str | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "default"}:
        return None
    result = float(text)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"temperature must be finite/non-negative or default, got {value!r}")
    return result


def parse_ints(value: str, *, name: str, minimum: int = 0) -> list[int]:
    values = [int(item) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item < minimum for item in values):
        raise ValueError(f"{name} must be distinct integers >= {minimum}")
    return values


def candidate_prefix(event: Any, length: int, label: str) -> np.ndarray:
    chunk = np.asarray(event.env_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < length:
        raise ValueError(f"{label}: expected >=({length},7), got {chunk.shape}")
    return chunk[:length].copy()


def make_root_plan(args: argparse.Namespace) -> dict[str, Any]:
    tasks = parse_ints(args.tasks, name="--tasks", minimum=1)
    cursors = parse_ints(args.cursors, name="--cursors", minimum=1)
    pre_chunks = parse_ints(args.pre_decision_chunks, name="--pre-decision-chunks", minimum=1)
    if any(cursor >= args.native_horizon for cursor in cursors):
        raise ValueError("each cursor must be inside the native action chunk")
    if args.roots_per_task < 1 or args.downstream_repeats < 2:
        raise ValueError("need positive roots and at least two downstream repeats")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.plan_label):
        raise ValueError("--plan-label must contain only letters, numbers, '_' and '-'")
    roots: list[dict[str, Any]] = []
    for task_offset, task_number in enumerate(tasks):
        for replicate in range(args.roots_per_task):
            phase = task_offset + replicate
            cursor = cursors[phase % len(cursors)]
            prefix_count = pre_chunks[(task_offset + 2 * replicate) % len(pre_chunks)]
            root_id = f"{args.plan_label}_t{task_number:02d}_r{replicate:03d}_pc{prefix_count}_c{cursor}"
            roots.append({
                "root_id": root_id,
                "seed_key": root_id,
                "task_id": f"{args.suite}_{task_number:06d}",
                "task_number": task_number,
                "suite": args.suite,
                "replicate": replicate,
                "init_state_id": int((task_offset + 3 * replicate) % args.init_states),
                "environment_seed": stable_seed("v6-local-cf-env", args.seed, root_id),
                "perturbation_seed": stable_seed("v6-local-cf-pos", args.seed, root_id),
                "source_generation_seed": stable_seed("v6-local-cf-source", args.seed, root_id),
                "refresh_generation_seed": stable_seed("v6-local-cf-refresh", args.seed, root_id),
                "pre_decision_generation_seeds": [
                    stable_seed("v6-local-cf-pre", args.seed, root_id, i)
                    for i in range(prefix_count)
                ],
                "downstream_seeds": [
                    stable_seed("v6-local-cf-mu", args.seed, root_id, i)
                    for i in range(args.downstream_repeats)
                ],
                "cursor": cursor,
                "pre_decision_chunks": prefix_count,
                "native_chunk_horizon": int(args.native_horizon),
                "perturb_dim": "position",
                "perturb_level": float(args.perturb_level),
                "position_target_mode": args.position_target_mode,
                "selection_outcomes_used": False,
            })
    return {
        "schema_version": PLAN_SCHEMA,
        "selection_outcomes_used": False,
        "selection": {
            "split": args.split_name, "suite": args.suite, "tasks": tasks,
            "roots_per_task": int(args.roots_per_task), "cursors": cursors,
            "pre_decision_chunks": pre_chunks, "native_chunk_horizon": int(args.native_horizon),
            "downstream_repeats": int(args.downstream_repeats),
            "perturb_level": float(args.perturb_level),
            "position_target_mode": args.position_target_mode, "seed": int(args.seed),
            "design": (
                "uniform task x replicate roots with cyclic late-phase strata; no terminal "
                "outcome is used for sampling; C and one R candidate are frozen before all "
                "same-root terminal rollouts"
            ),
        },
        "roots": roots,
    }


def run_branch(
    *, handle: Any, forkable: Any, snapshot: Any, expected_qpos: str, task: str,
    candidate: np.ndarray, bundle: Mapping[str, Any], downstream_seed: int,
    downstream_temperature: float | None,
) -> dict[str, Any]:
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    forkable.restore(snapshot, check_task_fingerprint=True)
    single = handle.vector_env.envs[0]
    if qpos_sha256(single) != expected_qpos:
        raise AssertionError("same-root restore qpos checksum drift")
    observation = observation_from_libero_env(single)
    horizon = int(getattr(single, "_max_episode_steps", 600))
    candidate_steps = 0
    continuation_steps = 0
    success = False
    started = time.perf_counter()
    for action in candidate:
        if current_timestep(handle.control_env) >= horizon:
            break
        observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        candidate_steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            return {"success": success, "candidate_steps": candidate_steps, "continuation_steps": 0,
                    "stop_reason": "success" if success else "terminal_failure",
                    "elapsed_s": round(time.perf_counter() - started, 6)}
    continuation = InProcessSmolVLAContinuation(
        bundle, temperature=downstream_temperature, seed=int(downstream_seed),
    )
    continuation.reset()
    stop_reason = "horizon"
    while current_timestep(handle.control_env) < horizon:
        action = continuation.act(observation, task=task)
        observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        continuation_steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            stop_reason = "success" if success else "terminal_failure"
            break
    return {"success": bool(success), "candidate_steps": candidate_steps,
            "continuation_steps": continuation_steps, "stop_reason": stop_reason,
            "elapsed_s": round(time.perf_counter() - started, 6)}


def collect_root(
    root: Mapping[str, Any], *, bundle: Mapping[str, Any], args: argparse.Namespace,
    bddl_dir: Path, artifact_dir: Path,
) -> dict[str, Any]:
    from rase.backends.libero_clean import clean_task_name
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.envs.forkable_env import ForkableEnv

    handle = make_libero_env_for_task(
        str(root["task_id"]), init_state_id=int(root["init_state_id"]),
        seed=int(root["environment_seed"]), observation_height=args.observation_height,
        observation_width=args.observation_width, libero_clean_root=str(args.libero_clean_root),
        libero_flavor="clean",
    )
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        task_name = clean_task_name(str(root["suite"]), int(root["task_number"]))
        interest, bddl_path = exact_bddl_objects(bddl_dir, task_name)
        goal_targets = bddl_goal_manipulated_objects(bddl_path)
        targets = choose_perturbation_targets(goal_targets, mode=str(root["position_target_mode"]))
        forkable = ForkableEnv(handle.control_env)
        observation = observation_from_libero_env(single)
        pre_events: list[dict[str, Any]] = []
        for chunk_index, generation_seed in enumerate(root["pre_decision_generation_seeds"]):
            event = force_fresh_inference(
                bundle, observation, task=task, boundary_step=chunk_index * int(root["native_chunk_horizon"]),
                generation_seed=int(generation_seed), horizon=int(root["native_chunk_horizon"]),
                temperature=args.source_temperature,
            )
            chunk = candidate_prefix(event, int(root["native_chunk_horizon"]), "pre-decision chunk")
            pre_events.append({"index": chunk_index, "seed": int(generation_seed), "chunk_sha256": array_sha256(chunk)})
            for action in chunk:
                observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                terminal, success = terminal_values(term, trunc, info)
                if terminal:
                    return {"schema_version": ROOT_SCHEMA, "status": "unavailable", "root": dict(root),
                            "reason": "terminal_before_predecision", "terminal_success": success, "branches": []}
        old_source_qpos = qpos_sha256(single)
        old_event = force_fresh_inference(
            bundle, observation, task=task,
            boundary_step=int(root["pre_decision_chunks"]) * int(root["native_chunk_horizon"]),
            generation_seed=int(root["source_generation_seed"]), horizon=int(root["native_chunk_horizon"]),
            temperature=args.source_temperature,
        )
        old_chunk = candidate_prefix(old_event, int(root["native_chunk_horizon"]), "old chunk")
        for action in old_chunk[:int(root["cursor"])]:
            observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                return {"schema_version": ROOT_SCHEMA, "status": "unavailable", "root": dict(root),
                        "reason": "terminal_before_decision", "terminal_success": success, "branches": []}
        pre_perturb_qpos = qpos_sha256(single)
        moved = apply_position_perturbation(
            single, level=float(root["perturb_level"]), seed=int(root["perturbation_seed"]), targets=targets,
        )
        boundary_qpos = qpos_sha256(single)
        if not moved or boundary_qpos == pre_perturb_qpos:
            raise AssertionError("valid position perturbation was not applied")
        snapshot = forkable.snapshot()
        boundary_observation = observation_from_libero_env(single)
        continue_chunk = old_chunk[int(root["cursor"]):].copy()
        refresh_event = force_fresh_inference(
            bundle, boundary_observation, task=task, boundary_step=int(root["cursor"]),
            generation_seed=int(root["refresh_generation_seed"]), horizon=int(root["native_chunk_horizon"]),
            temperature=args.source_temperature,
        )
        refresh_chunk = candidate_prefix(refresh_event, len(continue_chunk), "refresh chunk")
        if int(root["refresh_generation_seed"]) == int(root["source_generation_seed"]):
            raise AssertionError("refresh seed must differ from old source seed")
        candidates = {
            "C": (continue_chunk, None, array_sha256(old_chunk)),
            "R": (refresh_chunk, int(root["refresh_generation_seed"]), array_sha256(refresh_event.env_chunk)),
        }
        branches: list[dict[str, Any]] = []
        for downstream_rep, downstream_seed in enumerate(root["downstream_seeds"]):
            for kind, (candidate, candidate_seed, full_sha) in candidates.items():
                result = run_branch(
                    handle=handle, forkable=forkable, snapshot=snapshot, expected_qpos=boundary_qpos,
                    task=task, candidate=candidate, bundle=bundle, downstream_seed=int(downstream_seed),
                    downstream_temperature=args.downstream_temperature,
                )
                branches.append({
                    "schema_version": BRANCH_SCHEMA, "status": "complete", "root_id": root["root_id"],
                    "task_id": root["task_id"], "suite": root["suite"], "candidate_kind": kind,
                    "downstream_rep": downstream_rep, "downstream_seed": int(downstream_seed),
                    "candidate_generation_seed": candidate_seed, "candidate_chunk_sha256": array_sha256(candidate),
                    "candidate_full_chunk_sha256": full_sha, "candidate_steps": int(candidate.shape[0]),
                    "root_snapshot_qpos_sha256": boundary_qpos, "pre_perturb_qpos_sha256": pre_perturb_qpos,
                    "old_chunk_source_qpos_sha256": old_source_qpos, "cursor": int(root["cursor"]),
                    "pre_decision_chunks": int(root["pre_decision_chunks"]), "perturb_level": float(root["perturb_level"]),
                    "source_temperature": args.source_temperature, "downstream_temperature": args.downstream_temperature,
                    "success": bool(result["success"]), "rollout": result,
                })
        if {row["downstream_seed"] for row in branches if row["candidate_kind"] == "C"} != set(root["downstream_seeds"]):
            raise AssertionError("C lacks a planned downstream seed")
        if {row["downstream_seed"] for row in branches if row["candidate_kind"] == "R"} != set(root["downstream_seeds"]):
            raise AssertionError("R lacks a matched downstream seed")
        artifact_path = artifact_dir / f"{root['root_id']}.npz"
        arrays = {"continue_candidate": continue_chunk, "refresh_candidate": refresh_chunk, "old_chunk": old_chunk}
        arrays.update(flatten_numeric_observation(boundary_observation))
        atomic_npz(artifact_path, arrays)
        return {
            "schema_version": ROOT_SCHEMA, "status": "complete", "root": dict(root), "instruction": task,
            "bddl_source": "clean_execution_assets", "bddl_path": str(bddl_path),
            "targets_from_goal": goal_targets, "targets_from_obj_of_interest": interest,
            "perturbation_targets": targets, "moved_objects": moved, "predecision_events": pre_events,
            "boundary": {"qpos_sha256": boundary_qpos, "pre_perturb_qpos_sha256": pre_perturb_qpos,
                         "old_chunk_source_qpos_sha256": old_source_qpos, "artifact": str(artifact_path)},
            "branches": branches,
        }
    finally:
        handle.close()


def merge_jsonl(roots_dir: Path, output: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(roots_dir.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("branches") or [])
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--roots-per-task", type=int, required=True)
    parser.add_argument("--init-states", type=int, default=10)
    parser.add_argument("--cursors", default="3,5,8")
    parser.add_argument("--pre-decision-chunks", default="1,2,3")
    parser.add_argument("--native-horizon", type=int, default=10)
    parser.add_argument("--downstream-repeats", type=int, default=2)
    parser.add_argument("--perturb-level", type=float, default=0.04)
    parser.add_argument("--position-target-mode", choices=("all_goal_subjects", "first_goal_subject"), default="first_goal_subject")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--plan-label", required=True)
    parser.add_argument("--source-temperature", type=parse_temperature, default=0.5)
    parser.add_argument("--downstream-temperature", type=parse_temperature, default=0.5)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--libero-clean-root", type=Path, default=Path("/root/autodl-tmp/src/LIBERO"))
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        parser.error(f"--fresh-run refuses existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = make_root_plan(args)
    plan_path = output / "root_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing root plan differs")
    else:
        atomic_json(plan_path, plan)
    roots_dir = output / "roots"; roots_dir.mkdir(exist_ok=True)
    artifact_dir = output / "decision_artifacts"; artifact_dir.mkdir(exist_ok=True)
    from libero.libero.utils import get_libero_path, set_libero_path
    from rase.collect.forked_rollout import load_lerobot_policy_bundle
    previous_bddl = get_libero_path("bddl_files")
    bddl_dir = clean_bddl_directory(args.libero_clean_root, args.suite)
    if not args.policy.is_absolute(): args.policy = ROOT / args.policy
    if not args.tokenizer.is_absolute(): args.tokenizer = ROOT / args.tokenizer
    bundle = load_lerobot_policy_bundle(args.policy, device="cuda", num_steps=args.native_horizon,
        n_action_steps=args.native_horizon, tokenizer_path=args.tokenizer,
        observation_height=args.observation_height, observation_width=args.observation_width)
    atomic_json(output / "run_manifest.json", {
        "schema_version": "rase-v6-local-cf-run/v1", "root_plan": str(plan_path),
        "selection_outcomes_used": False, "source_policy": "one frozen SmolVLA",
        "protocol": "C/R candidates frozen before same-root paired downstream seed rollouts",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    })
    errors = 0; started = time.perf_counter()
    try:
        # Interleave pre-registered tasks by replicate.  This changes only
        # collection order, never the root plan/seed set or an outcome-driven
        # sampling decision; it lets an early partial audit reveal whether
        # labels exist across tasks instead of spending the first 30 roots on
        # one potentially unrecoverable task.
        execution_roots = sorted(
            plan["roots"],
            key=lambda value: (int(value["replicate"]), int(value["task_number"])),
        )
        for index, root in enumerate(execution_roots):
            target = roots_dir / f"{root['root_id']}.json"
            if target.exists():
                print(f"LocalCF skip {index + 1}/{len(plan['roots'])} {root['root_id']}", flush=True); continue
            try:
                record = collect_root(root, bundle=bundle, args=args, bddl_dir=bddl_dir, artifact_dir=artifact_dir)
            except Exception as exc:
                errors += 1
                record = {"schema_version": ROOT_SCHEMA, "status": "error", "root": root,
                          "error_type": type(exc).__name__, "error": str(exc)[:2000], "branches": []}
            atomic_json(target, record)
            print(f"LocalCF {index + 1}/{len(execution_roots)} {root['root_id']} status={record['status']} elapsed_min={(time.perf_counter()-started)/60:.1f}", flush=True)
    finally:
        set_libero_path(previous_bddl)
    n_branches = merge_jsonl(roots_dir, output / "local_cf_records.jsonl")
    summary = {"n_planned_roots": len(plan["roots"]), "n_branch_rows": n_branches,
               "uncaught_root_errors": errors, "elapsed_s": round(time.perf_counter()-started, 3)}
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if n_branches else 2


if __name__ == "__main__":
    raise SystemExit(main())

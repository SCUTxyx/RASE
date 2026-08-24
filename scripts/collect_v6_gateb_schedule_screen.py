#!/usr/bin/env python3
"""Gate-B screen: fixed SmolVLA refresh schedules under one dynamic shift.

This is deliberately *not* a local candidate-selection experiment.  A root
is one seeded LIBERO episode with a pre-registered external 4 cm xy shift at
an absolute environment timestep.  From the same initial root, each arm runs
the frozen SmolVLA with a fixed re-planning period E.  The environment change
is applied after exactly ``perturb_at_step`` actions for every arm, regardless
of whether that arm happens to re-plan at that point.

Thus differences between E={2,4,6,8,10} are trajectory-level consequences of
their refresh schedules.  There is no selector, learned model, outcome-based
root selection, or test-time best-of-K operation in this collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already-audited dynamic-position implementation.  In particular,
# this avoids reintroducing the historic BDDL/body-name no-op perturbation bug.
from collect_v6_stage0_direct_position import (  # noqa: E402
    apply_position_perturbation,
    array_sha256,
    atomic_json,
    bddl_goal_manipulated_objects,
    choose_perturbation_targets,
    clean_bddl_directory,
    exact_bddl_objects,
    force_fresh_inference,
    qpos_sha256,
    stable_seed,
    terminal_values,
)


ROOT_SCHEMA = "rase-v6-gateb-schedule-root/v1"
PLAN_SCHEMA = "rase-v6-gateb-schedule-plan/v1"


def parse_temperature(value: str | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "default"}:
        return None
    result = float(text)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"temperature must be non-negative finite or default, got {value!r}")
    return result


def parse_int_list(value: str, *, name: str, positive: bool = True) -> list[int]:
    values = [int(item) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty list of distinct integers")
    if positive and any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def candidate_prefix(event: Any, length: int, *, label: str) -> np.ndarray:
    chunk = np.asarray(event.env_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < length:
        raise ValueError(f"{label}: invalid native action chunk {chunk.shape}; need at least ({length},7)")
    return chunk[:length].copy()


def make_root_plan(args: argparse.Namespace) -> dict[str, Any]:
    schedules = parse_int_list(args.schedules, name="--schedules")
    triggers = parse_int_list(args.perturb_at_steps, name="--perturb-at-steps")
    if any(value > args.native_horizon for value in schedules):
        raise ValueError("every schedule must be <= --native-horizon; this screen never executes a stale tail beyond one native chunk")
    tasks = parse_int_list(args.tasks, name="--tasks")
    if args.replicates_per_task < 1:
        raise ValueError("--replicates-per-task must be positive")
    if args.init_states < 1:
        raise ValueError("--init-states must be positive")
    if any(value >= args.max_episode_steps for value in triggers):
        raise ValueError("all perturbation steps must fall before --max-episode-steps")
    if not args.plan_label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--plan-label must contain only letters, numbers, '_' and '-'")

    roots: list[dict[str, Any]] = []
    for task_offset, task_number in enumerate(tasks):
        for replicate in range(args.replicates_per_task):
            root_id = f"{args.plan_label}_t{task_number:02d}_r{replicate:02d}"
            # The perturbation time is fixed before rollout and never depends on
            # policy outputs or outcomes.  13/25/38 reproduce the late phases
            # (10+3, 20+5, 30+8) used by the passed 4 cm Gate-A formal; 50 gives
            # a fourth, still pre-registered late-phase stratum.
            trigger = triggers[(task_offset + replicate) % len(triggers)]
            roots.append({
                "root_id": root_id,
                "seed_key": root_id,
                "task_id": f"{args.suite}_{task_number:06d}",
                "suite": args.suite,
                "task_number": task_number,
                "replicate": replicate,
                "init_state_id": int((task_offset + replicate) % args.init_states),
                "environment_seed": stable_seed("v6-gateb-env", args.seed, root_id),
                "perturbation_seed": stable_seed("v6-gateb-position", args.seed, root_id),
                "perturb_at_step": int(trigger),
                "perturb_dim": "position",
                "perturb_level": float(args.perturb_level),
                "position_target_mode": args.position_target_mode,
                "selection_outcomes_used": False,
            })
    return {
        "schema_version": PLAN_SCHEMA,
        "selection_outcomes_used": False,
        "selection": {
            "suite": args.suite,
            "tasks": tasks,
            "replicates_per_task": int(args.replicates_per_task),
            "schedules": schedules,
            "native_chunk_horizon": int(args.native_horizon),
            "max_episode_steps": int(args.max_episode_steps),
            "perturbation_at_absolute_env_steps": triggers,
            "position_perturbation_level": float(args.perturb_level),
            "position_target_mode": args.position_target_mode,
            "seed": int(args.seed),
            "design": (
                "every task x replicate is evaluated under every fixed refresh schedule; "
                "the external perturbation occurs after a pre-registered absolute "
                "environment step, independent of policy schedule and outcome"
            ),
        },
        "roots": roots,
    }


def run_schedule(
    *, root: Mapping[str, Any], schedule: int, bundle: Mapping[str, Any],
    args: argparse.Namespace, bddl_dir: Path,
) -> dict[str, Any]:
    """Run one complete schedule from the root initial state.

    At each multiple of E we sample one fresh native chunk and execute its
    first E actions.  If the external shift falls in the middle of the chunk,
    the remaining planned suffix is intentionally stale until the next fixed
    re-plan boundary; that is the schedule intervention being measured.
    """
    from rase.backends.libero_clean import clean_task_name
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    handle = make_libero_env_for_task(
        str(root["task_id"]), init_state_id=int(root["init_state_id"]),
        seed=int(root["environment_seed"]), observation_height=args.observation_height,
        observation_width=args.observation_width, libero_clean_root=str(args.libero_clean_root),
        libero_flavor="clean",
    )
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        catalog_task_name = clean_task_name(str(root["suite"]), int(root["task_number"]))
        interest_targets, bddl_path = exact_bddl_objects(bddl_dir, catalog_task_name)
        goal_targets = bddl_goal_manipulated_objects(bddl_path)
        targets = choose_perturbation_targets(goal_targets, mode=str(root["position_target_mode"]))
        initial_qpos_sha = qpos_sha256(single)
        observation = observation_from_libero_env(single)
        perturbed = False
        perturbation: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []
        success = False
        stop_reason = "horizon"
        started = time.perf_counter()

        while current_timestep(handle.control_env) < int(args.max_episode_steps):
            boundary_step = int(current_timestep(handle.control_env))
            generation_seed = stable_seed(
                "v6-gateb-replan", args.seed, root["seed_key"], "absolute_step", boundary_step,
            )
            event = force_fresh_inference(
                bundle, observation, task=task, boundary_step=boundary_step,
                generation_seed=generation_seed, horizon=int(args.native_horizon),
                temperature=args.source_temperature,
            )
            remaining = int(args.max_episode_steps) - boundary_step
            execute_n = min(int(schedule), remaining)
            actions = candidate_prefix(event, execute_n, label=f"schedule={schedule}, step={boundary_step}")
            event_record: dict[str, Any] = {
                "boundary_step": boundary_step,
                "generation_seed": int(generation_seed),
                "inference_event_id": str(event.inference_event_id),
                "full_chunk_sha256": array_sha256(np.asarray(event.env_chunk, dtype=np.float32)),
                "executed_prefix_sha256": array_sha256(actions),
                "executed_steps": int(execute_n),
            }
            for action_index, action in enumerate(actions):
                observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                now_step = int(current_timestep(handle.control_env))
                terminal, success = terminal_values(term, trunc, info)
                # A terminal episode cannot receive a valid post-step
                # intervention.  Treat it as unavailable rather than mutating
                # an environment which has already declared an outcome.
                if not terminal and not perturbed and now_step == int(root["perturb_at_step"]):
                    before_qpos_sha = qpos_sha256(single)
                    moved = apply_position_perturbation(
                        single, level=float(root["perturb_level"]),
                        seed=int(root["perturbation_seed"]), targets=targets,
                    )
                    after_qpos_sha = qpos_sha256(single)
                    if not moved:
                        raise RuntimeError("no_target_free_joint_moved")
                    if after_qpos_sha == before_qpos_sha:
                        raise AssertionError("position perturbation did not change qpos")
                    perturbed = True
                    perturbation = {
                        "applied_after_env_step": now_step,
                        "within_event_action_index": int(action_index),
                        "pre_qpos_sha256": before_qpos_sha,
                        "post_qpos_sha256": after_qpos_sha,
                        "moved_objects": moved,
                    }
                if terminal:
                    stop_reason = "success" if success else "terminal_failure"
                    break
            events.append(event_record)
            if stop_reason != "horizon":
                break

        if not perturbed:
            raise RuntimeError(
                f"episode ended before the pre-registered perturbation step {root['perturb_at_step']}"
            )
        return {
            "schedule": int(schedule), "status": "complete", "success": bool(success),
            "stop_reason": stop_reason,
            "initial_qpos_sha256": initial_qpos_sha,
            "final_qpos_sha256": qpos_sha256(single),
            "n_inference_events": len(events), "inference_events": events,
            "perturbation": perturbation,
            "bddl_source": "clean_execution_assets", "bddl_path": str(bddl_path),
            "targets_from_goal": goal_targets, "targets_from_obj_of_interest": interest_targets,
            "perturbation_targets": targets,
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
    finally:
        handle.close()


def merge_jsonl(roots_dir: Path, output: Path) -> int:
    rows: list[Mapping[str, Any]] = []
    for path in sorted(roots_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for schedule in record.get("schedules") or []:
            if isinstance(schedule, Mapping):
                rows.append({"root": record.get("root"), "schedule_result": schedule})
    atomic_jsonl(output, rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--replicates-per-task", type=int, default=4)
    parser.add_argument("--init-states", type=int, default=10)
    parser.add_argument("--schedules", default="2,4,6,8,10")
    parser.add_argument("--perturb-at-steps", default="13,25,38,50")
    parser.add_argument("--native-horizon", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--perturb-level", type=float, default=0.04)
    parser.add_argument("--position-target-mode", choices=("all_goal_subjects", "first_goal_subject"), default="first_goal_subject")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--plan-label", default="v6_gateb_schedule_screen_v1")
    parser.add_argument("--source-temperature", type=parse_temperature, default=0.5)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--libero-clean-root", type=Path, default=Path("/root/autodl-tmp/src/LIBERO"))
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        parser.error(f"--fresh-run refuses existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    roots_dir = output / "roots"
    roots_dir.mkdir(exist_ok=True)
    plan = make_root_plan(args)
    plan_path = output / "root_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing root plan differs; use a new output directory")
    else:
        atomic_json(plan_path, plan)

    from libero.libero.utils import get_libero_path, set_libero_path
    from rase.collect.forked_rollout import load_lerobot_policy_bundle

    previous_bddl = get_libero_path("bddl_files")
    bddl_dir = clean_bddl_directory(args.libero_clean_root, str(args.suite))
    if not args.policy.is_absolute():
        args.policy = ROOT / args.policy
    if not args.tokenizer.is_absolute():
        args.tokenizer = ROOT / args.tokenizer
    bundle = load_lerobot_policy_bundle(
        args.policy, device="cuda", num_steps=args.native_horizon,
        n_action_steps=args.native_horizon, tokenizer_path=args.tokenizer,
        observation_height=args.observation_height, observation_width=args.observation_width,
    )
    atomic_json(output / "run_manifest.json", {
        "schema_version": "rase-v6-gateb-schedule-run/v1",
        "root_plan": str(plan_path), "selection_outcomes_used": False,
        "source_policy": "one frozen SmolVLA", "bddl_source": "clean_execution_assets",
        "candidate_protocol": (
            "each arm refreshes the same frozen source at fixed period E; the external "
            "position shift is applied after a fixed absolute environment step, independent "
            "of arm, source output, and outcome"
        ),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    })
    started = time.perf_counter()
    errors = 0
    schedules = parse_int_list(args.schedules, name="--schedules")
    try:
        for index, root in enumerate(plan["roots"]):
            path = roots_dir / f"{root['root_id']}.json"
            if path.exists():
                print(f"GateB skip {index + 1}/{len(plan['roots'])} {root['root_id']}", flush=True)
                continue
            results: list[dict[str, Any]] = []
            root_error: str | None = None
            for schedule in schedules:
                try:
                    results.append(run_schedule(root=root, schedule=schedule, bundle=bundle, args=args, bddl_dir=bddl_dir))
                except Exception as exc:  # Persist a root-level audit trail and continue other roots.
                    errors += 1
                    root_error = f"schedule={schedule}: {type(exc).__name__}: {str(exc)[:1500]}"
                    results.append({"schedule": int(schedule), "status": "error", "error": root_error})
                    break
            status = "complete" if root_error is None and len(results) == len(schedules) else "error"
            atomic_json(path, {
                "schema_version": ROOT_SCHEMA, "status": status, "root": root,
                "schedules": results, "error": root_error,
            })
            elapsed_min = (time.perf_counter() - started) / 60.0
            print(f"GateB {index + 1}/{len(plan['roots'])} {root['root_id']} status={status} elapsed_min={elapsed_min:.1f}", flush=True)
    finally:
        set_libero_path(previous_bddl)
    n_rows = merge_jsonl(roots_dir, output / "schedule_records.jsonl")
    summary = {
        "n_planned_roots": len(plan["roots"]), "n_schedule_rows": n_rows,
        "uncaught_schedule_errors": errors, "elapsed_s": round(time.perf_counter() - started, 3),
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if n_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

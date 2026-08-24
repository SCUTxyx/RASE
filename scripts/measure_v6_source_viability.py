#!/usr/bin/env python3
"""Exact-asset source-viability calibration for the V6 Stage-0 domain.

This is deliberately *not* a selector experiment.  It measures whether the
frozen source itself has a non-degenerate success rate after a position shift
that moves only BDDL goal-subject objects.  It fixes the two pre-V6 baseline
ambiguities: BDDL assets match the clean execution environment, and the
SmolVLA sampling temperature is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from collect_v6_stage0_direct_position import (  # noqa: E402
    apply_position_perturbation,
    atomic_json,
    bddl_goal_manipulated_objects,
    choose_perturbation_targets,
    clean_bddl_directory,
    exact_bddl_objects,
    stable_seed,
    terminal_values,
)


SCHEMA = "rase-v6-source-viability/v1"


def parse_temperature(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "default"}:
        return None
    result = float(text)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"temperature must be non-negative finite or 'none', got {value!r}")
    return result


def temperature_label(value: float | None) -> str:
    return "default" if value is None else f"t{value:g}".replace(".", "p")


def make_plan(args: argparse.Namespace, *, temperature: float | None) -> dict[str, Any]:
    tasks = [int(item) for item in args.tasks.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks is empty")
    roots: list[dict[str, Any]] = []
    for task_number in tasks:
        for episode in range(args.episodes_per_task):
            root_id = f"v6cal_{temperature_label(temperature)}_t{task_number:02d}_e{episode:02d}"
            roots.append({
                "root_id": root_id,
                "task_id": f"{args.suite}_{task_number:06d}",
                "task_number": task_number,
                "init_state_id": episode % args.init_states,
                # Keep the physical state and perturbation matched across the
                # separately launched temperature conditions.
                "environment_seed": stable_seed("v6cal-env", args.seed, task_number, episode),
                "perturbation_seed": stable_seed("v6cal-pos", args.seed, task_number, episode),
                "generation_seed": stable_seed(
                    "v6cal-generation", args.seed, temperature_label(temperature), task_number, episode
                ),
                "selection_outcomes_used": False,
            })
    return {
        "schema_version": SCHEMA,
        "selection_outcomes_used": False,
        "suite": args.suite,
        "position_level": args.position_level,
        "position_target_mode": args.position_target_mode,
        "temperature": temperature,
        "temperature_label": temperature_label(temperature),
        "roots": roots,
    }


def run_root(
    root: Mapping[str, Any], *, args: argparse.Namespace, bundle: Mapping[str, Any],
    bddl_dir: Path, temperature: float | None,
) -> dict[str, Any]:
    from rase.backends.libero_clean import clean_task_name
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action
    from rase.collect.pool_candidates import observation_from_libero_env

    catalog_task_name = clean_task_name(args.suite, int(root["task_number"]))
    _interest, bddl_path = exact_bddl_objects(bddl_dir, catalog_task_name)
    all_goal_targets = bddl_goal_manipulated_objects(bddl_path)
    targets = choose_perturbation_targets(
        all_goal_targets, mode=args.position_target_mode,
    )
    handle = make_libero_env_for_task(
        str(root["task_id"]), init_state_id=int(root["init_state_id"]),
        seed=int(root["environment_seed"]), observation_height=args.observation_height,
        observation_width=args.observation_width, libero_clean_root=str(args.libero_clean_root),
        libero_flavor="clean",
    )
    try:
        single = handle.vector_env.envs[0]
        moved = apply_position_perturbation(
            single, level=float(args.position_level), seed=int(root["perturbation_seed"]),
            targets=targets,
        )
        if not moved:
            raise RuntimeError("no_goal_subject_free_joint_moved")
        observation = observation_from_libero_env(single)
        task = str(single.task_description)
        continuation = InProcessSmolVLAContinuation(
            bundle, temperature=temperature, seed=int(root["generation_seed"]),
        )
        continuation.reset()
        horizon = int(getattr(single, "_max_episode_steps", 600))
        success = False
        stop_reason = "horizon"
        started = time.perf_counter()
        for step in range(horizon):
            action = continuation.act(observation, task=task)
            observation, _reward, terminated, truncated, info = handle.vector_env.step(
                as_batched_action(action)
            )
            terminal, success = terminal_values(terminated, truncated, info)
            if terminal:
                stop_reason = "success" if success else "terminal_failure"
                break
        return {
            "schema_version": SCHEMA,
            "status": "complete",
            "root": dict(root),
            "temperature": temperature,
            "bddl_source": "clean_execution_assets",
            "bddl_path": str(bddl_path),
            "catalog_task_name": catalog_task_name,
            "targets_from_goal": all_goal_targets,
            "perturbation_targets": targets,
            "position_target_mode": args.position_target_mode,
            "moved_objects": moved,
            "success": bool(success),
            "rollout": {
                "steps": step + 1,
                "stop_reason": stop_reason,
                "elapsed_s": round(time.perf_counter() - started, 6),
            },
        }
    finally:
        handle.close()


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--init-states", type=int, default=10)
    parser.add_argument("--position-level", type=float, default=0.2)
    parser.add_argument(
        "--position-target-mode", choices=("all_goal_subjects", "first_goal_subject"),
        default="all_goal_subjects",
    )
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--libero-clean-root", type=Path, default=Path("/root/autodl-tmp/src/LIBERO"))
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.init_states < 1:
        parser.error("--episodes-per-task and --init-states must be positive")
    temperature = parse_temperature(args.temperature)
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        parser.error(f"--fresh-run refuses existing output {output}")
    output.mkdir(parents=True, exist_ok=True)
    roots_dir = output / "roots"
    roots_dir.mkdir(exist_ok=True)
    plan = make_plan(args, temperature=temperature)
    atomic_json(output / "root_plan.json", plan)
    bddl_dir = clean_bddl_directory(args.libero_clean_root, args.suite)
    if not args.policy.is_absolute():
        args.policy = ROOT / args.policy
    if not args.tokenizer.is_absolute():
        args.tokenizer = ROOT / args.tokenizer
    from rase.collect.forked_rollout import load_lerobot_policy_bundle

    bundle = load_lerobot_policy_bundle(
        args.policy, device="cuda", num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer, observation_height=args.observation_height,
        observation_width=args.observation_width,
    )
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    failures = 0
    for index, root in enumerate(plan["roots"]):
        path = roots_dir / f"{root['root_id']}.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete" and existing.get("root") == root:
                records.append(existing)
                print(
                    f"V6 source calibration skip {index + 1}/{len(plan['roots'])} "
                    f"{root['root_id']} status=complete",
                    flush=True,
                )
                continue
        try:
            record = run_root(root, args=args, bundle=bundle, bddl_dir=bddl_dir, temperature=temperature)
        except Exception as exc:
            failures += 1
            record = {
                "schema_version": SCHEMA, "status": "error", "root": root,
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
            }
        atomic_json(path, record)
        records.append(record)
        print(
            f"V6 source calibration {index + 1}/{len(plan['roots'])} "
            f"{root['root_id']} status={record['status']}",
            flush=True,
        )
    complete = [record for record in records if record["status"] == "complete"]
    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    for record in complete:
        task_id = str(record["root"]["task_id"])
        per_task[task_id]["episodes"] += 1
        per_task[task_id]["successes"] += int(record["success"])
    summary = {
        "schema_version": SCHEMA,
        "temperature": temperature,
        "position_level": args.position_level,
        "position_target_definition": "deterministic BDDL goal-subject subset selected before rollout",
        "position_target_mode": args.position_target_mode,
        "bddl_source": "clean_execution_assets",
        "n_planned_roots": len(plan["roots"]),
        "n_complete_roots": len(complete),
        "n_errors": failures,
        "successes": sum(int(record["success"]) for record in complete),
        "success_rate": (
            sum(int(record["success"]) for record in complete) / len(complete)
            if complete else None
        ),
        "per_task": dict(sorted(per_task.items())),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    write_jsonl(output / "records.jsonl", records)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

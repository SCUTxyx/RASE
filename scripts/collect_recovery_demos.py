#!/usr/bin/env python3
"""R4: Targeted recovery demonstration collection.

Two modes:
  1. B1 (matched OFT data): OFT demos from standard init states
  2. B2 (matched nominal): SmolVLA successful closed-loop trajectories
  3. B3 (targeted recovery): student runs until stagnation, OFT recovers from boundary

Output: per-task JSONL files with trajectory metadata and per-chunk NPZ files with
observations + teacher actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.smolvla_candidate_policy import load_smolvla_candidate_policy
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.envs.forkable_env import ForkableEnv
from rase.oracle.client import OracleClient

STAGNATION_WINDOW = 20
STAGNATION_EPS = 1e-6
MAX_STUDENT_STEPS = 300
MAX_TEACHER_STEPS = 300
MAX_SUCCESS_STEPS = 200
DEFAULT_MAX_RECOVERY = 3
DEFAULT_MIN_RECOVERABLE = 2


def _progress_signal(control_env: Any) -> float:
    try:
        obs = observation_from_libero_env(control_env.envs[0])
        if "robot0_eef_pos" in obs:
            return float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]).reshape(-1)))
    except Exception:
        pass
    return 0.0


def _get_task_instruction(control_env: Any) -> str:
    try:
        return str(getattr(control_env.envs[0], "task_description", "") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# B1: Matched OFT data — OFT runs from standard init states
# ---------------------------------------------------------------------------

def collect_matched_oft_data(
    *,
    task_id: str,
    init_state_id: int,
    client: OracleClient,
    n_episodes: int,
    episode_start: int,
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    rng = np.random.RandomState(seed)
    for ep in range(n_episodes):
        ep_seed = int(rng.randint(0, 10_000))
        handle = make_libero_env_for_task(task_id, init_state_id=init_state_id, seed=ep_seed)
        single = handle.control_env
        continuation = OracleChunkContinuation(client, instruction=_get_task_instruction(handle), env_id=0, control_env=single)

        success = False
        steps = 0
        stop = "horizon"
        chunk_idx = 0
        for _ in range(MAX_TEACHER_STEPS):
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            action = continuation.act(obs, task="")
            obs_arr = observation_from_libero_env(handle.vector_env.envs[0])
            np.savez_compressed(
                output_dir / f"chunk_ep{episode_start + ep:04d}_step{chunk_idx:04d}.npz",
                **{k: np.asarray(v) for k, v in obs_arr.items() if not k.startswith("pixels")},
                teacher_action=np.asarray(action, dtype=np.float32),
            )
            chunk_idx += 1
            _o, _r, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            steps += 1
            if success_from_info(info):
                success = True
                stop = "success"
                break
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                stop = "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated"
                break
        episodes.append({
            "episode_id": f"{task_id}_b1_{episode_start + ep:04d}",
            "task_id": task_id,
            "init_state_id": init_state_id,
            "mode": "matched_oft",
            "baseline": "B1",
            "success": success,
            "steps": steps,
            "stop_reason": stop,
            "chunk_count": chunk_idx,
        })
        handle.close()
    return episodes


# ---------------------------------------------------------------------------
# B2: Matched nominal — SmolVLA successful closed-loop trajectories
# ---------------------------------------------------------------------------

def collect_matched_nominal_data(
    *,
    task_id: str,
    init_state_id: int,
    policy_bundle: dict[str, Any],
    n_episodes: int,
    episode_start: int,
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    rng = np.random.RandomState(seed)
    attempts = 0
    while len(episodes) < n_episodes and attempts < n_episodes * 10:
        ep_seed = int(rng.randint(0, 100_000))
        handle = make_libero_env_for_task(task_id, init_state_id=init_state_id, seed=ep_seed)
        success = False
        steps = 0
        stop = "horizon"
        chunk_idx = 0
        for _ in range(MAX_SUCCESS_STEPS):
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            action = select_env_action(policy_bundle, obs, task="")
            np.savez_compressed(
                output_dir / f"chunk_ep{episode_start + len(episodes):04d}_step{chunk_idx:04d}.npz",
                **{k: np.asarray(v) for k, v in obs.items() if not k.startswith("pixels")},
                student_action=np.asarray(action, dtype=np.float32),
            )
            chunk_idx += 1
            _o, _r, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            steps += 1
            if success_from_info(info):
                success = True
                stop = "success"
                break
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                stop = "terminated" if bool(np.asarray(term).reshape(-1)[0]) else "truncated"
                break
        handle.close()
        if success:
            episodes.append({
                "episode_id": f"{task_id}_b2_{episode_start + len(episodes):04d}",
                "task_id": task_id,
                "init_state_id": init_state_id,
                "mode": "matched_nominal",
                "baseline": "B2",
                "success": success,
                "steps": steps,
                "stop_reason": stop,
                "chunk_count": chunk_idx,
            })
        attempts += 1
    return episodes


# ---------------------------------------------------------------------------
# B3: Targeted recovery — student rolls until stagnation, OFT recovers
# ---------------------------------------------------------------------------

def collect_targeted_recovery_demos(
    *,
    task_id: str,
    init_state_id: int,
    policy_bundle: dict[str, Any],
    client: OracleClient,
    max_recovery_per_task: int,
    episode_start: int,
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    episodes: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []

    student_successes = 0
    student_episodes = 0

    while len(recoveries) < max_recovery_per_task and student_episodes < max_recovery_per_task * 20:
        ep_seed = int(rng.randint(0, 100_000))
        handle = make_libero_env_for_task(task_id, init_state_id=init_state_id, seed=ep_seed)
        forkable = ForkableEnv(handle.control_env)
        single = handle.control_env.envs[0]

        progress_history: list[float] = []
        success = False
        steps = 0
        stagnation_triggered = False
        stagnation_snapshot = None

        for _ in range(MAX_STUDENT_STEPS):
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            action = select_env_action(policy_bundle, obs, task="")
            _o, _r, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            steps += 1
            progress = _progress_signal(handle.control_env)
            progress_history.append(progress)

            if success_from_info(info):
                success = True
                break

            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break

            if len(progress_history) >= STAGNATION_WINDOW:
                recent = progress_history[-STAGNATION_WINDOW:]
                delta = max(recent) - min(recent)
                if delta < STAGNATION_EPS:
                    stagnation_triggered = True
                    stagnation_snapshot = forkable.snapshot()
                    break

        if success:
            handle.close()
            student_successes += 1
            continue

        if not stagnation_triggered or stagnation_snapshot is None:
            handle.close()
            continue

        forkable.restore(stagnation_snapshot)

        continuation = OracleChunkContinuation(
            client, instruction=_get_task_instruction(handle), env_id=0, control_env=single,
        )

        t_success = False
        t_steps = 0
        t_stop = "horizon"
        chunk_idx = 0
        for _ in range(MAX_TEACHER_STEPS):
            t_obs = observation_from_libero_env(handle.vector_env.envs[0])
            t_act = continuation.act(t_obs, task="")
            obs_arr = observation_from_libero_env(handle.vector_env.envs[0])
            np.savez_compressed(
                output_dir / f"chunk_ep{episode_start + len(recoveries):04d}_step{chunk_idx:04d}.npz",
                **{k: np.asarray(v) for k, v in obs_arr.items() if not k.startswith("pixels")},
                teacher_action=np.asarray(t_act, dtype=np.float32),
            )
            chunk_idx += 1
            _o2, _r2, t_term, t_trunc, t_info = handle.vector_env.step(
                as_batched_action(t_act)
            )
            t_steps += 1
            if success_from_info(t_info):
                t_success = True
                t_stop = "success"
                break
            if bool(np.asarray(t_term).reshape(-1)[0]):
                t_stop = "terminated"
                break
            if bool(np.asarray(t_trunc).reshape(-1)[0]):
                t_stop = "truncated"
                break

        if t_success:
            recoveries.append({
                "episode_id": f"{task_id}_b3_{episode_start + len(recoveries):04d}",
                "task_id": task_id,
                "init_state_id": init_state_id,
                "mode": "targeted_recovery",
                "baseline": "B3",
                "student_success": False,
                "stagnation_step": steps,
                "teacher_success": t_success,
                "teacher_steps": t_steps,
                "teacher_stop": t_stop,
                "chunk_count": chunk_idx,
            })

        handle.close()
        student_episodes += 1

    episodes = clean + recoveries
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["b1", "b2", "b3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tasks", nargs="*", default=[], help="Override task list")
    parser.add_argument("--n-episodes-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smolvla-checkpoint", type=Path, default=ROOT / "ckpts" / "smolvla_libero")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "ckpts" / "SmolVLM2-500M-Instruct")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    suite_map = {
        "Object": "libero_object", "Goal": "libero_goal",
        "Spatial": "libero_spatial", "Long": "libero_10",
    }
    api_suite = suite_map.get(args.suite, f"libero_{args.suite.lower()}")

    tasks: list[str]
    if args.tasks:
        tasks = args.tasks
    else:
        prefix = api_suite
        tasks = [f"{prefix}_000001"]  # Default single task

    print(f"Mode: {args.mode.upper()}, Suite: {args.suite}, Tasks: {tasks}")
    print(f"Episodes per task: {args.n_episodes_per_task}")

    client = None
    policy_bundle = None

    if args.mode in ("b1", "b3"):
        client = OracleClient(args.endpoint, timeout_ms=120_000)
        print("OFT client connected")
    if args.mode in ("b2", "b3"):
        policy_bundle = load_smolvla_policy_bundle(
            Path(str(args.smolvla_checkpoint)), device="cuda",
            num_steps=10, n_action_steps=10,
            tokenizer_path=Path(str(args.tokenizer_path)),
            observation_height=360, observation_width=360,
        )
        print("SmolVLA student loaded")

    all_episodes: list[dict[str, Any]] = []
    for task_idx, task_id in enumerate(tasks):
        print(f"\n  [{task_idx + 1}/{len(tasks)}] {task_id}")
        for init_idx in range(min(10, max(3, args.n_episodes_per_task * 2))):
            ep_offset = (task_idx * 1000) + (init_idx * 100)
            if args.mode == "b1":
                eps = collect_matched_oft_data(
                    task_id=task_id, init_state_id=init_idx,
                    client=client,
                    n_episodes=max(1, args.n_episodes_per_task // 5),
                    episode_start=ep_offset,
                    output_dir=chunks_dir, seed=args.seed + ep_offset,
                )
            elif args.mode == "b2":
                eps = collect_matched_nominal_data(
                    task_id=task_id, init_state_id=init_idx,
                    policy_bundle=policy_bundle,
                    n_episodes=max(1, args.n_episodes_per_task // 5),
                    episode_start=ep_offset,
                    output_dir=chunks_dir, seed=args.seed + ep_offset,
                )
            else:
                eps = collect_targeted_recovery_demos(
                    task_id=task_id, init_state_id=init_idx,
                    policy_bundle=policy_bundle, client=client,
                    max_recovery_per_task=max(1, args.n_episodes_per_task // 5),
                    episode_start=ep_offset,
                    output_dir=chunks_dir, seed=args.seed + ep_offset,
                )
            all_episodes.extend(eps)
            if len(all_episodes) >= args.n_episodes_per_task:
                break
        print(f"    collected: {len(all_episodes)} episodes (target: {args.n_episodes_per_task})")

    if client:
        client.close()

    # Write index
    index_path = output_dir / "collection_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, sort_keys=True) + "\n")

    n_success = sum(1 for e in all_episodes if e.get("success") or e.get("teacher_success"))
    n_total = len(all_episodes)
    summary = {
        "mode": args.mode, "suite": args.suite, "n_episodes": n_total,
        "n_success": n_success, "success_rate": n_success / max(1, n_total),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(f"Episodes: {n_total} total, {n_success} success ({summary['success_rate']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

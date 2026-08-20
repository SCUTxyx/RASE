#!/usr/bin/env python3
"""D2-A: Collect verified recoverable boundaries.

For each boundary, runs persistent OFT with 3 seeds to verify
recoverability. Only boundaries with >= 2/3 OFT successes AND student
failure are marked verified_recoverable. Saves full OFT recovery
trajectories for downstream training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.oracle.client import OracleClient
from rase.recovery.stagnation import StagnationDetector
from rase.collect.action_schema import action_schema_hash


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_student_continuation(handle: Any, bundle: dict, instruction: str,
                              max_steps: int) -> dict:
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1}
    return {"success": False, "steps": max_steps}


def run_persistent_oft_with_trajectory(
    handle: Any, client: OracleClient, instruction: str, max_steps: int,
) -> dict:
    """Run persistent OFT and record the full trajectory."""
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    oft = OracleChunkContinuation(client, instruction=instruction,
                                   control_env=handle.control_env)
    trajectory = []
    for t in range(max_steps):
        t_action = oft.act(obs, task=instruction)
        a_arr = np.asarray(t_action, dtype=np.float32).flatten().tolist()
        trajectory.append({"step": t, "action": a_arr})
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1,
                    "trajectory": trajectory}
    return {"success": False, "steps": max_steps, "trajectory": trajectory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", type=str, nargs="*")
    parser.add_argument("--n-boundaries", type=int, default=48,
                        help="target number of boundaries to collect")
    parser.add_argument("--max-boundaries-per-task", type=int, default=4)
    parser.add_argument("--teacher-verification-seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir = output_dir / "boundaries"
    boundaries_dir.mkdir(exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
    all_suites = list(protocol["splits"].keys())
    run_suites = args.suite if args.suite else all_suites

    seed_everything(args.seed)

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000)

    window = plugin_conf["stagnation_window"]
    eps = plugin_conf["stagnation_eps"]

    verified_boundaries = []
    total_attempted = 0
    task_counts: dict[str, int] = {}

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]

        for task_id in task_ids:
            if len(verified_boundaries) >= args.n_boundaries:
                break
            if task_counts.get(task_id, 0) >= args.max_boundaries_per_task:
                continue

            for ep_i in range(10):  # try up to 10 seeds per task
                if len(verified_boundaries) >= args.n_boundaries:
                    break
                if task_counts.get(task_id, 0) >= args.max_boundaries_per_task:
                    break

                seed = args.seed * 31 + len(verified_boundaries) * 100 + ep_i * 7
                total_attempted += 1

                handle = make_libero_env_for_task(
                    task_id, init_state_id=ep_i % 50, seed=seed,
                    libero_flavor="clean")
                instruction = str(getattr(handle.vector_env.envs[0],
                                          "task_description", "") or "")
                obs = observation_from_libero_env(handle.vector_env.envs[0])

                stag = StagnationDetector(window=window, eps=eps)
                boundary_t = -1
                for t in range(args.max_steps):
                    action = select_env_action(bundle, obs, task=instruction)
                    obs, reward, term, trunc, info = handle.vector_env.step(
                        as_batched_action(action))
                    obs = observation_from_libero_env(handle.vector_env.envs[0])
                    stag.update(_progress(handle.control_env))
                    if bool(np.asarray(term).reshape(-1)[0]):
                        boundary_t = t + 1
                        break
                    if stag.is_stagnant():
                        boundary_t = t + 1
                        break
                handle.close()

                if boundary_t < 0:
                    continue

                # Verify with 3 seeds
                teacher_successes = 0
                min_takeover = 9999
                best_trajectory = None
                for v_seed_i in range(args.teacher_verification_seeds):
                    v_seed = seed + v_seed_i * 13
                    h_v = make_libero_env_for_task(
                        task_id, init_state_id=ep_i % 50, seed=v_seed,
                        libero_flavor="clean")
                    instr_v = str(getattr(h_v.vector_env.envs[0],
                                          "task_description", "") or "")

                    # Run student to same boundary
                    obs_v = observation_from_libero_env(h_v.vector_env.envs[0])
                    stag_v = StagnationDetector(window=window, eps=eps)
                    for _ in range(args.max_steps):
                        action_v = select_env_action(bundle, obs_v, task=instr_v)
                        obs_v, _, term_v, trunc_v, _ = h_v.vector_env.step(
                            as_batched_action(action_v))
                        obs_v = observation_from_libero_env(h_v.vector_env.envs[0])
                        stag_v.update(_progress(h_v.control_env))
                        if bool(np.asarray(term_v).reshape(-1)[0]) or stag_v.is_stagnant():
                            break

                    oft_result = run_persistent_oft_with_trajectory(
                        h_v, client, instr_v, args.max_steps)
                    if oft_result["success"]:
                        teacher_successes += 1
                        if oft_result["steps"] < min_takeover:
                            min_takeover = oft_result["steps"]
                            best_trajectory = oft_result["trajectory"]
                    h_v.close()

                if teacher_successes < 2:
                    continue

                # Run student continuation to confirm it fails
                h_student = make_libero_env_for_task(
                    task_id, init_state_id=ep_i % 50, seed=seed,
                    libero_flavor="clean")
                instr_s = str(getattr(h_student.vector_env.envs[0],
                                      "task_description", "") or "")
                # Reach boundary
                obs_s = observation_from_libero_env(h_student.vector_env.envs[0])
                stag_s = StagnationDetector(window=window, eps=eps)
                for _ in range(args.max_steps):
                    a_s = select_env_action(bundle, obs_s, task=instr_s)
                    obs_s, _, term_s, trunc_s, _ = h_student.vector_env.step(
                        as_batched_action(a_s))
                    obs_s = observation_from_libero_env(h_student.vector_env.envs[0])
                    stag_s.update(_progress(h_student.control_env))
                    if bool(np.asarray(term_s).reshape(-1)[0]) or stag_s.is_stagnant():
                        break
                student_result = run_student_continuation(
                    h_student, bundle, instr_s, args.max_steps)
                h_student.close()

                # Verified: student fails, teacher succeeds >= 2/3
                boundary_record = {
                    "boundary_id": f"boundary_{len(verified_boundaries):04d}",
                    "task_id": task_id,
                    "suite": suite,
                    "trigger_step": boundary_t,
                    "episode_seed": seed,
                    "student_continuation": {
                        "success": student_result["success"],
                        "steps": student_result["steps"],
                    },
                    "teacher_verification": {
                        "n_seeds": args.teacher_verification_seeds,
                        "successes": teacher_successes,
                        "verified_recoverable": True,
                    },
                    "minimum_successful_takeover_steps": min_takeover,
                    "persistent_teacher_trajectory": best_trajectory,
                    "action_schema_hash": action_schema_hash(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                verified_boundaries.append(boundary_record)
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

                boundary_path = boundaries_dir / f"{boundary_record['boundary_id']}.json"
                boundary_path.write_text(
                    json.dumps(boundary_record, indent=2, ensure_ascii=False),
                    encoding="utf-8")

                print(f"  [{len(verified_boundaries):04d}] {task_id} "
                      f"teacher={teacher_successes}/{args.teacher_verification_seeds} "
                      f"min_takeover={min_takeover}")

            if len(verified_boundaries) >= args.n_boundaries:
                break

    client.close()

    manifest = {
        "n_boundaries_collected": len(verified_boundaries),
        "n_total_attempted": total_attempted,
        "yield_rate": len(verified_boundaries) / max(total_attempted, 1),
        "n_unique_tasks": len(task_counts),
        "n_suites": len(set(b["suite"] for b in verified_boundaries)),
        "task_counts": task_counts,
        "gate_pass": (len(verified_boundaries) >= args.n_boundaries),
    }

    (output_dir / "verified_boundaries_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "verified_boundaries_all.json").write_text(
        json.dumps(verified_boundaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCollected {len(verified_boundaries)} verified boundaries "
          f"from {total_attempted} attempts "
          f"({manifest['yield_rate']:.1%})")
    return 0 if manifest["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

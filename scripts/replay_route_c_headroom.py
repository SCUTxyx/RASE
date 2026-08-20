#!/usr/bin/env python3
"""D0-C: Oracle teacher headroom replay.

From existing boundary snapshots, runs three branches:
  1. Student continuation (SmolVLA only)
  2. Learned plugin (SmolVLA + RecoveryPluginExecutor with takeover)
  3. Persistent OFT (teacher closed-loop, no switch-back)

Plus diagnostic H-sweep: H=[8, 32, 64] short OFT takeover then switch to SmolVLA.
The H-sweep does NOT participate in the formal recoverability gate.

Formal gate: persistent OFT success AND student continuation failure.
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
from rase.recovery.residual_plugin import load_plugin
from rase.recovery.plugin_executor import RecoveryPluginExecutor
from rase.collect.smolvla_feature_extractor import build_feature_vector
from rase.recovery.stagnation import StagnationDetector


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


# ── Branch runners ────────────────────────────────────────────────

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


def run_persistent_oft(handle: Any, client: OracleClient, instruction: str,
                        max_steps: int) -> dict:
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    oft = OracleChunkContinuation(client, instruction=instruction,
                                   control_env=handle.control_env)
    for t in range(max_steps):
        t_action = oft.act(obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1}
    return {"success": False, "steps": max_steps}


def run_short_oft_then_student(handle: Any, client: OracleClient, bundle: dict,
                                 instruction: str, h: int,
                                 max_continuation: int) -> dict:
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    oft = OracleChunkContinuation(client, instruction=instruction,
                                   control_env=handle.control_env)
    for t in range(h):
        t_action = oft.act(obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1, "oft_steps": t + 1}
    for t in range(max_continuation):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": h + t + 1, "oft_steps": h}
    return {"success": False, "steps": h + max_continuation, "oft_steps": h}


def run_plugin_continuation(handle: Any, bundle: dict, executor: RecoveryPluginExecutor,
                              instruction: str, max_steps: int,
                              feature_level: str = "F2") -> dict:
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    executor.reset()
    takeover_count = 0
    for t in range(max_steps):
        student_action = select_env_action(bundle, obs, task=instruction)
        progress = _progress(handle.control_env)
        if executor.should_takeover(progress):
            takeover_count += 1
        proprio = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
        if len(proprio) < 7:
            quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
            proprio = np.concatenate([proprio, quat])[:7]
        stagnation_len = len(executor._stagnation.progress_values)
        obs_feat = build_feature_vector(
            smolvla_latent=None, proprio=proprio,
            student_action=student_action.flatten(),
            stagnation_length=stagnation_len, progress_delta=0.0,
            feature_level=feature_level,
        )
        action, info = executor.step(obs, student_action, progress, obs_features=obs_feat)
        if info.get("takeover"):
            takeover_count += 1
        executor.record_history(proprio, student_action, progress, obs_feat)
        obs, reward, term, trunc, env_info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(env_info), "steps": t + 1,
                    "takeover_steps": takeover_count}
    return {"success": False, "steps": max_steps, "takeover_steps": takeover_count}


# ── Classification ─────────────────────────────────────────────────

def classify_branches(student: dict, plugin: dict, persistent_oft: dict) -> str:
    s_ok = student["success"]
    p_ok = plugin["success"]
    pf_ok = persistent_oft["success"]
    if not pf_ok and not s_ok and not p_ok:
        return "boundary_unrecoverable"
    if not pf_ok and not s_ok and p_ok:
        return "plugin_only_unreliable"
    if pf_ok and not s_ok and not p_ok:
        return "plumbing_or_training_issue"
    if pf_ok and not s_ok and p_ok:
        return "recovery_pathway_verified"
    if pf_ok and not s_ok and plugin.get("takeover_steps", 0) > 0 and not p_ok:
        return "plugin_takeover_no_recovery"
    if not pf_ok and s_ok:
        return "not_failure_boundary"
    if pf_ok and s_ok:
        return "student_and_oft_both_succeed"
    return "unclassified"


# ── Main ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plugin-ckpt", type=Path, default=None)
    parser.add_argument("--suite", type=str, nargs="*")
    parser.add_argument("--n-snapshots", type=int, default=4)
    parser.add_argument("--matched-seeds", type=int, default=3)
    parser.add_argument("--max-student-steps", type=int, default=300)
    parser.add_argument("--max-teacher-steps", type=int, default=300)
    parser.add_argument("--h-sweep", type=int, nargs="*", default=[8, 32, 64])
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--feature-level", type=str, default="F2",
                        choices=["F0", "F1", "F2"])
    parser.add_argument("--oft-server-port", type=int, default=5555)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
    all_suites = list(protocol["splits"].keys())
    run_suites = args.suite if args.suite else [all_suites[0]]

    seed_everything(args.seed)

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    plugin = None
    executor = None
    if args.plugin_ckpt and args.plugin_ckpt.is_file():
        plugin = load_plugin(str(args.plugin_ckpt))
        plugin.eval()
        executor = RecoveryPluginExecutor(plugin, bundle,
            history_window=plugin_conf["plugin_history_window"],
            stagnation_window=plugin_conf["stagnation_window"],
            stagnation_eps=plugin_conf["stagnation_eps"],
            max_takeover_steps=plugin_conf["max_takeover_steps"],
            delta_clip=plugin_conf["delta_clip_per_dim"])

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000)

    replay_results = []
    classification_counts: dict[str, int] = {}
    snapshot_idx = 0

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]
        for task_id in task_ids[:args.n_snapshots]:
            for seed_i in range(args.matched_seeds):
                if snapshot_idx >= args.n_snapshots:
                    break
                seed = args.seed * 31 + snapshot_idx * 100 + seed_i * 7
                max_student = min(50, args.max_student_steps // 2)

                # Run to boundary via stagnation
                handle_b0 = make_libero_env_for_task(
                    task_id, init_state_id=seed_i % 50, seed=seed,
                    libero_flavor="clean")
                instruction = str(getattr(handle_b0.vector_env.envs[0],
                                          "task_description", "") or "")
                obs = observation_from_libero_env(handle_b0.vector_env.envs[0])
                stag = StagnationDetector(
                    window=plugin_conf["stagnation_window"],
                    eps=plugin_conf["stagnation_eps"])
                boundary_reached = False
                for t in range(max_student):
                    action = select_env_action(bundle, obs, task=instruction)
                    obs, reward, term, trunc, info = handle_b0.vector_env.step(
                        as_batched_action(action))
                    obs = observation_from_libero_env(handle_b0.vector_env.envs[0])
                    stag.update(_progress(handle_b0.control_env))
                    if bool(np.asarray(term).reshape(-1)[0]):
                        boundary_reached = True
                        break
                    if stag.is_stagnant():
                        boundary_reached = True
                        break
                handle_b0.close()
                if not boundary_reached:
                    continue

                branches: dict[str, dict] = {}

                # Branch 1: Student
                h_student = make_libero_env_for_task(
                    task_id, init_state_id=seed_i % 50, seed=seed,
                    libero_flavor="clean")
                branches["student"] = run_student_continuation(
                    h_student, bundle, instruction, args.max_student_steps)
                h_student.close()

                # Branch 2: Plugin
                if executor:
                    h_plugin = make_libero_env_for_task(
                        task_id, init_state_id=seed_i % 50, seed=seed,
                        libero_flavor="clean")
                    branches["plugin"] = run_plugin_continuation(
                        h_plugin, bundle, executor, instruction,
                        args.max_student_steps, args.feature_level)
                    h_plugin.close()
                else:
                    branches["plugin"] = {"success": False, "steps": 0,
                                           "takeover_steps": 0, "note": "no plugin"}

                # Branch 3: Persistent OFT
                h_oft = make_libero_env_for_task(
                    task_id, init_state_id=seed_i % 50, seed=seed,
                    libero_flavor="clean")
                branches["persistent_oft"] = run_persistent_oft(
                    h_oft, client, instruction, args.max_teacher_steps)
                h_oft.close()

                # H-sweep
                h_sweep_results = {}
                for h in args.h_sweep:
                    h_s = make_libero_env_for_task(
                        task_id, init_state_id=seed_i % 50, seed=seed,
                        libero_flavor="clean")
                    h_sweep_results[f"H_{h}"] = run_short_oft_then_student(
                        h_s, client, bundle, instruction, h,
                        args.max_teacher_steps - h)
                    h_s.close()

                category = classify_branches(
                    branches["student"], branches["plugin"],
                    branches["persistent_oft"])
                classification_counts[category] = classification_counts.get(category, 0) + 1

                rec = {
                    "snapshot_idx": snapshot_idx, "task_id": task_id,
                    "suite": suite, "seed": seed, "seed_i": seed_i,
                    "student": branches["student"],
                    "plugin": branches["plugin"],
                    "persistent_oft": branches["persistent_oft"],
                    "h_sweep": h_sweep_results,
                    "classification": category,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                replay_results.append(rec)
                snapshot_idx += 1
                print(f"  [{suite}] {task_id} seed={seed}: "
                      f"s={branches['student']['success']} "
                      f"p={branches['plugin']['success']} "
                      f"oft={branches['persistent_oft']['success']} "
                      f"→ {category}")

    client.close()

    persistent_oft_rescues = sum(
        1 for r in replay_results
        if r["persistent_oft"]["success"] and not r["student"]["success"])
    gate_pass = persistent_oft_rescues >= 1

    summary = {
        "total_snapshots": len(replay_results),
        "persistent_oft_rescues": persistent_oft_rescues,
        "classification_counts": classification_counts,
        "gate_pass": gate_pass,
    }

    (output_dir / "headroom_replay.json").write_text(
        json.dumps(replay_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "headroom_classification.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "headroom_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nHeadroom: {persistent_oft_rescues}/{len(replay_results)} OFT rescues, "
          f"gate={gate_pass}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

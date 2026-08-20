#!/usr/bin/env python3
"""Route C B0-B4 paired evaluation using existing RASE infrastructure."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from rase.collect.smolvla_feature_extractor import SmolVLAFeatureExtractor, build_feature_vector


STAGNATION_WINDOW = 5
STAGNATION_EPS = 2e-2


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def run_b0(handle: Any, bundle: dict, instruction: str, max_steps: int) -> dict:
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "mode": "B0"}
    return {"success": False, "steps": max_steps, "mode": "B0"}


def run_b1(handle: Any, bundle: dict, client: OracleClient, instruction: str,
           max_student: int, max_teacher: int) -> dict:
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    progress_vals = []
    handover = False
    boundary_t = -1

    for t in range(max_student):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "mode": "B1", "handover": False}
        progress_vals.append(_progress(handle.control_env))
        if t >= STAGNATION_WINDOW and np.std(progress_vals[-STAGNATION_WINDOW:]) < STAGNATION_EPS:
            boundary_t = t + 1
            handover = True
            break

    if handover:
        oft = OracleChunkContinuation(client, instruction=instruction, control_env=handle.control_env)
        for ti in range(max_teacher):
            t_action = oft.act(obs, task=instruction)
            obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            terminated = bool(np.asarray(term).reshape(-1)[0])
            truncated = bool(np.asarray(trunc).reshape(-1)[0])
            if terminated or truncated:
                return {"success": success_from_info(info), "steps": boundary_t + ti + 1,
                        "mode": "B1", "handover": True}
        return {"success": False, "steps": boundary_t + max_teacher, "mode": "B1", "handover": True}
    return {"success": False, "steps": max_student, "mode": "B1", "handover": False}


def run_b3(handle: Any, bundle: dict, executor: RecoveryPluginExecutor,
           instruction: str, max_steps: int,
           feature_level: str = "F2",
           feature_extractor: SmolVLAFeatureExtractor | None = None) -> dict:
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    executor.reset()
    takeover_count = 0

    for t in range(max_steps):
        student_action = select_env_action(bundle, obs, task=instruction)
        progress = _progress(handle.control_env)

        # Build observation features with REAL SmolVLA latent
        if feature_extractor is not None and feature_level == "F2":
            smolvla_latent = feature_extractor.extract(obs)
        else:
            smolvla_latent = None

        proprio = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
        if len(proprio) < 7:
            quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
            proprio = np.concatenate([proprio, quat])[:7]
        else:
            proprio = proprio[:7]

        stagnation_len = len(executor._stagnation.progress_values) if hasattr(executor, '_stagnation') else 0
        progress_delta = 0.0
        obs_feat = build_feature_vector(
            smolvla_latent=smolvla_latent,
            proprio=proprio,
            student_action=student_action.flatten(),
            stagnation_length=stagnation_len,
            progress_delta=progress_delta,
            feature_level=feature_level,
        )

        if executor.should_takeover(progress):
            takeover_count += 1

        action, info = executor.step(obs, student_action, progress, obs_features=obs_feat)
        if info.get("takeover"):
            takeover_count += 1
        executor.record_history(proprio, student_action, progress, obs_feat)

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "mode": "B3",
                    "takeover_steps": takeover_count}
    return {"success": False, "steps": max_steps, "mode": "B3", "takeover_steps": takeover_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plugin-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="*", default=["B0", "B3"],
                        choices=["B0", "B1", "B2", "B3"])
    parser.add_argument("--suite", type=str, nargs="*",
                        help="suite(s) to eval (default: first)")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-student-steps", type=int, default=300)
    parser.add_argument("--max-teacher-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    parser.add_argument("--feature-level", type=str, default="F2",
                        choices=["F0", "F1", "F2"])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]

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
    feature_extractor = None
    if args.plugin_ckpt and args.plugin_ckpt.is_file() and "B3" in args.modes:
        plugin = load_plugin(str(args.plugin_ckpt))
        plugin.eval()
        executor = RecoveryPluginExecutor(plugin, bundle,
                                          history_window=plugin_conf["plugin_history_window"],
                                          stagnation_window=plugin_conf["stagnation_window"],
                                          stagnation_eps=plugin_conf["stagnation_eps"],
                                          max_takeover_steps=plugin_conf["max_takeover_steps"],
                                          delta_clip=plugin_conf["delta_clip_per_dim"])
        # Initialize SmolVLA feature extractor for REAL latent extraction
        if args.feature_level == "F2":
            feature_extractor = SmolVLAFeatureExtractor(bundle)

    all_suites = list(protocol["splits"].keys())
    run_suites = args.suite if args.suite else [all_suites[0]]
    all_results: dict[str, list[dict]] = {m: [] for m in args.modes}

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000) \
        if "B1" in args.modes else None

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]

        for task_id in task_ids[:3]:
            for ep_i in range(args.n_episodes):
                seed = (args.seed * 31 + run_suites.index(suite) * 100 + ep_i * 7) % (2**31)

                if "B0" in args.modes:
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=seed,
                                                       libero_flavor="clean")
                    instr = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
                    r = run_b0(handle, bundle, instr, args.max_student_steps)
                    r.update({"suite": suite, "task_id": task_id, "seed": seed})
                    all_results["B0"].append(r)
                    handle.close()

                if "B1" in args.modes and client:
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=seed,
                                                       libero_flavor="clean")
                    instr = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
                    r = run_b1(handle, bundle, client, instr,
                               args.max_student_steps, args.max_teacher_steps)
                    r.update({"suite": suite, "task_id": task_id, "seed": seed})
                    all_results["B1"].append(r)
                    handle.close()

                if "B2" in args.modes:
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=seed,
                                                       libero_flavor="clean")
                    instr = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
                    r = run_b0(handle, bundle, instr, args.max_student_steps)
                    r["mode"] = "B2"
                    r.update({"suite": suite, "task_id": task_id, "seed": seed})
                    all_results["B2"].append(r)
                    handle.close()

                if "B3" in args.modes and executor:
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=seed,
                                                       libero_flavor="clean")
                    instr = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
                    r = run_b3(handle, bundle, executor, instr, args.max_student_steps,
                              feature_level=args.feature_level,
                              feature_extractor=feature_extractor)
                    r.update({"suite": suite, "task_id": task_id, "seed": seed})
                    all_results["B3"].append(r)
                    handle.close()

    if client:
        client.close()

    summary = {}
    for mode in args.modes:
        mr = all_results.get(mode, [])
        if mr:
            s = sum(1 for r in mr if r.get("success"))
            summary[mode] = {"n": len(mr), "success": s, "rate": s / max(len(mr), 1)}

    (output_dir / "paired_eval_results.json").write_text(
        json.dumps({"results": all_results, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    b0_rate = summary.get("B0", {}).get("rate", 0)
    b3_rate = summary.get("B3", {}).get("rate", 0)
    b2_rate = summary.get("B2", {}).get("rate", 0)

    dev_gate = {"B0_success": b0_rate, "B3_success": b3_rate, "B2_success": b2_rate,
                "B3_gt_B2": b3_rate > b2_rate,
                "clean_degradation": b0_rate - b3_rate <= 0.02,
                "gate_pass": b3_rate > b2_rate and (b0_rate - b3_rate <= 0.02)}
    (output_dir / "dev_decision.json").write_text(
        json.dumps(dev_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Eval: B0={b0_rate:.2%} B3={b3_rate:.2%} B2={b2_rate:.2%} gate={dev_gate['gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

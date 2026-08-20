#!/usr/bin/env python3
"""Route C data collection: R0 (recovery), N0 (nominal), F0 (teacher-fails).

Uses existing RASE infrastructure for environment management and OFT interfacing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
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
from rase.recovery.feature_pipeline import RecoveryFeaturePipeline
from rase.recovery.history_buffer import RecoveryHistoryBuffer
from rase.recovery.schema import make_version_metadata


STAGNATION_WINDOW = 5
STAGNATION_EPS = 2e-2  # loosened from 2e-4: catch "flailing without progress", not just frozen


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _progress_signal(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def _get_proprio(obs: dict[str, Any]) -> np.ndarray:
    """Extract proprioceptive vector from a LIBERO observation dict."""
    pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
    quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
    return np.concatenate([pos, quat])[:7]


def _estimate_stagnation_len(progress_vals: list[float]) -> int:
    """Heuristic: count consecutive trailing steps with near-zero progress delta."""
    if len(progress_vals) < 2:
        return 0
    n = 0
    for i in range(len(progress_vals) - 1, 0, -1):
        if abs(progress_vals[i] - progress_vals[i - 1]) < STAGNATION_EPS:
            n += 1
        else:
            break
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["R0", "N0", "F0", "all"], default="all")
    parser.add_argument("--suite", type=str, nargs="*",
                        help="suite(s) to collect (default: first)")
    parser.add_argument("--n-episodes-per-task", type=int, default=4)
    parser.add_argument("--max-student-steps", type=int, default=300)
    parser.add_argument("--max-teacher-steps", type=int, default=300)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"],
                        help="protocol split to collect from (default: dev)")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    action_dim = protocol["action_schema"]["action_dim"]
    delta_clip = protocol["plugin_config"].get("delta_clip_per_dim", 0.5)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for m in ["R0", "N0", "F0"]:
        (output_root / m).mkdir(exist_ok=True)

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

    # Initialize shared feature pipeline with the same SmolVLA checkpoint
    feature_pipeline = RecoveryFeaturePipeline(bundle)

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000)
    collected: dict[str, list[dict]] = {"R0": [], "N0": [], "F0": []}

    for suite in run_suites:
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite][args.split]

        for task_id in task_ids[:4]:
            for ep_i in range(args.n_episodes_per_task):
                ep_seed = (args.seed * 10000 + run_suites.index(suite) * 100 + ep_i) % (2**32)

                # ── N0: nominal student rollout ──
                if args.mode in ("N0", "all"):
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=ep_seed,
                                                       libero_flavor="clean")
                    single = handle.vector_env.envs[0]
                    instruction = str(getattr(single, "task_description", "") or "")
                    obs = observation_from_libero_env(single)
                    steps = []
                    for t in range(args.max_student_steps):
                        action = select_env_action(bundle, obs, task=instruction)
                        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                        obs = observation_from_libero_env(single)
                        terminated = bool(np.asarray(term).reshape(-1)[0])
                        truncated = bool(np.asarray(trunc).reshape(-1)[0])
                        step = {"t": t, "action": action.tolist(), "reward": float(reward),
                                "done": terminated or truncated,
                                "success": success_from_info(info) if (terminated or truncated) else False,
                                "delta_target": [0.0] * action_dim, "label": "N0", "phase": "nominal"}
                        steps.append(step)
                        if terminated or truncated:
                            break
                    ep_rec = {"suite": suite, "task_id": task_id, "episode_seed": ep_seed, "mode": "N0",
                              "n_steps": len(steps),
                              "student_success": steps[-1]["success"] if steps else False,
                              "steps": steps}
                    (output_root / "N0" / f"{suite}_{task_id}_N0_s{ep_seed}.json").write_text(
                        json.dumps(ep_rec, indent=2, ensure_ascii=False), encoding="utf-8")
                    collected["N0"].append({"suite": suite, "task_id": task_id, "seed": ep_seed,
                                            "steps": len(steps)})
                    handle.close()

                # ── R0 / F0: student → boundary → OFT recovery ──
                if args.mode in ("R0", "F0", "all"):
                    handle = make_libero_env_for_task(task_id, init_state_id=ep_i % 50, seed=ep_seed,
                                                       libero_flavor="clean")
                    single = handle.vector_env.envs[0]
                    instruction = str(getattr(single, "task_description", "") or "")
                    obs = observation_from_libero_env(single)

                    # Fresh history buffer and stagnation tracking
                    history_buffer = RecoveryHistoryBuffer(
                        window=args.history_window,
                        proprio_dim=8,
                        action_dim=action_dim,
                    )
                    progress_vals: list[float] = []
                    student_steps_list: list[dict] = []
                    boundary_t = -1

                    for t in range(args.max_student_steps):
                        action = select_env_action(bundle, obs, task=instruction)
                        proprio_raw = _get_proprio(obs)
                        pre_step_progress = _progress_signal(handle.control_env)

                        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                        obs = observation_from_libero_env(single)
                        terminated = bool(np.asarray(term).reshape(-1)[0])
                        truncated = bool(np.asarray(trunc).reshape(-1)[0])
                        prog = _progress_signal(handle.control_env)
                        progress_vals.append(prog)

                        # Build obs_features for this student step (F0 for student
                        # rollout; full F2 features will be used during teacher recovery)
                        features = feature_pipeline.extract(
                            obs, proprio_raw, action,
                            stagnation_length=0,
                            progress_delta=0.0,
                            feature_level="F2",
                        )

                        # Record into history buffer (student action = executed action)
                        history_buffer.append(proprio_raw, action, prog, executed_action=action)

                        step = {"t": t, "action": action.tolist(),
                                "reward": float(reward), "done": terminated or truncated,
                                "success": success_from_info(info) if (terminated or truncated) else False,
                                "obs_features": features.obs_features.tolist(),
                                "phase": "student_roll_in"}
                        student_steps_list.append(step)

                        if terminated or truncated:
                            boundary_t = t + 1
                            break
                        if t >= STAGNATION_WINDOW:
                            window = progress_vals[t - STAGNATION_WINDOW + 1:t + 1]
                            if np.std(window) < STAGNATION_EPS and max(window) > 1e-8:
                                boundary_t = t + 1
                                break
                    if boundary_t < 0:
                        boundary_t = len(student_steps_list)

                    student_ok = student_steps_list[min(boundary_t - 1, len(student_steps_list) - 1)]["success"] \
                        if student_steps_list else True

                    # OFT recovery
                    recovery_steps: list[dict] = []
                    teacher_ok = False
                    if not (isinstance(student_ok, bool) and student_ok):
                        oft = OracleChunkContinuation(client, instruction=instruction,
                                                       control_env=handle.control_env)
                        for ti in range(args.max_teacher_steps):
                            proprio_raw = _get_proprio(obs)
                            student_action = select_env_action(bundle, obs, task=instruction)

                            # Capture history BEFORE this step (no current info leaked)
                            hist_tensor = history_buffer.get_history_tensor()

                            # Extract features using current observation
                            # stagnation_length estimated from progress_vals
                            stag_len = _estimate_stagnation_len(progress_vals)
                            features = feature_pipeline.extract(
                                obs, proprio_raw, student_action,
                                stagnation_length=stag_len,
                                progress_delta=0.0,
                                feature_level="F2",
                            )

                            t_action = oft.act(obs, task=instruction)
                            delta_t = np.clip(t_action - student_action, -delta_clip, delta_clip)

                            obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
                            obs = observation_from_libero_env(single)
                            terminated = bool(np.asarray(term).reshape(-1)[0])
                            truncated = bool(np.asarray(trunc).reshape(-1)[0])

                            # Record to history after this step (executed = teacher action)
                            prog = _progress_signal(handle.control_env)
                            history_buffer.append(proprio_raw, student_action, prog,
                                                  executed_action=t_action)

                            step = {"t": ti, "action": student_action.tolist(),
                                    "teacher_action": t_action.tolist(),
                                    "delta_target": delta_t.tolist(),
                                    "obs_features": features.obs_features.tolist(),
                                    "history_before": hist_tensor["data"].tolist(),
                                    "history_mask": hist_tensor["mask"].tolist(),
                                    "reward": float(reward), "done": terminated or truncated,
                                    "success": success_from_info(info) if (terminated or truncated) else False,
                                    "phase": "teacher_recovery"}
                            recovery_steps.append(step)
                            if terminated or truncated:
                                teacher_ok = success_from_info(info)
                                break

                    recoverable = teacher_ok and not student_ok
                    post_irr = not student_ok and not teacher_ok

                    # Build version metadata for episode files
                    version_meta = make_version_metadata(
                        pipeline_version=feature_pipeline.version,
                        extractor_sha=feature_pipeline.extractor_sha,
                        history_window=args.history_window,
                        proprio_dim=8,
                        action_dim=action_dim,
                        obs_feature_dim=int(feature_pipeline.latent_dim + 7 + 7 + 2),
                    )

                    if recoverable and args.mode in ("R0", "all"):
                        ep_rec = {"suite": suite, "task_id": task_id, "episode_seed": ep_seed, "mode": "R0",
                                  "boundary_step": boundary_t, "student_success": student_ok,
                                  "teacher_success": teacher_ok, "boundary_history": [],
                                  "student_rollin": student_steps_list,
                                  "teacher_recovery": recovery_steps,
                                  **version_meta}
                        (output_root / "R0" / f"{suite}_{task_id}_R0_s{ep_seed}.json").write_text(
                            json.dumps(ep_rec, indent=2, ensure_ascii=False), encoding="utf-8")
                        collected["R0"].append({"suite": suite, "task_id": task_id, "seed": ep_seed,
                                                "boundary_step": boundary_t,
                                                "recovery_steps": len(recovery_steps)})

                    elif post_irr and args.mode in ("F0", "all"):
                        ep_rec = {"suite": suite, "task_id": task_id, "episode_seed": ep_seed, "mode": "F0",
                                  "boundary_step": boundary_t, "student_success": student_ok,
                                  "teacher_success": teacher_ok, "boundary_history": [],
                                  "student_rollin": student_steps_list,
                                  "teacher_recovery": recovery_steps,
                                  **version_meta}
                        (output_root / "F0" / f"{suite}_{task_id}_F0_s{ep_seed}.json").write_text(
                            json.dumps(ep_rec, indent=2, ensure_ascii=False), encoding="utf-8")
                        collected["F0"].append({"suite": suite, "task_id": task_id, "seed": ep_seed,
                                                "boundary_step": boundary_t,
                                                "recovery_steps": len(recovery_steps)})
                    handle.close()

    client.close()

    r0_unique = set((d["suite"], d["task_id"]) for d in collected["R0"])
    n_r0 = len(collected["R0"])
    gate = {"R0_count": n_r0, "unique_R0_boundaries": len(r0_unique),
            "N0_count": len(collected["N0"]), "F0_count": len(collected["F0"]),
            "R0_min_48": n_r0 >= 48, "coverage_pass": len(r0_unique) >= 2}
    gate["gate_pass"] = n_r0 > 0

    (output_root / "collection_summary.json").write_text(
        json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "round0_plugin_data_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Collection: R0={n_r0}, N0={len(collected['N0'])}, F0={len(collected['F0'])}, "
          f"unique_R0={len(r0_unique)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

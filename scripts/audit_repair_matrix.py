#!/usr/bin/env python3
"""PRE-C0-R0 Step 2: Counterfactual Repair Matrix collection.

Simplified version: run B0 episodes, snapshot at multiple predetermined points
(midpoint, late, etc.), then fork and run Base/F0/F2 repair arms.

Uses ForkableEnv with handle.control_env for snapshot/restore.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def _proprio(obs: dict) -> np.ndarray:
    p = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
    q = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
    return np.concatenate([p, q])[:7].astype(np.float32)


def _object_drop_check(env_handle: Any) -> bool:
    try:
        env = env_handle if hasattr(env_handle, "is_object_dropped") else env_handle.vector_env.envs[0]
        if hasattr(env, "is_object_dropped"):
            return bool(env.is_object_dropped())
    except Exception:
        pass
    return False


# ── B0 snapshot collection ───────────────────────────────────────────

def run_b0_and_collect_snapshots(handle, bundle, instruction, max_steps,
                                   snapshot_fracs=(0.3, 0.5, 0.7),
                                   max_per_ep=2) -> dict:
    """Run B0 episode, snapshot env at fraction-based points."""
    from rase.envs.forkable_env import ForkableEnv
    from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    forkable = ForkableEnv(handle.control_env)
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    snapshots = []

    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])

        # Snapshot at fraction-based points
        if not terminated and not truncated and len(snapshots) < max_per_ep:
            for frac in snapshot_fracs:
                expected_t = int(max_steps * frac)
                if t == min(expected_t, max_steps - 10) and t > 10:
                    try:
                        snap = forkable.snapshot()
                        snapshots.append({
                            "t": t,
                            "type": f"step_{frac:.0%}",
                            "deviation": 0.0,
                            "snapshot": snap,
                        })
                    except Exception as e:
                        print(f"    WARNING: snapshot failed at t={t}: {e}")
                    break

        if terminated or truncated:
            success = success_from_info(info)
            return {"success": success, "steps": t + 1, "snapshots": snapshots}

    return {"success": False, "steps": max_steps, "snapshots": snapshots}


# ── Repair arm runners ───────────────────────────────────────────────

def run_repair_base(handle, bundle, instruction, max_steps) -> dict:
    from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    obs = observation_from_libero_env(handle.vector_env.envs[0])
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "Base",
                    "drop": _object_drop_check(handle), "collision": False}
    return {"success": False, "steps": max_steps, "arm": "Base",
            "drop": _object_drop_check(handle), "collision": False}


def run_repair_f0(handle, bundle, constant_delta, instruction, max_steps) -> dict:
    from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    obs = observation_from_libero_env(handle.vector_env.envs[0])
    delta = np.array(constant_delta, dtype=np.float32)
    for t in range(max_steps):
        student_action = select_env_action(bundle, obs, task=instruction)
        delta_clipped = np.clip(delta, -0.5, 0.5)
        mixed = np.clip(student_action.flatten() + delta_clipped, -1.0, 1.0)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(mixed.reshape(1, -1)))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "F0",
                    "drop": _object_drop_check(handle), "collision": False}
    return {"success": False, "steps": max_steps, "arm": "F0",
            "drop": _object_drop_check(handle), "collision": False}


def run_repair_f2(handle, bundle, plugin, feature_extractor, instruction, max_steps) -> dict:
    from rase.collect.smolvla_feature_extractor import build_feature_vector
    from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    obs = observation_from_libero_env(handle.vector_env.envs[0])
    plugin_history = []
    HISTORY_WINDOW = 8

    for t in range(max_steps):
        if feature_extractor is not None:
            feature_extractor.start_capture()
        student_action = select_env_action(bundle, obs, task=instruction)
        if feature_extractor is not None:
            smolvla_latent = feature_extractor.finish_capture()
        else:
            smolvla_latent = None

        progress_val = _progress(handle.control_env)
        proprio_val = _proprio(obs)

        obs_feat = build_feature_vector(
            smolvla_latent=smolvla_latent,
            proprio=proprio_val,
            student_action=student_action.flatten(),
            stagnation_length=len(plugin_history),
            progress_delta=0.0,
            feature_level="F2",
        )

        hist_arr = np.zeros((HISTORY_WINDOW, 23), dtype=np.float32)
        recent = plugin_history[-HISTORY_WINDOW:]
        for hi, h in enumerate(recent):
            p = np.asarray(h.get("proprio", np.zeros(8)), dtype=np.float32).flatten()
            a = np.asarray(h.get("student_action", np.zeros(7)), dtype=np.float32).flatten()
            p_pad = np.zeros(8, dtype=np.float32)
            a_pad = np.zeros(7, dtype=np.float32)
            p_pad[:min(len(p), 8)] = p[:8]
            a_pad[:min(len(a), 7)] = a[:7]
            idx = hi + HISTORY_WINDOW - len(recent)
            if 0 <= idx < HISTORY_WINDOW:
                hist_arr[idx, :8] = p_pad
                hist_arr[idx, 8:15] = a_pad
                hist_arr[idx, 15] = float(h.get("progress", 0))
                hist_arr[idx, 16:23] = a_pad

        obs_feat_plugin = obs_feat
        if hasattr(plugin, 'obs_feature_dim') and len(obs_feat_plugin) != plugin.obs_feature_dim:
            obs_feat_plugin = np.zeros(plugin.obs_feature_dim, dtype=np.float32)
            n_copy = min(len(obs_feat), plugin.obs_feature_dim)
            obs_feat_plugin[:n_copy] = obs_feat[:n_copy]

        delta = plugin.predict_delta(hist_arr, obs_feat_plugin, student_action.flatten())
        delta_clipped = np.clip(delta, -0.5, 0.5)
        mixed = np.clip(student_action.flatten() + delta_clipped, -1.0, 1.0)

        plugin_history.append({
            "proprio": proprio_val,
            "student_action": student_action.flatten(),
            "progress": float(progress_val),
        })
        if len(plugin_history) > HISTORY_WINDOW * 2:
            plugin_history = plugin_history[-HISTORY_WINDOW * 2:]

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(mixed.reshape(1, -1)))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "F2",
                    "drop": _object_drop_check(handle), "collision": False}
    return {"success": False, "steps": max_steps, "arm": "F2",
            "drop": _object_drop_check(handle), "collision": False}


def run_repair_guided(handle, client, instruction, max_steps) -> dict:
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.policy_step import as_batched_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    obs = observation_from_libero_env(handle.vector_env.envs[0])
    oft = OracleChunkContinuation(client, instruction=instruction, control_env=handle.control_env)
    for t in range(max_steps):
        action = oft.act(obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "Guided",
                    "drop": _object_drop_check(handle), "collision": False}
    return {"success": False, "steps": max_steps, "arm": "Guided",
            "drop": _object_drop_check(handle), "collision": False}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "runs/route_c_final/protocol_frozen.json")
    parser.add_argument("--f0-ckpt", type=Path,
                        default=ROOT / "runs/route_c_controls/F0/plugin_best.pt")
    parser.add_argument("--f2-ckpt", type=Path,
                        default=ROOT / "runs/route_c_controls/F2/plugin_best.pt")
    parser.add_argument("--f0-vector", type=Path,
                        default=ROOT / "runs/pre_c0_r0/f0_constant_vector.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n-episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--oft-port", type=int, default=5555)
    parser.add_argument("--snapshot-limit", type=int, default=50)
    parser.add_argument("--skip-guided", action="store_true",
                        help="Skip Guided (OFT) arm to speed up")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")

    from rase.collect.forked_rollout import load_smolvla_policy_bundle
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    f0_data = json.loads(args.f0_vector.read_text(encoding="utf-8"))
    f0_c = f0_data["f0_constant_vector_c"]
    print(f"F0 constant vector: |c|={np.linalg.norm(f0_c):.6f}")

    from rase.recovery.residual_plugin import load_plugin
    f2_plugin = load_plugin(str(args.f2_ckpt))
    f2_plugin.eval()
    print(f"F2 plugin loaded: obs_feature_dim={f2_plugin.obs_feature_dim}")

    from rase.collect.smolvla_feature_extractor import SmolVLAFeatureExtractor
    f2_extractor = SmolVLAFeatureExtractor(bundle)

    from rase.oracle.client import OracleClient
    client = OracleClient(f"tcp://127.0.0.1:{args.oft_port}", timeout_ms=60000)
    print(f"OFT client on port {args.oft_port}")

    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.envs.forkable_env import ForkableEnv
    from rase.recovery.action_cache import reset_policy_action_cache, reset_policy_history

    task_ids = protocol["splits"][args.suite]["dev"]
    print(f"Tasks: {task_ids}")

    # ── Phase 1: Collect snapshots ──────────────────────────────
    all_snapshots = []
    snapshot_idx = 0

    for task_id in task_ids[:2]:
        for ep_i in range(args.n_episodes_per_task):
            if len(all_snapshots) >= args.snapshot_limit:
                break
            seed_val = (20260807 * 31 + ep_i * 7) % (2**31)
            init_state = ep_i % 50

            seed_everything(seed_val)
            bundle["policy"].reset()

            handle = make_libero_env_for_task(task_id, init_state_id=init_state,
                                               seed=seed_val, libero_flavor="clean")
            instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")

            result = run_b0_and_collect_snapshots(handle, bundle, instruction, args.max_steps)
            handle.close()
            time.sleep(2.0)

            for snap_data in result["snapshots"]:
                snap_data["task_id"] = task_id
                snap_data["suite"] = args.suite
                snap_data["episode_success"] = result["success"]
                snap_data["seed"] = seed_val
                snap_data["init_state_id"] = init_state
                snap_data["snapshot_id"] = snapshot_idx
                all_snapshots.append(snap_data)
                snapshot_idx += 1

            print(f"  [{ep_i+1}/{args.n_episodes_per_task}] {task_id} "
                  f"s={result['success']} ({result['steps']} steps) "
                  f"snapshots={len(result['snapshots'])} "
                  f"(total: {len(all_snapshots)})")

        if len(all_snapshots) >= args.snapshot_limit:
            break

    print(f"\nPhase 1: {len(all_snapshots)} snapshots from {ep_i+1} episodes")

    # ── Phase 2: Run repair arms ────────────────────────────────
    arm_order = [
        ("Base", lambda h, s: run_repair_base(h, bundle, s, args.max_steps)),
        ("F0", lambda h, s: run_repair_f0(h, bundle, f0_c, s, args.max_steps)),
        ("F2", lambda h, s: run_repair_f2(h, bundle, f2_plugin, f2_extractor, s, args.max_steps)),
    ]
    if not args.skip_guided:
        arm_order.append(("Guided", lambda h, s: run_repair_guided(h, client, s, args.max_steps)))

    matrix_path = output_dir / "counterfactual_matrix.jsonl"
    with open(matrix_path, "w") as mf:
        for i, snap_data in enumerate(all_snapshots):
            print(f"\n[{i+1}/{len(all_snapshots)}] Snapshot {snap_data['snapshot_id']} "
                  f"t={snap_data['t']} task={snap_data['task_id']}")

            row = {
                "snapshot_id": snap_data["snapshot_id"],
                "task_id": snap_data["task_id"],
                "suite": snap_data["suite"],
                "snapshot_type": snap_data.get("type", "unknown"),
                "snapshot_t": snap_data["t"],
                "episode_success": snap_data["episode_success"],
                "seed": snap_data["seed"],
                "init_state_id": snap_data["init_state_id"],
                "outcomes": {},
            }

            for arm_name, arm_fn in arm_order:
                seed_everything(snap_data["seed"])
                handle = make_libero_env_for_task(
                    snap_data["task_id"],
                    init_state_id=snap_data["init_state_id"],
                    seed=snap_data["seed"],
                    libero_flavor="clean",
                )
                forkable = ForkableEnv(handle.control_env)
                forkable.restore(snap_data["snapshot"])
                bundle["policy"].reset()
                reset_policy_action_cache(bundle["policy"])
                reset_policy_history(bundle["policy"])

                instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")

                arm_result = arm_fn(handle, instruction)
                handle.close()
                time.sleep(2.0)

                row["outcomes"][arm_name] = arm_result
                status = "SUCCESS" if arm_result["success"] else "FAIL"
                print(f"    {arm_name}: {status} ({arm_result['steps']} steps)")

            mf.write(json.dumps(row, default=str) + "\n")
            mf.flush()

    print(f"\nMatrix: {matrix_path} ({len(all_snapshots)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

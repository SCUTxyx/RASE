#!/usr/bin/env python3
"""Route C counterfactual data collection.

For each R0 episode, replay from the beginning with two branches:

  Branch A: student continuation (SmolVLA only)
  Branch B: plugin takeover (student + Plugin forced from boundary)

Each branch creates a fresh environment with the same (task_id, init_state_id,
seed) and runs the full episode. Branch B's Plugin takeover is forced from
the boundary step indicated by the R0 data.

Then classify each boundary state as rescue/harm/neutral/both-fail based
on the two outcomes. These outcome-grounded labels are the supervision
signal for Selector v2 training.

Note: SmolVLA flow-matching is stochastic, so the student_rollin in each
branch may differ slightly. The ForkableEnv approach (exact state snapshot)
is not compatible with the "clean" LIBERO flavor used for R0 collection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.smolvla_feature_extractor import (
    build_feature_vector,
    SmolVLAFeatureExtractor,
)
from rase.recovery.residual_plugin import load_plugin


# --- helpers ----------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _proprio(obs: dict) -> np.ndarray:
    p = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
    q = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
    return np.concatenate([p, q])[:7].astype(np.float32)


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


# --- branch runners ---------------------------------------------------

def run_student_episode(handle, bundle, instruction, max_steps):
    """Run full episode: SmolVLA student only."""
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1}
    return {"success": False, "steps": max_steps}


def run_plugin_episode(handle, bundle, plugin, instruction, max_steps,
                       boundary_step, boundary_history, boundary_obs_feat):
    """Full episode: student until boundary_step, then Plugin takeover forced."""
    ACTION_DIM = 7
    HISTORY_WINDOW = 8
    MIX_RAMP = [0.0, 0.3, 0.6, 1.0]
    DELTA_CLIP = 0.5
    ACTION_RATE_LIMIT = 0.1

    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)

    history_buffer = list(boundary_history) if boundary_history else []
    last_action = None
    takeover_count = 0

    for t in range(max_steps):
        student_action = select_env_action(bundle, obs, task=instruction)
        progress_val = _progress(handle.control_env)
        proprio_val = _proprio(obs)

        if t < boundary_step:
            action = student_action
        else:
            if t == boundary_step:
                takeover_count = 0

            hist_arr = np.zeros((HISTORY_WINDOW, 8 + 7 + 1 + 7), dtype=np.float32)
            recent = history_buffer[-HISTORY_WINDOW:]
            for hi, h in enumerate(recent):
                p = np.asarray(h["proprio"], dtype=np.float32).flatten()
                a = np.asarray(h["student_action"], dtype=np.float32).flatten()
                p_pad = np.zeros(8, dtype=np.float32)
                a_pad = np.zeros(ACTION_DIM, dtype=np.float32)
                p_pad[:min(len(p), 8)] = p[:8]
                a_pad[:min(len(a), 7)] = a[:ACTION_DIM]
                idx = hi + HISTORY_WINDOW - len(recent)
                if idx >= 0:
                    hist_arr[idx] = np.concatenate([
                        p_pad, a_pad, [float(h.get("progress", 0.0))], a_pad])

            if t == boundary_step and boundary_obs_feat is not None:
                obs_feat = np.asarray(boundary_obs_feat, dtype=np.float32).flatten()[:144]
            else:
                obs_feat = build_feature_vector(
                    smolvla_latent=None, proprio=proprio_val,
                    student_action=student_action.flatten(),
                    stagnation_length=0, progress_delta=0.0,
                    feature_level="F0")

            delta = plugin.predict_delta(hist_arr, obs_feat, student_action.flatten())
            g = MIX_RAMP[min(takeover_count, len(MIX_RAMP) - 1)]
            delta_c = np.clip(delta, -DELTA_CLIP, DELTA_CLIP)
            mixed = np.clip(student_action.flatten() + g * delta_c, -1.0, 1.0)
            if last_action is not None:
                mixed = np.clip(mixed, last_action - ACTION_RATE_LIMIT,
                                last_action + ACTION_RATE_LIMIT)
            last_action = mixed.copy()
            takeover_count += 1
            action = mixed.reshape(1, -1)

        history_buffer.append({
            "proprio": proprio_val,
            "student_action": student_action.flatten(),
            "progress": float(progress_val),
        })
        if len(history_buffer) > HISTORY_WINDOW * 2:
            history_buffer = history_buffer[-HISTORY_WINDOW * 2:]

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            return {"success": success_from_info(info), "steps": t + 1,
                    "takeover_steps": takeover_count}
    return {"success": False, "steps": max_steps, "takeover_steps": takeover_count}


# --- classification ---------------------------------------------------

def classify_outcome(student_result, plugin_result):
    s_ok = student_result["success"]
    p_ok = plugin_result["success"]
    if not s_ok and p_ok:
        return {"label": 1, "category": "rescue"}
    elif s_ok and not p_ok:
        return {"label": -1, "category": "harm"}
    elif not s_ok and not p_ok:
        return {"label": 0, "category": "both_fail"}
    else:
        return {"label": 0, "category": "both_ok"}


# --- main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plugin-ckpt", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    if not args.plugin_ckpt.is_file():
        print(f"ERROR: Plugin checkpoint not found: {args.plugin_ckpt}")
        return 1
    plugin = load_plugin(str(args.plugin_ckpt))
    plugin.eval()
    print(f"Loaded plugin from {args.plugin_ckpt}")

    r0_dir = args.r0_dir / "R0" if (args.r0_dir / "R0").is_dir() else args.r0_dir
    r0_files = sorted(r0_dir.glob("*.json"))
    if not r0_files:
        print(f"ERROR: No R0 files found in {r0_dir}")
        return 1

    output_jsonl = output_dir / "counterfactual_labels.jsonl"
    completed = set()
    if args.skip_existing and output_jsonl.is_file():
        try:
            with open(output_jsonl) as f:
                for line in f:
                    completed.add(json.loads(line).get("episode_id", ""))
            print(f"Found {len(completed)} already done, will skip")
        except Exception:
            pass

    print(f"Running counterfactuals on {len(r0_files)} episodes")

    results = []
    stats = {"rescue": 0, "harm": 0, "both_fail": 0, "both_ok": 0, "error": 0}

    for ep_idx, ep_path in enumerate(r0_files):
        ep_data = json.loads(ep_path.read_text(encoding="utf-8"))
        episode_id = ep_path.stem

        if episode_id in completed:
            print(f"  [{ep_idx+1}/{len(r0_files)}] {episode_id} (skip)")
            continue

        suite = ep_data["suite"]
        task_id = ep_data["task_id"]
        episode_seed = ep_data["episode_seed"]
        boundary_step = ep_data["boundary_step"]
        init_state_id = episode_seed % 50

        teacher_recovery = ep_data.get("teacher_recovery", [])
        if not teacher_recovery:
            print(f"  [{ep_idx+1}/{len(r0_files)}] {episode_id} NO recovery, skip")
            continue

        boundary_rec = teacher_recovery[0]
        boundary_obs_feat = np.asarray(boundary_rec["obs_features"], dtype=np.float32)
        boundary_history_raw = boundary_rec.get("history_before", [])

        boundary_history = []
        if boundary_history_raw:
            for row in boundary_history_raw:
                boundary_history.append({
                    "proprio": np.asarray(row[:8], dtype=np.float32),
                    "student_action": np.asarray(row[8:15], dtype=np.float32),
                    "progress": float(row[15]) if len(row) > 15 else 0.0,
                })

        student_action_at_boundary = np.asarray(
            boundary_rec["action"], dtype=np.float32).flatten()[:7]

        print(f"  [{ep_idx+1}/{len(r0_files)}] {episode_id} "
              f"task={task_id} s={episode_seed} b={boundary_step}")

        try:
            # Branch A: student only
            seed_everything(episode_seed)
            bundle["policy"].reset()
            handle_a = make_libero_env_for_task(
                task_id, init_state_id=init_state_id, seed=episode_seed,
                libero_flavor="clean")
            instruction = str(getattr(
                handle_a.vector_env.envs[0], "task_description", "") or "")
            student_result = run_student_episode(
                handle_a, bundle, instruction, args.max_steps)
            handle_a.close()

            time.sleep(1.0)

            # Branch B: plugin takeover
            seed_everything(episode_seed)
            bundle["policy"].reset()
            handle_b = make_libero_env_for_task(
                task_id, init_state_id=init_state_id, seed=episode_seed,
                libero_flavor="clean")
            plugin_result = run_plugin_episode(
                handle_b, bundle, plugin, instruction, args.max_steps,
                boundary_step, boundary_history, boundary_obs_feat)
            handle_b.close()

            outcome = classify_outcome(student_result, plugin_result)
            stats[outcome["category"]] += 1

            rec = {
                "episode_id": episode_id,
                "suite": suite,
                "task_id": task_id,
                "episode_seed": episode_seed,
                "init_state_id": init_state_id,
                "boundary_step": boundary_step,
                "student_success": student_result["success"],
                "student_steps": student_result["steps"],
                "plugin_success": plugin_result["success"],
                "plugin_steps": plugin_result["steps"],
                "plugin_takeover_steps": plugin_result.get("takeover_steps", 0),
                "label": outcome["label"],
                "category": outcome["category"],
                "boundary_obs_features": boundary_obs_feat[:144].tolist(),
                "student_action_at_boundary": student_action_at_boundary.tolist(),
                "boundary_history": [[float(v) for v in row]
                                     for row in boundary_history_raw[:8]]
                    if boundary_history_raw else [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(rec)

            ls = {-1: "HARM", 1: "RESCUE", 0: "neutral"}[outcome["label"]]
            print(f"    student={student_result['success']}({student_result['steps']}) "
                  f"plugin={plugin_result['success']}({plugin_result['steps']}) "
                  f"-> {outcome['category']} [{ls}]")

        except Exception as exc:
            print(f"    ERROR: {exc}")
            stats["error"] += 1

    with open(output_jsonl, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_rescue = sum(1 for r in results if r["label"] == 1)
    n_harm = sum(1 for r in results if r["label"] == -1)
    n_total = len(results)

    summary = {
        "total": n_total,
        "rescue": n_rescue,
        "harm": n_harm,
        "both_fail": stats["both_fail"],
        "both_ok": stats["both_ok"],
        "error": stats["error"],
        "usable_labels": n_rescue + n_harm,
        "gate_pass": (n_rescue + n_harm) >= 10 or n_total >= len(r0_files) * 0.5,
    }

    (output_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nCounterfactual collection complete.")
    print(f"  Total: {n_total}  Rescue: {n_rescue}  Harm: {n_harm}")
    print(f"  Both-fail: {stats['both_fail']}  Both-ok: {stats['both_ok']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Usable: {n_rescue + n_harm} (gate: {'PASS' if summary['gate_pass'] else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

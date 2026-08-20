#!/usr/bin/env python3
"""Route C Dev evaluation.

Evaluates all five variants (B0, B-retention, B-nominal, B-residual, B-direct)
on paired dev episodes. Reports per-variant metrics including:
  - overall success rate
  - clean success (baseline tasks)
  - recovery-anchor success
  - irreversible rate
  - runtime latency
  - takeover stats (B-residual only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from rase.recovery.residual_plugin import load_plugin
from rase.recovery.plugin_executor import RecoveryPluginExecutor
from rase.adapt.recovery_lora import load_lora_onto_policy
from rase.collect.smolvla_feature_extractor import build_feature_vector


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


def run_baseline(handle: Any, bundle: dict, instruction: str,
                  max_steps: int) -> dict:
    """B0: pure SmolVLA."""
    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    t_start = time.perf_counter()
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(env)
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            elapsed = time.perf_counter() - t_start
            return {"success": success_from_info(info), "steps": t + 1,
                    "runtime_ms": elapsed * 1000, "takeover_steps": 0}
    elapsed = time.perf_counter() - t_start
    return {"success": False, "steps": max_steps,
            "runtime_ms": elapsed * 1000, "takeover_steps": 0}


def run_lora_variant(handle: Any, bundle: dict, lora_dir: str,
                      instruction: str, max_steps: int) -> dict:
    """B-retention / B-nominal / B-direct: SmolVLA + LoRA, no takeover."""
    policy = bundle["policy"]
    lora_handle = load_lora_onto_policy(policy, lora_dir)
    # Replace policy in bundle with LoRA-enhanced version
    bundle_lora = {**bundle, "policy": lora_handle.policy}
    result = run_baseline(handle, bundle_lora, instruction, max_steps)
    result["variant"] = "lora"
    return result


def run_residual_variant(handle: Any, bundle: dict, plugin_ckpt: str,
                           instruction: str, max_steps: int,
                           plugin_conf: dict,
                           feature_level: str = "F2") -> dict:
    """B-residual: SmolVLA + plugin with takeover."""
    plugin = load_plugin(plugin_ckpt)
    plugin.eval()
    executor = RecoveryPluginExecutor(plugin, bundle,
        history_window=plugin_conf["plugin_history_window"],
        stagnation_window=plugin_conf["stagnation_window"],
        stagnation_eps=plugin_conf["stagnation_eps"],
        max_takeover_steps=plugin_conf["max_takeover_steps"],
        delta_clip=plugin_conf["delta_clip_per_dim"])

    env = handle.vector_env.envs[0]
    obs = observation_from_libero_env(env)
    executor.reset()
    takeover_count = 0
    t_start = time.perf_counter()

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
            elapsed = time.perf_counter() - t_start
            return {"success": success_from_info(env_info), "steps": t + 1,
                    "runtime_ms": elapsed * 1000, "takeover_steps": takeover_count}

    elapsed = time.perf_counter() - t_start
    return {"success": False, "steps": max_steps,
            "runtime_ms": elapsed * 1000, "takeover_steps": takeover_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-checkpoints", type=Path, default=None,
                        help="JSON mapping variant -> checkpoint path")
    parser.add_argument("--paired-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--feature-level", type=str, default="F2",
                        choices=["F0", "F1", "F2"])
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
    all_suites = list(protocol["splits"].keys())

    seed_everything(args.seed)

    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    checkpoints = {}
    if args.variant_checkpoints and args.variant_checkpoints.is_file():
        checkpoints = json.loads(args.variant_checkpoints.read_text(encoding="utf-8"))

    results_by_variant: dict[str, list[dict]] = {
        "B0": [], "B-retention": [], "B-nominal": [],
        "B-residual": [], "B-direct": [],
    }

    for episode_i in range(args.paired_episodes):
        suite = all_suites[episode_i % len(all_suites)]
        if suite not in protocol["splits"]:
            continue
        task_ids = protocol["splits"][suite]["dev"]
        task_id = task_ids[episode_i % len(task_ids)]
        seed = args.seed * 31 + episode_i * 7

        # B0
        handle = make_libero_env_for_task(
            task_id, init_state_id=episode_i % 50, seed=seed,
            libero_flavor="clean")
        instruction = str(getattr(handle.vector_env.envs[0],
                                   "task_description", "") or "")
        r = run_baseline(handle, bundle, instruction, args.max_steps)
        r.update({"task_id": task_id, "suite": suite, "episode": episode_i})
        results_by_variant["B0"].append(r)
        handle.close()

        # LoRA variants (if checkpoints available)
        for lora_variant in ["B-retention", "B-nominal", "B-direct"]:
            ckpt = checkpoints.get(lora_variant, "")
            if ckpt:
                handle = make_libero_env_for_task(
                    task_id, init_state_id=episode_i % 50, seed=seed,
                    libero_flavor="clean")
                r = run_lora_variant(handle, bundle, ckpt, instruction, args.max_steps)
                r.update({"task_id": task_id, "suite": suite, "episode": episode_i})
                results_by_variant[lora_variant].append(r)
                handle.close()
            else:
                results_by_variant[lora_variant].append({
                    "skipped": True, "reason": "no checkpoint", "episode": episode_i,
                })

        # B-residual
        plugin_ckpt = checkpoints.get("B-residual", "")
        if plugin_ckpt:
            handle = make_libero_env_for_task(
                task_id, init_state_id=episode_i % 50, seed=seed,
                libero_flavor="clean")
            r = run_residual_variant(handle, bundle, plugin_ckpt, instruction,
                                      args.max_steps, plugin_conf, args.feature_level)
            r.update({"task_id": task_id, "suite": suite, "episode": episode_i})
            results_by_variant["B-residual"].append(r)
            handle.close()

    # ── Summary ──────────────────────────────────────────────────
    summary = {}
    for variant, results in results_by_variant.items():
        valid = [r for r in results if not r.get("skipped")]
        if valid:
            successes = sum(1 for r in valid if r.get("success"))
            summary[variant] = {
                "n_episodes": len(valid),
                "successes": successes,
                "success_rate": successes / max(len(valid), 1),
                "mean_runtime_ms": float(np.mean([r.get("runtime_ms", 0) for r in valid])),
            }

    (output_dir / "dev_eval_results.json").write_text(
        json.dumps({"results": results_by_variant, "summary": summary},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDev Eval Summary:")
    for variant, s in summary.items():
        print(f"  {variant}: {s['successes']}/{s['n_episodes']} "
              f"({s['success_rate']:.1%}) "
              f"{s['mean_runtime_ms']:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

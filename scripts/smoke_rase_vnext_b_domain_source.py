#!/usr/bin/env python3
"""B-domain feasibility smoke: pi0-fast source-only behaviour on libero_90.

Verifies: (1) env construction for libero_90 (KITCHEN_SCENE tasks);
(2) pi0-fast produces valid actions (no crash, finite, env steps);
(3) source success rate is low enough to qualify as a harder domain.
Fallback (OFT) is NOT used here — its adapter is libero-suite specific.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states  # noqa: E402
from rase.backends.libero_plus_paths import ensure_libero_plus_paths  # noqa: E402
from rase.collect.forked_rollout import load_lerobot_policy_bundle  # noqa: E402
from rase.collect.policy_step import as_batched_action  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--tasks", type=int, default=4)
    args = parser.parse_args()

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()

    from libero.libero import benchmark as libero_benchmark

    tasks = json.loads(args.tasks_file.read_text())[: args.tasks]
    suite_cls = libero_benchmark.get_benchmark_dict()["libero_90"]
    suite = suite_cls()
    task_names = [str(task.name) for task in suite.tasks]
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda", num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    policy = bundle["policy"]

    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv

    def make_env(task_index: int):
        def make_single():
            return LiberoEnv(
                task_suite=suite, task_id=task_index, task_suite_name="libero_90",
                camera_name="agentview_image,robot0_eye_in_hand_image",
                init_states=True, episode_index=0, n_envs=1,
                obs_type="pixels_agent_pos", observation_height=360,
                observation_width=360, control_mode="relative",
            )
        return gym.vector.SyncVectorEnv([make_single])

    report: dict = {"schema_version": "b-domain-source-smoke/v1", "tasks": {}}
    for task_name in tasks:
        try:
            task_index = task_names.index(task_name)
        except ValueError:
            report["tasks"][task_name[:40]] = {"error": "task not in suite"}
            continue
        try:
            vector_env = make_env(task_index)
            observation, _ = vector_env.reset(seed=[0])
            single = vector_env.envs[0]
            instruction = str(getattr(single, "task_description", ""))
            from rase.collect.pool_candidates import observation_from_libero_env
            from rase.collect.forked_rollout import InProcessLeRobotContinuation

            cont = InProcessLeRobotContinuation(bundle, seed=7, capture=False)
            obs = observation_from_libero_env(single)
            prefix_steps = 0
            success = False
            stop_reason = "max_steps"
            actions_valid = True
            for step in range(args.prefix_steps + args.max_steps):
                try:
                    action = cont.act(obs, task=instruction)
                except Exception as exc:
                    actions_valid = False
                    stop_reason = f"policy_error:{type(exc).__name__}:{str(exc)[:100]}"
                    break
                if not np.isfinite(action).all():
                    actions_valid = False
                    stop_reason = "non_finite_action"
                    break
                obs, _, term, trunc, info = vector_env.step(as_batched_action(action))
                prefix_steps += 1
                if prefix_steps == args.prefix_steps:
                    # continue rolling to terminal to measure source success
                    pass
                terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
                if terminal:
                    from rase.collect.policy_step import success_from_info
                    success = bool(success_from_info(info))
                    stop_reason = "success" if success else "terminal_failure"
                    break
            report["tasks"][task_name[:50]] = {
                "instruction": instruction[:60],
                "prefix_steps": prefix_steps,
                "success": success,
                "stop_reason": stop_reason,
                "actions_valid": actions_valid,
            }
            vector_env.close()
        except Exception as exc:
            report["tasks"][task_name[:40]] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

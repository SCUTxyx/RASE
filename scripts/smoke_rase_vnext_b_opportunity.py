#!/usr/bin/env python3
"""B-domain opportunity smoke: continue vs requery on libero_90 (no OFT).

For each task: run pi0-fast prefix to step 8 (deterministic, fixed seed),
restore the boundary snapshot, then execute continue and requery candidates
with K=3 matched seeds.  Reports per-candidate success and the heterogeneous
rate (candidates disagree on success) — the minimal signal required for the
selector idea on a harder domain.
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
from rase.collect.policy_step import as_batched_action, success_from_info  # noqa: E402
from rase.envs.forkable_env import ForkableEnv  # noqa: E402


def stable_seed(*parts: object) -> int:
    import hashlib
    token = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()

    from libero.libero import benchmark as libero_benchmark
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv

    tasks = json.loads(args.tasks_file.read_text())[: args.tasks]
    suite_cls = libero_benchmark.get_benchmark_dict()["libero_90"]
    suite = suite_cls()
    task_names = [str(t.name) for t in suite.tasks]
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda", num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    from rase.collect.forked_rollout import InProcessLeRobotContinuation
    from rase.collect.pool_candidates import observation_from_libero_env

    report: dict = {"schema_version": "b-domain-opportunity-smoke/v1", "tasks": {}}
    for task_name in tasks:
        try:
            task_index = task_names.index(task_name)
        except ValueError:
            report["tasks"][task_name[:40]] = {"error": "task not in suite"}
            continue
        entry: dict = {"instruction": "", "candidates": {}, "verdict": None}
        try:
            def make_single():
                return LiberoEnv(
                    task_suite=suite, task_id=task_index, task_suite_name="libero_90",
                    camera_name="agentview_image,robot0_eye_in_hand_image",
                    init_states=True, episode_index=0, n_envs=1,
                    obs_type="pixels_agent_pos", observation_height=360,
                    observation_width=360, control_mode="relative",
                )
            vector_env = gym.vector.SyncVectorEnv([make_single])
            single = vector_env.envs[0]
            entry["instruction"] = str(getattr(single, "task_description", ""))[:70]
            forkable = ForkableEnv(single._env)
            vector_env.reset(seed=[0])
            snapshot0 = forkable.snapshot()

            def rollout_from_snapshot(seed: int) -> tuple[bool, str]:
                forkable.restore(snapshot0, check_task_fingerprint=False)
                vector_env.reset(seed=[seed])
                obs = observation_from_libero_env(single)
                instruction = str(getattr(single, "task_description", ""))
                cont = InProcessLeRobotContinuation(bundle, seed=seed, capture=False)
                steps = 0
                success = False
                stop = "max_steps"
                while steps < args.prefix_steps + args.max_steps:
                    action = cont.act(obs, task=instruction)
                    obs, _, term, trunc, info = vector_env.step(as_batched_action(action))
                    steps += 1
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        success = bool(success_from_info(info))
                        stop = "success" if success else "terminal_failure"
                        break
                return success, stop

            # continue: same policy, same seed chain (replay prefix then continue)
            # requery: fresh seed (independent sampling path)
            for operator, seed_salt in (("continue.source", "cont"), ("requery.source", "req")):
                successes: list[bool] = []
                stops: list[str] = []
                for rep in range(args.repeats):
                    seed = stable_seed("b-opp", task_name, seed_salt, rep)
                    success, stop = rollout_from_snapshot(seed)
                    successes.append(success)
                    stops.append(stop)
                entry["candidates"][operator] = {"success": successes, "stop": stops}
            cont_ok = any(entry["candidates"]["continue.source"]["success"])
            req_ok = any(entry["candidates"]["requery.source"]["success"])
            entry["verdict"] = (
                "heterogeneous" if (cont_ok != req_ok) else
                ("all_fail" if not cont_ok and not req_ok else "both_succeed")
            )
            vector_env.close()
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        report["tasks"][task_name[:50]] = entry

    report["summary"] = {
        "heterogeneous": sum(1 for t in report["tasks"].values() if t.get("verdict") == "heterogeneous"),
        "all_fail": sum(1 for t in report["tasks"].values() if t.get("verdict") == "all_fail"),
        "both_succeed": sum(1 for t in report["tasks"].values() if t.get("verdict") == "both_succeed"),
        "total": len(report["tasks"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

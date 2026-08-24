#!/usr/bin/env python3
"""G0: π0.5 fixed-horizon baseline on LIBERO-PRO perturbation domains.

Protocol: for a given perturbation type (object/lan/swap/task) and suite,
load LIBERO-PRO bddl/init via set_libero_path, run frozen π0.5 (fixed
n_action_steps=10) to terminal, report per-task and aggregate success.

Usage (server, smolvla env):
  python scripts/g0_pro_baseline.py \
    --pert object --suite libero_object --tasks 10 --eps-per-task 8 \
    --policy ckpts/pi05_libero --output runs/g0_pro_object_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero.libero.utils import set_libero_path, get_libero_path
from rase.collect.forked_rollout import load_lerobot_policy_bundle, InProcessLeRobotContinuation
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.policy_step import as_batched_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env

SCHEMA = "rase-g0-pro-baseline/v1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pert", required=True, choices=["object", "lan", "swap", "task"])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--eps-per-task", type=int, default=8)
    ap.add_argument("--policy", default="ckpts/pi05_libero")
    ap.add_argument("--tokenizer", default="ckpts/paligemma_tokenizer_35e4f46")
    ap.add_argument("--pro-root", default="/root/autodl-tmp/libero_pro_root_")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = f"{args.pro_root}{args.pert}"
    old = get_libero_path("bddl_files")
    set_libero_path(root)
    print(f"[g0] libero root -> {root}", flush=True)
    try:
        bundle = load_lerobot_policy_bundle(
            args.policy, device="cuda", num_steps=10, n_action_steps=10,
            tokenizer_path=args.tokenizer,
            observation_height=360, observation_width=360)
        results = []
        t0 = time.time()
        for ti in range(1, args.tasks + 1):
            task_id = f"{args.suite}_{ti:06d}"
            ok = 0
            per_ep = []
            for ep in range(args.eps_per_task):
                h = make_libero_env_for_task(
                    task_id, init_state_id=ep % 10, seed=1000 + ep,
                    observation_height=360, observation_width=360,
                    libero_clean_root="/root/autodl-tmp/src/LIBERO",
                    libero_flavor="clean")
                try:
                    single = h.vector_env.envs[0]
                    task = str(single.task_description)
                    horizon = int(getattr(single, "_max_episode_steps", 600))
                    cont = InProcessLeRobotContinuation(bundle, seed=2000 + ep)
                    obs = observation_from_libero_env(single)
                    t = 0
                    success = False
                    stop = "horizon"
                    while t < horizon:
                        action = cont.act(obs, task=task)
                        obs, _, term, trunc, info = h.vector_env.step(as_batched_action(action))
                        t += 1
                        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                            success = bool(success_from_info(info))
                            stop = "success" if success else "terminal_failure"
                            break
                    ok += int(success)
                    per_ep.append({"ep": ep, "success": bool(success), "steps": t, "stop": stop})
                finally:
                    h.close()
            results.append({"task_id": task_id, "success": ok,
                            "episodes": args.eps_per_task, "per_episode": per_ep})
            el = time.time() - t0
            print(f"[g0] {task_id}: {ok}/{args.eps_per_task}  elapsed={el/60:.1f}m", flush=True)
        total_ok = sum(r["success"] for r in results)
        total_n = sum(r["episodes"] for r in results)
        report = {
            "schema_version": SCHEMA,
            "perturbation": args.pert, "suite": args.suite,
            "policy": args.policy,
            "n_tasks": len(results), "n_episodes": total_n,
            "successes": total_ok, "success_rate": total_ok / total_n,
            "elapsed_s": time.time() - t0,
            "per_task": [{k: r[k] for k in ("task_id", "success", "episodes")}
                         for r in results],
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in (
            "perturbation", "suite", "n_tasks", "successes", "success_rate")},
            indent=2), flush=True)
        return 0
    finally:
        set_libero_path(old)


if __name__ == "__main__":
    raise SystemExit(main())

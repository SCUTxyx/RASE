#!/usr/bin/env python3
"""R0b: replan-frequency x execution-noise experiment (LIBERO-Long, π0-fast).

Question: does more frequent replanning help when execution is noisy?
  k=4 (replan every 4 steps) vs k=10 (native chunk, replan every 10 steps)
  sigma=0 (clean) vs sigma=0.05 (action noise on pos/rot dims)

Mechanism: with execution noise, a chunk's open-loop rollout drifts from the
observation it was planned on; frequent replanning re-conditions on fresh
observations.  If sigma>0 improves k=4 over k=10, execution-verification /
early-intervention (detect drift -> replan) has real value on this stack.

Gate (preregistered): gain(k=4 over k=10) at sigma=0.05 >= 3pp AND
gain at sigma=0 near 0 (clean sanity: replan frequency alone ~neutral).

Usage (server, smolvla env; GPU free):
  python scripts/e6_replan_freq.py \
    --config configs/g2a_pi0fast_clean_long_v1.json \
    --tasks 1,2,9 --init-states 0,1,2,3 \
    --sigmas 0,0.05 --ks 4,10 \
    --output runs/e6_replan_freq_v1
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
from rase.collect.forked_rollout import load_lerobot_policy_bundle
from rase.collect.policy_step import as_batched_action, current_timestep
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import InProcessLeRobotContinuation

SCHEMA = "rase-e6-replan-freq/v1"


def run_episode(bundle, protocol, record, sigma, replan_k, seed, horizon_cap=400) -> dict:
    """Direct policy loop: predict native chunk, execute k steps, replan.

    Uses predict_action_chunk + manual queue (avoids continuation overhead);
    noise is added to pos/rot dims of executed actions.
    """
    from lerobot.envs.utils import preprocess_observation
    from rase.collect.policy_step import success_from_info
    from rase.collect.candidates import seed_everything
    handle = make_libero_env_for_task(
        str(record["task_id"]), init_state_id=int(record["init_state_id"]),
        seed=int(record["environment_seed"]), observation_height=360,
        observation_width=360, libero_clean_root=os.environ.get("LIBERO_CLEAN_ROOT"),
        libero_flavor="clean")
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        horizon = int(getattr(single, "_max_episode_steps", 600))
        if horizon_cap and horizon > horizon_cap:
            horizon = horizon_cap
        policy = bundle["policy"]
        obs = observation_from_libero_env(single)
        rng = np.random.default_rng(seed)
        seed_everything(seed)
        queue: list[np.ndarray] = []
        t = 0
        success = False
        stop = "horizon"
        steps = 0
        replans = 0
        while t < horizon:
            if len(queue) == 0:
                policy_observation = preprocess_observation(
                    {key: value for key, value in obs.items() if key != "task"})
                policy_observation["task"] = [task]
                env_observation = bundle["env_preprocessor"](policy_observation)
                processed = bundle["preprocessor"](env_observation)
                chunk = policy.predict_action_chunk(processed)
                chunk = bundle["postprocessor"](chunk)
                chunk = chunk.detach().cpu().numpy().reshape(-1, 7)
                queue = [chunk[i] for i in range(len(chunk))]
                replans += 1
            a = np.asarray(queue.pop(0), dtype=np.float64).reshape(-1)
            if sigma > 0:
                a = a.copy()
                a[:6] = a[:6] + rng.normal(0.0, sigma, 6)
            obs, _, term, trunc, info = handle.vector_env.step(as_batched_action(a))
            t += 1
            steps += 1
            if replan_k and (t % replan_k) == 0:
                queue = []  # force replan on next step
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                success = bool(success_from_info(info))
                stop = "success" if success else "terminal_failure"
                break
        return {"success": success, "stop": stop, "steps": steps, "replans": replans}
    finally:
        handle.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tasks", default="1,2,9")
    ap.add_argument("--init-states", default="0,1,2,3")
    ap.add_argument("--sigmas", default="0,0.05")
    ap.add_argument("--ks", default="4,10")
    ap.add_argument("--horizon-cap", type=int, default=400,
                    help="cap episode horizon (fair across arms; faster)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--policy-path", default="ckpts/pi0fast_libero")
    ap.add_argument("--tokenizer-path", default="ckpts/paligemma_tokenizer_35e4f46")
    ap.add_argument("--action-tokenizer-path", default="ckpts/pi0fast_action_tokenizer_79ae83e")
    args = ap.parse_args()

    protocol = json.loads(args.config.read_text())
    task_ids = [int(x) for x in args.tasks.split(",")]
    init_ids = [int(x) for x in args.init_states.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    records = [r for r in protocol["records"]
               if int(r["clean_task_index"]) in task_ids
               and int(r["init_state_id"]) in init_ids]

    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda",
        num_steps=int(protocol["num_steps"]),
        n_action_steps=int(protocol["n_action_steps"]),
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )

    results = {}
    for sigma in sigmas:
        for k in ks:
            key = f"sigma{sigma}_k{k}"
            out = []
            for idx, rec in enumerate(records):
                seed = int(rec["policy_seed"]) + 777 * (idx + 1)
                res = run_episode(bundle, protocol, rec, sigma, k, seed,
                                  horizon_cap=args.horizon_cap)
                out.append({**{kk: rec[kk] for kk in ("episode_id", "task_id")},
                            **res})
                print(f"[e6] {key} {rec['episode_id']} -> {res['success']}", flush=True)
            succ = [o["success"] for o in out]
            results[key] = {
                "n": len(out), "successes": sum(succ),
                "rate": sum(succ) / len(out) if out else None,
                "per_episode": out,
            }

    report = {"schema": SCHEMA, "results": {}}
    for key, r in results.items():
        report["results"][key] = {kk: r[kk] for kk in ("n", "successes", "rate")}
    # gate
    g0 = report["results"].get("sigma0_k4", {}).get("rate")
    g1 = report["results"].get("sigma0_k10", {}).get("rate")
    n0 = report["results"].get("sigma0.05_k4", {}).get("rate")
    n1 = report["results"].get("sigma0.05_k10", {}).get("rate")
    clean_gain = (g0 - g1) * 100 if g0 is not None and g1 is not None else None
    noisy_gain = (n0 - n1) * 100 if n0 is not None and n1 is not None else None
    report["gate"] = {
        "clean_gain_pp_k4_minus_k10": clean_gain,
        "noisy_gain_pp_k4_minus_k10": noisy_gain,
        "verdict": "PASS" if (noisy_gain is not None and noisy_gain >= 3.0
                              and abs(clean_gain or 0) < 3.0) else "FAIL",
        "note": "PASS -> early replan/verification has value under noise; "
                "FAIL -> replan frequency neutral, verification route closed",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

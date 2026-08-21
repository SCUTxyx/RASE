#!/usr/bin/env python3
"""E4-0: π0-fast T=0.7 candidate-pool information audit (RoboMonkey-style
best-of-K feasibility probe) on clean LIBERO Long.

Question: at a decision state, does sampling K=8 chunks with temperature=0.7
produce outcome-divergent candidates?  If yes -> SmolVLM2 verifier (E4-1)
has something to rank; if no -> BOKBO-style negative evidence.

Protocol (per episode):
  1. G2a-style env + greedy (T=0) run to decision point t=10.
  2. Freeze snapshot + obs.
  3. For k in 0..K-1: policy.reset(); seed_everything(seed_k);
     temperature=0.7; capture native chunk (10 steps).
  4. For each k: restore snapshot; execute chunk_k (10 steps); then greedy
     (T=0) continuation to terminal/horizon; record success_k.

Output: runs/e4_candidate_pool_audit/{episodes,summary}.json
Gate: >=2 states with mixed K outcomes AND oracle@8 > best-of-1 (>=3pp trend).

Usage (server, smolvla env):
  export LIBERO_CLEAN_ROOT=...  # same as g2a run script
  python scripts/e4_candidate_pool_audit.py \
    --config configs/g2a_pi0fast_clean_long_v1.json \
    --tasks 1,2,9 --episodes-per-task 2 --k 8 --temperature 0.7 \
    --decision-step 10 \
    --output runs/e4_candidate_pool_audit \
    --smoke-k 2   # smoke mode: 1 episode, K candidates limited
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
from rase.collect.policy_step import (
    as_batched_action, current_timestep, success_from_info,
    select_env_action_with_native_chunk,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.candidates import seed_everything

RESULT_SCHEMA = "rase-e4-candidate-pool-audit/v1"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def make_env(protocol: dict, record: dict):
    from rase.collect.libero_env_factory import make_libero_env_for_task
    handle = make_libero_env_for_task(
        str(record["task_id"]),
        init_state_id=int(record["init_state_id"]),
        seed=int(record["environment_seed"]),
        observation_height=360,
        observation_width=360,
        libero_clean_root=os.environ.get("LIBERO_CLEAN_ROOT"),
        libero_flavor="clean",
    )
    return handle


def get_sim_state(single):
    """LIBERO clean wraps robosuite at single._env (OffScreenRenderEnv)."""
    return single._env.get_sim_state()


def restore_env(single, snapshot) -> dict:
    """Restore simulator snapshot and regenerate obs (collect_same_root style)."""
    rob = single._env
    rob.set_state(snapshot)
    obs = observation_from_libero_env(single)
    try:
        rob.done = False
        rob.timestep = 0
    except Exception:
        pass
    return obs


def force_clear(single) -> None:
    try:
        inner = getattr(single, "env", single)
        inner.done = False
        inner.timestep = 0
    except Exception:
        pass


def sample_chunk_with_temperature(bundle, obs_at, task, temperature, seed, horizon=10):
    """Sample one native chunk with token-level temperature (π0-fast).

    NOTE: predict_action_chunk ignores its kwargs and reads
    ``policy.config.temperature`` (config default is 0.0), so we set the
    config field at runtime per call.  Returns env-space (T, 7) chunk.
    """
    from lerobot.envs.utils import preprocess_observation
    policy = bundle["policy"]
    policy_observation = preprocess_observation(
        {key: value for key, value in obs_at.items() if key != "task"})
    policy_observation["task"] = [task]
    env_observation = bundle["env_preprocessor"](policy_observation)
    processed = bundle["preprocessor"](env_observation)
    policy.reset()
    seed_everything(seed)
    policy.config.temperature = float(temperature)
    chunk = policy.predict_action_chunk(processed)
    chunk = bundle["postprocessor"](chunk)
    chunk = chunk.detach().cpu().numpy().reshape(-1, 7).astype(np.float32)
    if len(chunk) < horizon:
        chunk = np.pad(chunk, ((0, horizon - len(chunk)), (0, 0)))
    return chunk[:horizon]


def run_episode(bundle, protocol, record, args, log) -> dict:
    from rase.collect.forked_rollout import InProcessLeRobotContinuation
    handle = make_env(protocol, record)
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        horizon = int(getattr(single, "_max_episode_steps", 600))
        continuation = InProcessLeRobotContinuation(bundle, seed=int(record["policy_seed"]))
        observation = observation_from_libero_env(single)

        # ---- run greedily to decision step ----
        t = 0
        while t < args.decision_step and t < horizon:
            action = continuation.act(observation, task=task)
            observation, _, term, trunc, _ = handle.vector_env.step(as_batched_action(action))
            t += 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break

        # ---- freeze decision state ----
        snapshot = get_sim_state(single)
        obs_at = observation_from_libero_env(single)

        # ---- sample K candidates (temperature sampling, native chunk capture) ----
        chunks = []
        for k in range(args.k):
            seed = int(record["policy_seed"]) + 1000 * (k + 1) + int(record["clean_task_index"]) * 100000
            chunks.append(sample_chunk_with_temperature(
                bundle, obs_at, task, args.temperature, seed, horizon=args.native_h))

        # ---- roll out each candidate from the same snapshot ----
        outcomes = []
        for k, chunk in enumerate(chunks):
            obs_r = restore_env(single, snapshot)
            success = False
            stop = "horizon"
            steps = 0
            terminal_now = False
            # execute candidate native chunk
            for step in range(min(args.native_h, len(chunk))):
                a = chunk[step]
                obs_r, _, term, trunc, info = handle.vector_env.step(as_batched_action(a))
                steps += 1
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    success = bool(success_from_info(info))
                    stop = "success" if success else "terminal_failure"
                    terminal_now = True
                    break
            # greedy continuation to terminal
            if not terminal_now:
                cont = InProcessLeRobotContinuation(bundle, seed=int(record["policy_seed"]) + 50000 + k)
                obs_c = obs_r
                while current_timestep(handle.control_env) < horizon:
                    try:
                        action = cont.act(obs_c, task=task)
                    except BaseException:
                        stop = "policy_inference_error"
                        break
                    obs_c, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                    steps += 1
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        success = bool(success_from_info(info))
                        stop = "success" if success else "terminal_failure"
                        break
            outcomes.append({"candidate": k, "success": bool(success), "stop": stop, "steps": steps})
            force_clear(single)
            log.write(f"[e4] ep={record['episode_id']} k={k} success={success} stop={stop}\n")
            log.flush()

        # ---- chunk diversity ----
        arr = np.stack(chunks)
        l2 = np.zeros((args.k, args.k))
        for i in range(args.k):
            for j in range(args.k):
                l2[i, j] = float(np.linalg.norm(arr[i] - arr[j]))
        return {
            "schema_version": RESULT_SCHEMA,
            **{k: record[k] for k in ("episode_id", "task_id", "clean_task_index", "init_state_id", "environment_seed", "policy_seed")},
            "decision_step": t,
            "k": args.k,
            "temperature": args.temperature,
            "outcomes": outcomes,
            "successes": [o["success"] for o in outcomes],
            "n_success": sum(o["success"] for o in outcomes),
            "chunk_l2_mean": float(l2[np.triu_indices(args.k, 1)].mean()),
            "chunk_l2_max": float(l2[np.triu_indices(args.k, 1)].max()),
        }
    finally:
        handle.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tasks", default="1,2,9", help="clean_task_index list")
    ap.add_argument("--episodes-per-task", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--native-h", type=int, default=10)
    ap.add_argument("--decision-step", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke-k", type=int, default=0,
                    help="if >0: smoke mode (1 episode, K=smoke_k)")
    ap.add_argument("--policy-path", default="ckpts/pi0fast_libero")
    ap.add_argument("--tokenizer-path", default="ckpts/paligemma_tokenizer_35e4f46")
    ap.add_argument("--action-tokenizer-path", default="ckpts/pi0fast_action_tokenizer_79ae83e")
    args = ap.parse_args()

    protocol = read_json(args.config)
    task_ids = [int(x) for x in args.tasks.split(",")]
    records = [r for r in protocol["records"] if int(r["clean_task_index"]) in task_ids]
    if args.smoke_k > 0:
        records = records[:1]
        args.k = args.smoke_k

    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda",
        num_steps=int(protocol["num_steps"]),
        n_action_steps=int(protocol["n_action_steps"]),
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "e4.log").open("a")
    log.write(f"=== E4 START {time.strftime('%H:%M:%S')} tasks={task_ids} k={args.k} T={args.temperature} ===\n")

    episodes = []
    for rec in records:
        res = run_episode(bundle, protocol, rec, args, log)
        episodes.append(res)
        (args.output / f"episode_{res['episode_id']}.json").write_text(
            json.dumps(res, indent=2, sort_keys=True) + "\n")

    # ---- summary + gate ----
    mixed = [e for e in episodes if 0 < e["n_success"] < e["k"]]
    best_of_1 = np.mean([e["successes"][0] for e in episodes]) if episodes else 0.0
    oracle = np.mean([1.0 if e["n_success"] > 0 else 0.0 for e in episodes]) if episodes else 0.0
    summary = {
        "schema_version": RESULT_SCHEMA,
        "n_episodes": len(episodes),
        "k": args.k, "temperature": args.temperature,
        "per_episode": [{ "episode_id": e["episode_id"], "task_id": e["task_id"],
                          "successes": e["successes"], "n_success": e["n_success"],
                          "chunk_l2_mean": e["chunk_l2_mean"]} for e in episodes],
        "n_mixed_states": len(mixed),
        "best_of_1_rate": best_of_1,
        "oracle_at_k_rate": oracle,
        "oracle_gain_pp": (oracle - best_of_1) * 100.0,
        "gate": {
            "require_mixed_states": 2,
            "n_mixed_states": len(mixed),
            "require_oracle_gain_pp": 3.0,
            "oracle_gain_pp": (oracle - best_of_1) * 100.0,
            "verdict": "PASS" if len(mixed) >= 2 and (oracle - best_of_1) * 100 >= 3.0 else "FAIL",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

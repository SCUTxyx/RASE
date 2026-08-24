#!/usr/bin/env python3
"""V6 Stage 0: AC_R opportunity pilot (SmolVLA x Long x Pos 0.2).

Per root (chunk-boundary decision at k/H in {0.25,0.5,0.75}):
  - execute m=6 steps of a fresh chunk (seed S_old) from the decision state
  - freeze snapshot
  - C      : continue old chunk remainder (stale 4 steps) -> mu -> terminal
  - R-same : fresh obs + matched generation seed S_old -> new chunk -> mu
  - R-new  : fresh obs + new seeds (S_old + 1000*k, K=4) -> new chunk -> mu
mu = fixed-horizon greedy with matched continuation seeds across branches.
Statistics: R>C / C>R counts, E[A_R], AC_R = E[max] - max(E).
Pilot PASS: requery-better >=3 AND continue-better >=3.

Usage (server, smolvla env; GPU free):
  python scripts/e7_stage0_pilot.py \
    --pert 0.2 --suite libero_10 --roots 24 --k-new 4 \
    --policy ckpts/smolvla_libero --tokenizer ckpts/SmolVLM2-500M-Instruct \
    --output runs/e7_stage0_pilot_v1
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
from rase.collect.forked_rollout import load_lerobot_policy_bundle
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.policy_step import as_batched_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.candidates import seed_everything
from e4_candidate_pool_audit import get_sim_state, restore_env
from g0_pos_baseline import apply_position_perturbation, bddl_objects_of_interest
from lerobot.envs.utils import preprocess_observation

SCHEMA = "rase-v6-stage0-pilot/v1"
NATIVE_H = 10
MU_SEED_BASE = 777000


def infer_chunk(bundle, obs, task, seed):
    """Deterministic chunk inference under a given global seed."""
    policy = bundle["policy"]
    seed_everything(seed)
    policy_observation = preprocess_observation(
        {key: value for key, value in obs.items() if key != "task"})
    policy_observation["task"] = [task]
    env_observation = bundle["env_preprocessor"](policy_observation)
    processed = bundle["preprocessor"](env_observation)
    chunk = policy.predict_action_chunk(processed)
    chunk = bundle["postprocessor"](chunk)
    return chunk.detach().cpu().numpy().reshape(-1, 7).astype(np.float32)


def rollout_from(handle, single, bundle, task, obs, horizon, mu_seed):
    """Execute with fixed-horizon mu (replan every NATIVE_H steps, seeded)."""
    t = 0
    steps = 0
    success = False
    stop = "horizon"
    replans = 0
    while t < horizon:
        seed = mu_seed + replans
        chunk = infer_chunk(bundle, obs, task, seed)
        replans += 1
        for a in chunk:
            obs, _, term, trunc, info = handle.vector_env.step(as_batched_action(a))
            t += 1
            steps += 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                success = bool(success_from_info(info))
                stop = "success" if success else "terminal_failure"
                return success, steps, stop
    return success, steps, stop


def run_root(bundle, protocol_rec, args, rng, log, k_frac) -> dict:
    """One root: prefix -> decision state -> C/R branches."""
    from rase.collect.forked_rollout import InProcessLeRobotContinuation
    handle = make_libero_env_for_task(
        str(protocol_rec["task_id"]), init_state_id=int(protocol_rec["init_state_id"]),
        seed=int(protocol_rec["environment_seed"]), observation_height=360,
        observation_width=360, libero_clean_root="/root/autodl-tmp/src/LIBERO",
        libero_flavor="clean")
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        horizon = int(getattr(single, "_max_episode_steps", 600))
        # position perturbation (exact displacement by `level`)
        tgt = []
        bddl_dir = Path(f"/root/autodl-tmp/libero_pro_root_object/bddl_files/{args.suite}")
        for f in bddl_dir.glob("*.bddl"):
            if task.split(".")[0][:20] in f.read_text():
                tgt = bddl_objects_of_interest(str(f))
                break
        apply_position_perturbation(single, args.pert, rng, tgt or None)

        t_target = int(round(horizon * k_frac))
        # --- run prefix with fixed-horizon mu to the decision boundary ---
        obs = observation_from_libero_env(single)
        t = 0
        old_chunk = None
        while t < t_target:
            seed = MU_SEED_BASE + (t // NATIVE_H)
            chunk = infer_chunk(bundle, obs, task, seed)
            if t + NATIVE_H > t_target:
                old_chunk = chunk  # chunk straddling the decision boundary
            for a in chunk:
                if t >= t_target:
                    break
                obs, _, term, trunc, _ = handle.vector_env.step(as_batched_action(a))
                t += 1
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    return {"status": "early_terminal", "t": t}
        # execute m=6 steps of old_chunk from the decision state
        snapshot = get_sim_state(single)
        obs_at = observation_from_libero_env(single)
        s_old = MU_SEED_BASE + (t_target // NATIVE_H)
        if old_chunk is None:
            old_chunk = infer_chunk(bundle, obs_at, task, s_old)
        obs_m = obs_at
        for a in old_chunk[:6]:
            obs_m, _, term, trunc, _ = handle.vector_env.step(as_batched_action(a))
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                return {"status": "early_terminal_after_m", "t": t}
        snapshot_m = get_sim_state(single)

        mu_seed = MU_SEED_BASE + 1000  # matched continuation seeds across branches

        # ---- C: continue stale remainder ----
        obs_c = restore_env(single, snapshot_m)
        c_succ, c_steps, c_stop = rollout_from(
            handle, single, bundle, task, obs_c, horizon, mu_seed)
        restore_env(single, snapshot_m)

        # ---- R-same: fresh obs + matched generation seed ----
        obs_r = observation_from_libero_env(single)
        rs_chunk = infer_chunk(bundle, obs_r, task, s_old)
        for a in rs_chunk:
            obs_r, _, term, trunc, _ = handle.vector_env.step(as_batched_action(a))
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
        rs_succ, rs_steps, rs_stop = rollout_from(
            handle, single, bundle, task, obs_r, horizon, mu_seed)
        restore_env(single, snapshot_m)

        # ---- R-new: fresh obs + new seeds (K) ----
        rn = []
        for k in range(args.k_new):
            obs_r = observation_from_libero_env(single)
            rn_chunk = infer_chunk(bundle, obs_r, task, s_old + 1000 * (k + 1))
            for a in rn_chunk:
                obs_r, _, term, trunc, _ = handle.vector_env.step(as_batched_action(a))
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    break
            s, st, sp = rollout_from(handle, single, bundle, task, obs_r, horizon, mu_seed)
            rn.append({"k": k, "success": s, "steps": st, "stop": sp})
            restore_env(single, snapshot_m)

        return {
            "task_id": protocol_rec["task_id"],
            "init_state_id": protocol_rec["init_state_id"],
            "k_frac": args.k_frac,
            "t_target": t_target,
            "C": {"success": c_succ, "steps": c_steps, "stop": c_stop},
            "R_same": {"success": rs_succ, "steps": rs_steps, "stop": rs_stop},
            "R_new": rn,
        }
    finally:
        handle.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pert", type=float, default=0.2)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--roots", type=int, default=24)
    ap.add_argument("--k-new", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="ckpts/smolvla_libero")
    ap.add_argument("--tokenizer", default="ckpts/SmolVLM2-500M-Instruct")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    old = get_libero_path("bddl_files")
    set_libero_path("/root/autodl-tmp/libero_pro_root_object")
    print(f"[s0] domain: SmolVLA x {args.suite} x pos{args.pert} roots={args.roots}", flush=True)
    try:
        bundle = load_lerobot_policy_bundle(
            args.policy, device="cuda", num_steps=10, n_action_steps=10,
            tokenizer_path=args.tokenizer,
            observation_height=360, observation_width=360)
        rng = np.random.default_rng(args.seed)
        # roots: 24 = 8 x 3 layers (k/H in {0.25, 0.5, 0.75})
        n_per_layer = args.roots // 3
        kfracs = [0.25, 0.5, 0.75]
        tasks = [1, 2, 3, 5, 6, 7, 8, 9]  # 8 tasks (skip easy 4/10)
        results = []
        t0 = time.time()
        for li, kf in enumerate(kfracs):
            for ti in range(n_per_layer):
                task_id = f"{args.suite}_{tasks[ti % len(tasks)]:06d}"
                rec = {"task_id": task_id, "init_state_id": (li * n_per_layer + ti) % 10,
                       "environment_seed": 3000 + li * 100 + ti}
                res = run_root(bundle, rec, args, rng, None, kf)
                res["k_frac"] = kf
                results.append(res)
                el = time.time() - t0
                print(f"[s0] root {len(results)}: kf={kf} task={task_id} "
                      f"C={res.get('C',{}).get('success')} "
                      f"R_same={res.get('R_same',{}).get('success')} "
                      f"R_new={[r['success'] for r in res.get('R_new',[])]} "
                      f"elapsed={el/60:.1f}m", flush=True)
        # ---- statistics ----
        qc = np.array([r["C"]["success"] for r in results if "C" in r], dtype=float)
        qr_same = np.array([r["R_same"]["success"] for r in results if "R_same" in r], dtype=float)
        qr_new = np.array([[b["success"] for b in r["R_new"]]
                           for r in results if "R_new" in r], dtype=float)
        qr = np.maximum(qr_same, qr_new.max(axis=1))  # best refresh outcome
        r_better = int(np.sum(qr > qc))
        c_better = int(np.sum(qc > qr))
        e_c = qc.mean()
        e_r = qr.mean()
        ac_r = float((np.maximum(qc, qr) - np.maximum(e_c, e_r)).mean())
        report = {
            "schema_version": SCHEMA,
            "domain": f"SmolVLA x {args.suite} x pos{args.pert}",
            "n_roots": len(results),
            "continue_rate": float(e_c),
            "refresh_best_rate": float(e_r),
            "n_R_better": r_better,
            "n_C_better": c_better,
            "AC_R": ac_r,
            "gate": {
                "requery_better": r_better, "continue_better": c_better,
                "pass": bool(r_better >= 3 and c_better >= 3),
            },
            "per_root": results,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in (
            "domain", "n_roots", "continue_rate", "refresh_best_rate",
            "n_R_better", "n_C_better", "AC_R", "gate")}, indent=2), flush=True)
        return 0
    finally:
        set_libero_path(old)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""R0: execution-verification probe (CheckVLA-style premise check).

Question: on π0-fast / LIBERO-Long, is there a learnable signal in
"chunk execution outcome vs expectation"?
  P1 (predictability) : can a light predictor f(s_t, chunk) -> s_{t+8}
                        beat the identity baseline (s_{t+8} ~= s_t)?
  P2 (deviation-risk) : is per-state prediction error correlated with
                        terminal failure (deviation AUROC >= 0.65)?

Data: replay the E4-0 decision states (24 states x K=8 candidates; same
seeds -> same chunks), execute each native chunk (8 steps) from the frozen
snapshot, record branch-end proprio + object poses; terminal labels come
from E4-0 (successes).

Gate: P2 AUROC >= 0.65 AND P1 relative MSE improvement >= 10%.
If PASS -> R1 (execution-verification closed loop). If FAIL -> execution
verification has no signal on this stack; LIBERO-domain closeout.

Usage (server, smolvla env; GPU free):
  python scripts/e5_ev_probe.py \
    --e4-dir runs/e4_candidate_pool_audit_v1 \
    --config configs/g2a_pi0fast_clean_long_v1.json \
    --output runs/e5_ev_probe_v1
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
from rase.collect.candidates import seed_everything
from e4_candidate_pool_audit import (
    get_sim_state, restore_env, sample_chunk_with_temperature,
)

SCHEMA = "rase-e5-ev-probe/v1"


def object_poses(single) -> list:
    """Privileged object body xyz (robosuite sim)."""
    try:
        rob = single._env
        sim = rob.sim
        names = sim.model.body_names
        out = []
        for i, n in enumerate(names):
            ln = str(n).lower()
            if any(k in ln for k in ("robot", "ground", "table", "base",
                                     "link", "wall")):
                continue
            out.append([str(n), [float(x) for x in sim.data.body_xpos[i]]])
        return out
    except Exception:
        return []


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    s = np.sqrt(max(1.0 - w * w, 1e-12))
    return np.array([x, y, z]) / s * angle


def proprio8(obs) -> np.ndarray:
    """8-d proprio: eef_pos(3) + axisangle(3) + gripper_qpos(2)."""
    rs = obs["robot_state"]
    pos = np.asarray(rs["eef"]["pos"][0], dtype=np.float64)
    aa = quat2axisangle(np.asarray(rs["eef"]["quat"][0], dtype=np.float64))
    gq = np.asarray(rs["gripper"]["qpos"][0], dtype=np.float64)
    return np.concatenate([pos, aa, gq])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e4-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--native-h", type=int, default=8)
    ap.add_argument("--decision-step", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--policy-path", default="ckpts/pi0fast_libero")
    ap.add_argument("--tokenizer-path", default="ckpts/paligemma_tokenizer_35e4f46")
    ap.add_argument("--action-tokenizer-path", default="ckpts/pi0fast_action_tokenizer_79ae83e")
    args = ap.parse_args()

    e4 = json.loads((args.e4_dir / "summary.json").read_text())
    episodes = e4["per_episode"]
    outcomes = {e["episode_id"]: e["successes"] for e in episodes}
    protocol = json.loads(args.config.read_text())
    records = {r["episode_id"]: r for r in protocol["records"]
               if r["episode_id"] in outcomes}

    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda",
        num_steps=int(protocol["num_steps"]),
        n_action_steps=int(protocol["n_action_steps"]),
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )

    rows = []
    for rec in records.values():
        handle = make_libero_env_for_task(
            str(rec["task_id"]), init_state_id=int(rec["init_state_id"]),
            seed=int(rec["environment_seed"]), observation_height=360,
            observation_width=360, libero_clean_root=os.environ.get("LIBERO_CLEAN_ROOT"),
            libero_flavor="clean")
        try:
            single = handle.vector_env.envs[0]
            task = str(single.task_description)
            horizon = int(getattr(single, "_max_episode_steps", 600))
            cont = InProcessLeRobotContinuation(bundle, seed=int(rec["policy_seed"]))
            obs = observation_from_libero_env(single)
            t = 0
            while t < args.decision_step and t < horizon:
                action = cont.act(obs, task=task)
                obs, _, term, trunc, _ = handle.vector_env.step(as_batched_action(action))
                t += 1
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    break
            snapshot = get_sim_state(single)
            obs_at = observation_from_libero_env(single)
            s_t_prop = proprio8(obs_at)
            s_t_objs = object_poses(single)

            chunks = []
            for k in range(args.k):
                seed = int(rec["policy_seed"]) + 1000 * (k + 1) + int(rec["clean_task_index"]) * 100000
                chunks.append(sample_chunk_with_temperature(
                    bundle, obs_at, task, args.temperature, seed, horizon=args.native_h))

            for k, chunk in enumerate(chunks):
                obs_r = restore_env(single, snapshot)
                s8_prop = None
                s8_objs = None
                for step in range(args.native_h):
                    obs_r, _, term, trunc, _ = handle.vector_env.step(
                        as_batched_action(chunk[step]))
                s8_prop = proprio8(obs_r)
                s8_objs = object_poses(single)
                rows.append({
                    "episode_id": rec["episode_id"],
                    "candidate": k,
                    "s_t_proprio": s_t_prop.tolist(),
                    "chunk": chunk.tolist(),
                    "s_t8_proprio": s8_prop.tolist() if s8_prop is not None else None,
                    "s_t_objects": s_t_objs,
                    "s_t8_objects": s8_objs,
                    "success": bool(outcomes[rec["episode_id"]][k]),
                })
            print(f"[r0] {rec['episode_id']} done", flush=True)
        finally:
            handle.close()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    # ---------------- analysis ----------------
    from rase_common import canonical_chunk_features, auroc

    X_state = np.stack([np.asarray(r["s_t_proprio"]) for r in rows])
    X_chunk = np.stack([canonical_chunk_features(np.asarray(r["chunk"])) for r in rows])
    Y = np.stack([np.asarray(r["s_t8_proprio"]) for r in rows])
    succ = np.array([r["success"] for r in rows], dtype=float)

    # P1: ridge predictor (s_t, chunk) -> s_{t+8}; identity baseline
    def fit_ridge_reg(X, y, alpha=1.0):
        mean = X.mean(0); scale = X.std(0); scale[scale < 1e-8] = 1.0
        Xs = (X - mean) / scale
        design = np.column_stack((np.ones(len(Xs)), Xs))
        pen = np.eye(design.shape[1]) * alpha; pen[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + pen, design.T @ y)
        return mean, scale, beta

    def ridge_pred(X, mean, scale, beta):
        Xs = (np.asarray(X) - mean) / scale
        return np.column_stack((np.ones(len(Xs)), Xs)) @ beta

    X = np.concatenate([X_state, X_chunk], axis=1)
    mean, scale, beta = fit_ridge_reg(X, Y)
    Yhat = ridge_pred(X, mean, scale, beta)
    mse_pred = float(np.mean((Yhat - Y) ** 2))
    mse_ident = float(np.mean((X_state - Y) ** 2))
    rel_improve = (mse_ident - mse_pred) / max(mse_ident, 1e-12)

    # P2: deviation (per-state standardized) vs success
    keys = sorted(set(r["episode_id"] for r in rows))
    dev = np.array([float(np.linalg.norm(Yhat[i] - Y[i])) for i in range(len(rows))])
    dev_z = np.zeros(len(rows))
    for ep in keys:
        m = np.array([i for i, r in enumerate(rows) if r["episode_id"] == ep])
        d = dev[m]
        dev_z[m] = (d - d.mean()) / (d.std() + 1e-9)
    auc_global = auroc(dev, succ)
    auc_z = auroc(dev_z, succ)
    # failure rate by deviation tercile
    order = np.argsort(dev)
    terc = np.array_split(order, 3)
    terc_fail = [1.0 - succ[m].mean() for m in terc]

    report = {
        "schema": SCHEMA,
        "n_rows": len(rows), "n_states": len(keys), "k": args.k,
        "P1_predictability": {
            "mse_identity_baseline": mse_ident,
            "mse_predictor": mse_pred,
            "relative_improvement": rel_improve,
            "gate_ge_0_10": bool(rel_improve >= 0.10),
        },
        "P2_deviation_risk": {
            "auroc_global": auc_global,
            "auroc_per_state_z": auc_z,
            "failure_rate_by_deviation_tercile": [float(x) for x in terc_fail],
            "gate_auroc_ge_0_65": bool(auc_z >= 0.65),
        },
        "gate": {
            "verdict": "PASS" if (rel_improve >= 0.10 and auc_z >= 0.65) else "FAIL",
            "note": "PASS -> R1 execution-verification closed loop; FAIL -> no signal",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

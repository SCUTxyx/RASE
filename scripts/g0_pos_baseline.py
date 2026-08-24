#!/usr/bin/env python3
"""G0-pos: fixed-horizon baseline under object POSITION perturbation.

Implementation follows LIBERO-Pro (arXiv:2510.03827) + AAC protocol:
displace the manipulated object (obj_of_interest from BDDL) in the xy-plane
by `level` scene units from its init position, then rollout.

Usage (server, smolvla env):
  python scripts/g0_pos_baseline.py \
    --suite libero_object --tasks 10 --eps-per-task 8 --level 0.2 \
    --policy ckpts/smolvla_libero --tokenizer ckpts/SmolVLM2-500M-Instruct \
    --output runs/g0_pos_object_smolvla_l02_v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

SCHEMA = "rase-g0-pos-baseline/v1"


def bddl_objects_of_interest(bddl_path: str) -> list[str]:
    """Parse (:obj_of_interest ...) names from a BDDL file."""
    txt = Path(bddl_path).read_text()
    m = re.search(r"\(:obj_of_interest\s+(.*?)\)", txt, re.S)
    if not m:
        return []
    return re.findall(r"(\S+)\s*$", m.group(1).strip(), re.M) or \
        [t for t in m.group(1).split() if t]


def apply_position_perturbation(single, level: float, rng, target_objects=None):
    """Displace target objects in xy-plane by modifying free-joint qpos.

    robosuite body_xpos is derived from qpos; must edit qpos + sim.forward().
    """
    rob = single._env
    sim = rob.sim
    if target_objects:
        names = [n for n in target_objects if n in sim.model.body_names]
    else:
        names = []
        for i, n in enumerate(sim.model.body_names):
            ln = str(n).lower()
            if any(k in ln for k in ("robot", "ground", "table", "base",
                                     "link", "wall", "floor")):
                continue
            names.append(str(n))
    moved = []
    for name in names:
        # free joints for this body are named {body}_joint0 in robosuite
        joint_name = f"{name}_joint0"
        if joint_name not in sim.model.joint_names:
            continue
        jid = sim.model.joint_name2id(joint_name)
        if int(sim.model.jnt_type[jid]) != 0:  # not free joint
            continue
        adr = int(sim.model.jnt_qposadr[jid])
        ang = rng.uniform(0, 2 * np.pi)
        r = float(level)  # exact displacement (AAC-style magnitude)
        sim.data.qpos[adr] += r * np.cos(ang)
        sim.data.qpos[adr + 1] += r * np.sin(ang)
        moved.append(name)
    sim.forward()
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pert-root", default="object", choices=["object", "lan", "swap", "task"])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--eps-per-task", type=int, default=8)
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="ckpts/smolvla_libero")
    ap.add_argument("--tokenizer", default="ckpts/SmolVLM2-500M-Instruct")
    ap.add_argument("--pro-root", default="/root/autodl-tmp/libero_pro_root_")
    ap.add_argument("--bddl-root", default="/root/autodl-tmp/libero_pro_root_")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = f"{args.pro_root}{args.pert_root}"
    old = get_libero_path("bddl_files")
    set_libero_path(root)
    print(f"[g0p] libero root -> {root} level={args.level}", flush=True)
    try:
        bundle = load_lerobot_policy_bundle(
            args.policy, device="cuda", num_steps=10, n_action_steps=10,
            tokenizer_path=args.tokenizer,
            observation_height=360, observation_width=360)
        rng = np.random.default_rng(args.seed)
        results = []
        t0 = time.time()
        for ti in range(1, args.tasks + 1):
            task_id = f"{args.suite}_{ti:06d}"
            # locate bddl for obj_of_interest (map task index -> file name)
            bddl_dir = Path(f"{args.bddl_root}{args.pert_root}/bddl_files/{args.suite}")
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
                    # find bddl file by task description match
                    tgt = []
                    for f in bddl_dir.glob("*.bddl"):
                        txt = f.read_text()
                        if task.split(".")[0][:20] in txt:
                            tgt = bddl_objects_of_interest(str(f))
                            break
                    perturbed = apply_position_perturbation(single, args.level, rng, tgt or None)
                    obs = observation_from_libero_env(single)
                    cont = InProcessLeRobotContinuation(bundle, seed=2000 + ep)
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
                    per_ep.append({"ep": ep, "success": bool(success), "steps": t,
                                   "stop": stop, "perturbed_objects": perturbed})
                finally:
                    h.close()
            results.append({"task_id": task_id, "success": ok,
                            "episodes": args.eps_per_task, "per_episode": per_ep})
            el = time.time() - t0
            print(f"[g0p] {task_id}: {ok}/{args.eps_per_task}  elapsed={el/60:.1f}m", flush=True)
        total_ok = sum(r["success"] for r in results)
        total_n = sum(r["episodes"] for r in results)
        report = {
            "schema_version": SCHEMA,
            "perturbation": f"position@{args.level}", "suite": args.suite,
            "policy": args.policy, "seed": args.seed,
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

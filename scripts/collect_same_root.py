#!/usr/bin/env python3
"""Stage B: same-root counterfactual collection (roadmap §4-§5).

One physical state s_t -> multiple candidate chunks -> matched execution of
H steps from the SAME simulator snapshot -> consequence label via a unified
evaluator (not the candidate's own long-horizon competence).

Label modes:
  --label-mode reference : roll the reference model K steps from s_{t+H} and
                           record success + end proprio (sparse but clean);
  --label-mode progress  : record short-horizon state delta (displacement,
                           gripper change) as a progress proxy (dense, noisy).

Rows: {task, suite, episode_idx, decision_idx, model, chunk_raw,
       s_t proprio (8), s_{t+H} proprio (8), consequence_label,
       recovery_success (if reference mode)}.

Server-side env/model imports are deferred; syntax-checkable locally.

Usage (server, oft env):
  python collect_same_root.py \
    --models oft_spatial,oft_object \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --output runs/oft_opportunity/same_root_v1.jsonl \
    --episodes 2 --horizon 8 --label-mode reference \
    --reference oft_spatial --ref-steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK_MAX_STEPS = {
    "libero_spatial": 520, "libero_object": 520, "libero_goal": 520,
    "libero_10": 520, "libero_90": 400,
}


def chunk_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    pos = arr[:, :3]
    return {
        "chunk_mean_pos": [float(x) for x in arr[:, :3].mean(axis=0)],
        "chunk_mean_rot": [float(x) for x in arr[:, 3:6].mean(axis=0)],
        "chunk_std_pos": [float(x) for x in arr[:, :3].std(axis=0)],
        "chunk_std_rot": [float(x) for x in arr[:, 3:6].std(axis=0)],
        "chunk_gripper_mean": float(arr[:, 6].mean()),
        "chunk_gripper_std": float(arr[:, 6].std()),
        "chunk_total_disp": float(np.abs(np.diff(pos, axis=0)).sum()),
        "chunk_norm_mean": float(np.linalg.norm(pos, axis=1).mean()),
        "chunk_raw": [float(x) for x in arr.flatten()],
    }


def proprio_from_full_obs(full: dict) -> np.ndarray:
    """8-d proprio in the OpenVLA layout (eef_pos3 + axisangle3 + gripper2)."""
    from experiments.robot.libero.libero_utils import quat2axisangle  # noqa
    return np.concatenate([
        full["robot0_eef_pos"],
        quat2axisangle(full["robot0_eef_quat"]),
        full["robot0_gripper_qpos"],
    ])


def force_clear(env) -> None:
    """Clear robosuite episode-termination bookkeeping.  `done` is set at the
    end of every step when timestep >= horizon; simulator restore does NOT
    reset timestep, so multi-candidate forks accumulate steps and trip the
    'executing action in terminated episode' guard.  Reset both."""
    try:
        inner = getattr(env, "env", env)
        inner.done = False
        inner.timestep = 0
    except Exception:
        pass


def restore_env(env, snapshot) -> dict:
    """Restore simulator snapshot, regen obs, then clear terminated state."""
    env.set_state(snapshot)
    obs = env.regenerate_obs_from_state(snapshot)
    force_clear(env)
    return obs


def get_obs(env) -> dict:
    """robosuite OffScreenRenderEnv observation (force-updated)."""
    if hasattr(env, "_get_observations"):
        return env._get_observations(force_update=True)
    return env.get_observation()


def object_poses(env) -> list:
    """Privileged object body poses (name -> xyz) for divergence/labels."""
    try:
        sim = env.sim
        names = sim.model.body_names
        out = []
        for i, n in enumerate(names):
            ln = n.lower()
            if any(k in ln for k in ("robot", "ground", "table", "base",
                                     "link", "wall")):
                continue
            out.append([str(n), [float(x) for x in sim.data.body_xpos[i]]])
        return out
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True,
                        help="comma-separated candidate models (oft_*)")
    parser.add_argument("--ckpts-root", default="/root/autodl-tmp/RASE/ckpts")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--label-mode", choices=["reference", "progress"],
                        default="progress")
    parser.add_argument("--reference", default="oft_spatial")
    parser.add_argument("--ref-steps", type=int, default=40)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=8,
                        help="subset of tasks for the first pass")
    parser.add_argument("--decisions-per-episode", type=int, default=4,
                        help="only collect the first N decision points")
    args = parser.parse_args()

    sys.path.insert(0, "/root/autodl-tmp/openvla-oft")
    from experiments.robot.openvla_utils import (  # noqa
        get_vla, get_processor, get_action_head, get_proprio_projector,
        get_vla_action, resize_image_for_policy,
    )
    from experiments.robot.libero.libero_utils import (  # noqa
        get_libero_env, get_libero_image, get_libero_wrist_image,
        get_libero_dummy_action,
    )
    from experiments.robot.robot_utils import (  # noqa
        get_image_resize_size, normalize_gripper_action, invert_gripper_action,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa
    from libero.libero import benchmark  # noqa

    models = [m.strip() for m in args.models.split(",")]

    def load_model(name: str):
        cfg = argparse.Namespace(
            pretrained_checkpoint=f"{args.ckpts_root}/{name}",
            model_family="openvla", use_l1_regression=True, use_diffusion=False,
            num_diffusion_steps_inference=50, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK, lora_rank=32,
            unnorm_key=list(json.load(open(
                f"{args.ckpts_root}/{name}/dataset_statistics.json"
            )).keys())[0],
            load_in_8bit=False, load_in_4bit=False,
        )
        vla = get_vla(cfg)
        head = get_action_head(cfg, vla.llm_dim)
        proj = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
        proc = get_processor(cfg)
        return {"cfg": cfg, "vla": vla, "head": head, "proj": proj,
                "proc": proc, "name": name}

    def act(m: dict, observation: dict, task: str) -> list[np.ndarray]:
        return get_vla_action(
            m["cfg"], m["vla"], m["proc"], observation, task,
            action_head=m["head"], proprio_projector=m["proj"])

    def prep(obs, resize) -> dict:
        gi, gwi, q2a, rip = (get_libero_image, get_libero_wrist_image,
                             None, resize_image_for_policy)
        return {
            "full_image": rip(gi(obs), resize),
            "wrist_image": rip(gwi(obs), resize),
            "state": proprio_from_full_obs(obs),
        }

    resize = get_image_resize_size(argparse.Namespace(model_family="openvla"))
    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]
    benchmark_dict = benchmark.get_benchmark_dict()

    # reference model is loaded first and kept; candidates are loaded one at a
    # time (VRAM: 7B bf16 ≈ 15.5G each; two residents OK, more must swap).
    ref = load_model(args.reference)
    rows: list[dict] = []
    for i, task in enumerate(tasks):
        task_suite = benchmark_dict[task["suite"]]()
        task_obj = task_index = None
        for idx, t in enumerate(task_suite.tasks):
            if t.language == task["task"]:
                task_obj, task_index = t, idx
                break
        if task_obj is None:
            continue
        initial_states = task_suite.get_task_init_states(task_index)
        env, task_description = get_libero_env(
            task_obj, "openvla", resolution=args.env_img_res)
        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            t = 0
            decision_idx = 0
            max_steps = TASK_MAX_STEPS[task["suite"]]
            while t < max_steps + args.num_steps_wait and \
                    decision_idx < args.decisions_per_episode:
                if t < args.num_steps_wait:
                    obs, _, done, info = env.step(
                        get_libero_dummy_action("openvla"))
                    t += 1
                    continue
                # ---- decision point: same-root multi-candidate ----
                snapshot = env.get_sim_state()
                s_t_prop = proprio_from_full_obs(obs)
                s_t_obj = object_poses(env)  # decision-point object poses
                for name in models:
                    m = load_model(name)
                    obs_at = restore_env(env, snapshot)
                    observation = prep(obs_at, resize)
                    chunk = act(m, observation, task_description)
                    arr = np.asarray(chunk, dtype=np.float64)
                    future_prop: list[list[float]] = []
                    future_objects: list[list] = []
                    steps_taken = 0
                    while steps_taken < args.horizon:
                        # open-loop reuse of the candidate chunk beyond its
                        # native length so short-horizon consequences are real
                        a = normalize_gripper_action(
                            arr[steps_taken % len(arr)], binarize=True)
                        a = invert_gripper_action(a)
                        force_clear(env)
                        obs_at, _, done, info = env.step(a.tolist())
                        future_prop.append(
                            proprio_from_full_obs(obs_at).tolist())
                        if steps_taken % 8 == 7:
                            future_objects.append(object_poses(env))
                        steps_taken += 1
                        if done:
                            break
                    s_th_prop = proprio_from_full_obs(obs_at)
                    row = {
                        "task": task["task"], "suite": task["suite"],
                        "episode_idx": ep, "decision_idx": decision_idx,
                        "model": name,
                        "s_t_proprio": [float(x) for x in s_t_prop],
                        "s_th_proprio": [float(x) for x in s_th_prop],
                        "future_proprio": future_prop,
                        "future_objects": future_objects,
                        "future_steps": steps_taken,
                        "s_t_objects": s_t_obj,
                        "s_th_objects": object_poses(env),
                        **chunk_stats(arr),
                    }
                    if args.label_mode == "reference":
                        # unified evaluator: reference policy from s_{t+H}
                        obs_r = restore_env(env, snapshot)
                        queue = act(ref, prep(obs_r, resize), task_description)
                        q = [normalize_gripper_action(x, binarize=True) for x in queue]
                        q = [invert_gripper_action(x) for x in q]
                        succ = False
                        for h in range(args.ref_steps):
                            obs_r, _, done, info = env.step(q[h % len(q)].tolist())
                            if done:
                                succ = True
                                break
                        row["recovery_success"] = int(succ)
                        row["consequence_label"] = int(succ)
                    else:  # progress proxy: displacement + gripper change
                        d = float(np.linalg.norm(s_th_prop[:3] - s_t_prop[:3]))
                        g = float(abs(s_th_prop[7] - s_t_prop[7]))
                        row["displacement"] = d
                        row["gripper_delta"] = g
                        row["consequence_label"] = float(d)  # larger = progressed
                    rows.append(row)
                    del m  # free VRAM before next candidate
                    import gc
                    gc.collect()
                    obs = restore_env(env, snapshot)
                # continue the source trajectory one chunk (use reference)
                observation = prep(obs, resize)
                queue = act(ref, observation, task_description)
                for a in queue:
                    aa = normalize_gripper_action(a, binarize=True)
                    aa = invert_gripper_action(aa)
                    obs, _, done, info = env.step(aa.tolist())
                    if done:
                        break
                decision_idx += 1
                t += args.horizon
        env.close()
        print(f"[sr] {i + 1}/{len(tasks)} {task['task'][:40]} rows={len(rows)}",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"rows": len(rows), "models": models,
                      "label_mode": args.label_mode}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""RASE OPD collection: decision-point features for one OFT model.

Rolls out one model on the 20-task mixed domain; at every decision point
(every 8 steps, when the action queue is empty) records:
  - task, suite, episode idx (pairing key with the other model)
  - proprio (8-d)
  - the candidate action chunk produced at this decision point (8x7)
  - episode outcome (success) attached to every row of that episode

Run once with --model oft_spatial and once with --model oft_object; rows are
paired by (task, episode_idx, decision_idx) for pairwise distillation.

Usage:
  python rase_oft_dp_collect.py --model oft_spatial \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --output runs/oft_opportunity/dp_collect_spatial.jsonl \
    --episodes 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/openvla-oft")
sys.path.insert(0, str(ROOT))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import (  # noqa: E402
    normalize_gripper_action,
    invert_gripper_action,
    get_image_resize_size,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402
from libero.libero import benchmark  # noqa: E402

TASK_MAX_STEPS = {
    "libero_spatial": 520, "libero_object": 520, "libero_goal": 520,
    "libero_10": 520, "libero_90": 400,
}


class SingleModel:
    def __init__(self, name: str) -> None:
        cfg = argparse.Namespace(
            pretrained_checkpoint=f"/root/autodl-tmp/RASE/ckpts/{name}",
            model_family="openvla", use_l1_regression=True, use_diffusion=False,
            num_diffusion_steps_inference=50, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK, lora_rank=32,
            unnorm_key=list(json.load(open(
                f"/root/autodl-tmp/RASE/ckpts/{name}/dataset_statistics.json"
            )).keys())[0],
            load_in_8bit=False, load_in_4bit=False,
        )
        vla = get_vla(cfg)
        head = get_action_head(cfg, vla.llm_dim)
        proj = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
        proc = get_processor(cfg)
        self.cfg = cfg
        self.vla, self.head, self.proj, self.proc = vla, head, proj, proc

    def act(self, obs: dict, task: str) -> list[np.ndarray]:
        return get_vla_action(
            self.cfg, self.vla, self.proc, obs, task,
            action_head=self.head, proprio_projector=self.proj,
        )


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
             obs["robot0_gripper_qpos"])
        ),
    }
    return observation


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def chunk_stats(chunk: list[np.ndarray]) -> dict:
    arr = np.asarray(chunk, dtype=np.float64)  # (8, 7)
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
        "chunk_raw": [float(x) for x in arr.flatten()],  # 56-d, for later analysis
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["oft_spatial", "oft_object", "oft_goal", "oft_10"])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=20)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]
    benchmark_dict = benchmark.get_benchmark_dict()
    resize_size = get_image_resize_size(argparse.Namespace(model_family="openvla"))
    model = SingleModel(args.model)
    print(f"[collect] model {args.model} loaded", flush=True)

    rows: list[dict] = []
    for i, task in enumerate(tasks):
        task_suite = benchmark_dict[task["suite"]]()
        task_obj = None
        task_index = None
        for idx, t in enumerate(task_suite.tasks):
            if t.language == task["task"]:
                task_obj, task_index = t, idx
                break
        if task_obj is None:
            print(f"[collect] WARN task not found: {task['task'][:40]}", flush=True)
            continue
        initial_states = task_suite.get_task_init_states(task_index)
        env, task_description = get_libero_env(
            task_obj, "openvla", resolution=args.env_img_res)
        for ep in range(args.episodes):
            initial_state = initial_states[ep % len(initial_states)]
            env.reset()
            obs = env.set_init_state(initial_state)
            queue: list[np.ndarray] = []
            decision_idx = 0
            ep_rows: list[dict] = []
            t = 0
            max_steps = TASK_MAX_STEPS[task["suite"]]
            success = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, info = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue
                observation = prepare_observation(obs, resize_size)
                if len(queue) == 0:
                    chunk = model.act(observation, task_description)
                    queue = list(chunk)
                    ep_rows.append({
                        "task": task["task"], "suite": task["suite"],
                        "model": args.model, "episode_idx": ep,
                        "decision_idx": decision_idx,
                        "proprio": [float(x) for x in observation["state"]],
                        **chunk_stats(chunk),
                    })
                    decision_idx += 1
                action = process_action(queue.pop(0))
                obs, _, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1
            for row in ep_rows:
                row["success"] = int(success)
            rows.extend(ep_rows)
        env.close()
        print(f"[collect] {i + 1}/{len(tasks)} {task['task'][:40]} ep={args.episodes} "
              f"rows={len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    n_ok = sum(r["success"] for r in rows)
    print(json.dumps({
        "model": args.model, "rows": len(rows), "episodes_success": n_ok,
        "success_rate": n_ok / max(1, len(set((r["task"], r["episode_idx"]) for r in rows))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

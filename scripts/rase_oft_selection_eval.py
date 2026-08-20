#!/usr/bin/env python3
"""RASE closed-loop selection demo on the OFT model pair.

Per task, a leave-one-task-out instruction classifier (trained on the other
19 tasks' per-task success matrix) decides which model (oft_spatial or
oft_object) runs the episode; the chosen model then rolls out in closed loop
using the official OFT eval loop.  This verifies that the comparative
advantage is selectable from observable features and that selection
generalizes to the held-out task.

Baselines come from the matrix: A-only / B-only per-task rates.

Usage:
  python rase_oft_selection_eval.py \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --output runs/oft_opportunity/selection_closed_loop.json \
    --num_trials_per_task 10
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
from prismatic.vla.constants import (  # noqa: E402
    ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX,
    NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX,
)
from libero.libero import benchmark  # noqa: E402

TASK_MAX_STEPS = {
    "libero_spatial": 520,
    "libero_object": 520,
    "libero_goal": 520,
    "libero_10": 520,
    "libero_90": 400,
}


def bigram_features(text: str, vocab: dict[str, int]) -> np.ndarray:
    text = text.lower()
    x = np.zeros(len(vocab), dtype=np.float64)
    for i in range(len(text) - 1):
        idx = vocab.get(text[i:i + 2])
        if idx is not None:
            x[idx] += 1.0
    return x


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in texts:
        text = text.lower()
        for i in range(len(text) - 1):
            vocab.setdefault(text[i:i + 2], len(vocab))
    return vocab


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = (X - mean) / scale
    design = np.column_stack((np.ones(len(Xs)), Xs))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    y_mean = float(y.mean())
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ (y - y_mean))
    return np.concatenate([[y_mean + beta[0]], beta[1:]])


class DualModel:
    """Both OFT models resident; per-task selection is fixed per episode."""

    def __init__(self) -> None:
        self.models: dict[str, dict] = {}
        for name in ("oft_spatial", "oft_object"):
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
            self.models[name] = {"cfg": cfg, "vla": vla, "head": head,
                                 "proj": proj, "proc": proc}
            print(f"[dual] loaded {name}", flush=True)

    def act(self, model_name: str, obs: dict, task: str) -> list[np.ndarray]:
        m = self.models[model_name]
        return get_vla_action(
            m["cfg"], m["vla"], m["proc"], obs, task,
            action_head=m["head"], proprio_projector=m["proj"],
        )


def run_episode(cfg, env, task_description, model, dual, resize_size, initial_state):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue: list[np.ndarray] = []
    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        observation, _ = prepare_observation(obs, resize_size)
        if len(action_queue) == 0:
            action_queue = list(dual.act(model, observation, task_description))
        action = action_queue.pop(0)
        action = process_action(action)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1
    return success


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
    return observation, img


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=20)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]
    if len(tasks) < 2:
        print("need >= 2 tasks")
        return 1

    # LOO classifier data
    texts = [t["task"] for t in tasks]
    vocab = build_vocab(texts)
    X = np.stack([bigram_features(text, vocab) for text in texts])
    y = np.array([1.0 if t["B"] > t["A"] else 0.0 for t in tasks])
    n = len(tasks)
    decisions = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = ridge_fit(X[mask], y[mask])
        scores = 1.0 / (1.0 + np.exp(-(beta[0] + X[i] @ beta[1:])))
        decisions.append(bool(scores > 0.5))

    dual = DualModel()
    resize_size = get_image_resize_size(argparse.Namespace(model_family="openvla"))
    benchmark_dict = benchmark.get_benchmark_dict()

    report = {"schema": "rase-soft-selection-closed-loop/v1", "tasks": {}}
    n_ok = n_ep = 0
    for i, task in enumerate(tasks):
        suite_cls = benchmark_dict[task["suite"]]
        task_suite = suite_cls()
        task_obj = None
        task_index = None
        for idx, t in enumerate(task_suite.tasks):
            if t.language == task["task"]:
                task_obj = t
                task_index = idx
                break
        if task_obj is None:
            report["tasks"][task["task"][:50]] = {"error": "task not found in suite"}
            continue
        initial_states = task_suite.get_task_init_states(task_index)
        env, task_description = get_libero_env(
            task_obj, "openvla", resolution=args.env_img_res)
        cfg = argparse.Namespace(
            task_suite_name=task["suite"], num_steps_wait=args.num_steps_wait,
            env_img_res=args.env_img_res, model_family="openvla",
        )
        model = "oft_object" if decisions[i] else "oft_spatial"
        ok = 0
        for ep in range(args.num_trials_per_task):
            initial_state = initial_states[ep % len(initial_states)]
            success = run_episode(
                cfg, env, task_description, model, dual, resize_size,
                initial_state)
            ok += int(success)
            n_ok += int(success)
            n_ep += 1
        env.close()
        report["tasks"][task["task"][:60]] = {
            "suite": task["suite"],
            "chosen_model": model,
            "success": ok,
            "episodes": args.num_trials_per_task,
            "A_rate": task["A"],
            "B_rate": task["B"],
            "oracle_rate": max(task["A"], task["B"]),
        }
        print(f"[{i + 1}/{n}] {task['task'][:50]} -> {model}: {ok}/{args.num_trials_per_task}",
              flush=True)

    report["overall"] = {
        "success": n_ok,
        "episodes": n_ep,
        "rate": n_ok / n_ep if n_ep else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

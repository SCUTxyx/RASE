#!/usr/bin/env python3
"""Stage A collection: decision-point rows for ANY VLA (OpenVLA/OFT family or
LeRobot-format policies like pi0-fast / smolvla), using a unified wrapper.

Row schema is identical to rase_oft_dp_collect.py (proprio 8, chunk stats,
chunk_raw 56, task/suite/episode/decision, model, success) so that the frozen
v3 risk model can score unseen-VLA rows directly.

For LeRobot policies the continuation yields one action per call (native
10-step internal requery); we accumulate 10 executed actions into the chunk.

Usage (OpenVLA family, oft env):
  python collect_zs_vla.py --vla-type openvla --model oft_goal \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --output runs/oft_opportunity/dp_collect_goal.jsonl --episodes 3

Usage (LeRobot policy, smolvla env):
  python collect_zs_vla.py --vla-type lerobot \
    --policy-path /root/autodl-tmp/RASE/ckpts/pi0fast_libero \
    --tokenizer-path /root/autodl-tmp/RASE/ckpts/paligemma_tokenizer_35e4f46 \
    --action-tokenizer-path /root/autodl-tmp/RASE/ckpts/pi0fast_action_tokenizer_79ae83e \
    --model pi0fast --matrix ... --output ... --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


class OpenVLAWrapper:
    """OFT/OpenVLA checkpoint (runs in the `oft` conda env)."""

    def __init__(self, model_name: str, ckpts_root: str) -> None:
        ROOT = Path("/root/autodl-tmp/openvla-oft")
        sys.path.insert(0, str(ROOT))
        from experiments.robot.openvla_utils import (  # noqa
            get_vla, get_processor, get_action_head, get_proprio_projector,
            get_vla_action, resize_image_for_policy,
        )
        from experiments.robot.libero.libero_utils import (  # noqa
            get_libero_image, get_libero_wrist_image, quat2axisangle,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa
        from experiments.robot.robot_utils import get_image_resize_size  # noqa
        self._ofn = locals()

        cfg = argparse.Namespace(
            pretrained_checkpoint=f"{ckpts_root}/{model_name}",
            model_family="openvla", use_l1_regression=True, use_diffusion=False,
            num_diffusion_steps_inference=50, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK, lora_rank=32,
            unnorm_key=list(json.load(open(
                f"{ckpts_root}/{model_name}/dataset_statistics.json"
            )).keys())[0],
            load_in_8bit=False, load_in_4bit=False,
        )
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.head = get_action_head(cfg, self.vla.llm_dim)
        self.proj = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)
        self.proc = get_processor(cfg)
        self.resize = get_image_resize_size(
            argparse.Namespace(model_family="openvla"))
        self._prep = (
            get_libero_image, get_libero_wrist_image, quat2axisangle,
            resize_image_for_policy)

    def prepare_observation(self, obs: dict) -> dict:
        gi, gwi, q2a, rip = self._prep
        return {
            "full_image": rip(gi(obs), self.resize),
            "wrist_image": rip(gwi(obs), self.resize),
            "state": np.concatenate(
                (obs["robot0_eef_pos"], q2a(obs["robot0_eef_quat"]),
                 obs["robot0_gripper_qpos"])),
        }

    def act(self, observation: dict, task: str) -> list[np.ndarray]:
        return self._ofn["get_vla_action"](
            self.cfg, self.vla, self.proc, observation, task,
            action_head=self.head, proprio_projector=self.proj)


class LeRobotWrapper:
    """LeRobot-format policy (pi0-fast / smolvla; runs in `smolvla` env)."""

    def __init__(self, policy_path: str, model_name: str,
                 tokenizer_path: str | None, action_tokenizer_path: str | None,
                 observation_size: int = 360) -> None:
        sys.path.insert(0, "/root/autodl-tmp/RASE")
        from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states  # noqa
        from rase.backends.libero_plus_paths import ensure_libero_plus_paths  # noqa
        from rase.collect.forked_rollout import load_lerobot_policy_bundle  # noqa
        from rase.collect.forked_rollout import InProcessLeRobotContinuation  # noqa
        ensure_libero_plus_paths()
        _patch_lerobot_init_states()
        self._ofn = locals()
        self.model_name = model_name
        self.obs_size = observation_size
        self.bundle = load_lerobot_policy_bundle(
            policy_path, device="cuda", num_steps=10, n_action_steps=10,
            tokenizer_path=tokenizer_path,
            action_tokenizer_path=action_tokenizer_path,
            observation_height=observation_size,
            observation_width=observation_size,
        )
        self.cont = InProcessLeRobotContinuation(self.bundle, seed=7, capture=False)

    def prepare_observation(self, obs: dict) -> dict:
        # NOTE: not used for act(); see main loop — LeRobot observations must
        # be formatted via rase.collect.pool_candidates.observation_from_libero_env
        return obs

    def act(self, observation: dict, task: str) -> np.ndarray:
        a = self.cont.act(observation, task=task)
        arr = np.asarray(a, dtype=np.float64)
        return arr.reshape(-1)  # continuation returns (1, 7) batched


TASK_MAX_STEPS = {
    "libero_spatial": 520, "libero_object": 520, "libero_goal": 520,
    "libero_10": 520, "libero_90": 400,
}

CHUNK_STEP = 10  # lerobot native requery horizon


def chunk_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
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


def process_openvla_action(action):
    from experiments.robot.robot_utils import (  # noqa
        normalize_gripper_action, invert_gripper_action,
    )
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def get_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def lerobot_proprio(env) -> list[float]:
    """8-d proprio in the OpenVLA layout from the robosuite env."""
    try:
        unwrapped = env.envs[0].unwrapped
        full = unwrapped.get_observation()
        from experiments.robot.libero.libero_utils import quat2axisangle  # noqa
        return [float(x) for x in np.concatenate([
            full["robot0_eef_pos"],
            quat2axisangle(full["robot0_eef_quat"]),
            full["robot0_gripper_qpos"],
        ])]
    except Exception:
        return [0.0] * 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-type", choices=["openvla", "lerobot"], required=True)
    parser.add_argument("--model", default="oft_goal")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--action-tokenizer-path", default=None)
    parser.add_argument("--ckpts-root", default="/root/autodl-tmp/RASE/ckpts")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=20)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]

    # Ensure the STANDARD libero package (10-task suites, names matching the
    # matrix) is used even in the smolvla env, where the editable `libero`
    # points at LIBERO-plus (2518 table-variant tasks).  Register the standard
    # package under `libero.libero` in sys.modules before any import.
    if args.vla_type == "lerobot":
        import importlib.util as _ilu
        std_libero = "/root/autodl-tmp/envs/oft/lib/python3.10/site-packages/libero/libero"
        if "libero.libero" not in sys.modules and Path(std_libero).is_dir():
            _spec = _ilu.spec_from_file_location(
                "libero.libero", f"{std_libero}/__init__.py")
            _mod = _ilu.module_from_spec(_spec)
            _mod.__path__ = [std_libero]
            sys.modules["libero.libero"] = _mod
            _spec.loader.exec_module(_mod)

    if args.vla_type == "openvla":
        model = OpenVLAWrapper(args.model, args.ckpts_root)
    else:
        model = LeRobotWrapper(
            args.policy_path, args.model, args.tokenizer_path,
            args.action_tokenizer_path)

    # env (server-side import)
    if args.vla_type == "openvla":
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
    else:
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv
        import gymnasium as gym
        get_libero_dummy_action = lambda: None  # noqa: E731

    benchmark_dict = benchmark.get_benchmark_dict()
    rows: list[dict] = []
    for i, task in enumerate(tasks):
        task_suite = benchmark_dict[task["suite"]]()
        task_obj = task_index = None
        for idx, t in enumerate(task_suite.tasks):
            if t.language == task["task"]:
                task_obj, task_index = t, idx
                break
        if task_obj is None:
            print(f"[zs] WARN task not found: {task['task'][:40]}", flush=True)
            continue
        initial_states = task_suite.get_task_init_states(task_index)
        if args.vla_type == "openvla":
            env, task_description = get_libero_env(
                task_obj, "openvla", resolution=args.env_img_res)
        else:
            def make_single():
                return LiberoEnv(
                    task_suite=task_suite, task_id=task_index,
                    task_suite_name=task["suite"],
                    camera_name="agentview_image,robot0_eye_in_hand_image",
                    init_states=True, episode_index=0, n_envs=1,
                    obs_type="pixels_agent_pos",
                    observation_height=model.obs_size,
                    observation_width=model.obs_size,
                    control_mode="relative",
                )
            env = gym.vector.SyncVectorEnv([make_single])
            task_description = str(getattr(
                env.envs[0], "task_description", task["task"]))

        for ep in range(args.episodes):
            if args.vla_type == "openvla":
                env.reset()
                obs = env.set_init_state(
                    initial_states[ep % len(initial_states)])
            else:
                obs, _ = env.reset(seed=[ep % len(initial_states)])
            queue: list[np.ndarray] = []
            recent: list[np.ndarray] = []  # lerobot: last CHUNK_STEP actions
            decision_idx = 0
            ep_rows: list[dict] = []
            t = 0
            max_steps = TASK_MAX_STEPS[task["suite"]]
            success = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    if args.vla_type == "openvla":
                        obs, _, done, info = env.step(
                            get_libero_dummy_action("openvla"))
                    else:
                        obs, _, term, trunc, info = env.step(
                            np.asarray(get_dummy_action(), dtype=np.float32)
                            .reshape(1, -1))
                        done = bool(np.asarray(term).reshape(-1)[0])
                    t += 1
                    continue
                observation = model.prepare_observation(obs)
                if args.vla_type == "openvla":
                    if len(queue) == 0:
                        chunk = model.act(observation, task_description)
                        queue = list(chunk)
                        ep_rows.append({
                            "task": task["task"], "suite": task["suite"],
                            "model": args.model, "episode_idx": ep,
                            "decision_idx": decision_idx,
                            "proprio": [float(x) for x in
                                        observation["state"]],
                            **chunk_stats(np.asarray(chunk)),
                        })
                        decision_idx += 1
                    action = process_openvla_action(queue.pop(0))
                    obs, _, done, info = env.step(action.tolist())
                else:
                    from rase.collect.pool_candidates import (  # noqa
                        observation_from_libero_env,
                    )
                    observation = observation_from_libero_env(env.envs[0])
                    action = model.act(observation, task_description)
                    recent.append(np.asarray(action, dtype=np.float64))
                    obs, _, term, trunc, info = env.step(
                        np.asarray(action, dtype=np.float32).reshape(1, -1))
                    done = bool(np.asarray(term).reshape(-1)[0])
                    if len(recent) == CHUNK_STEP:
                        ep_rows.append({
                            "task": task["task"], "suite": task["suite"],
                            "model": args.model, "episode_idx": ep,
                            "decision_idx": decision_idx,
                            "proprio": lerobot_proprio(env),
                            **chunk_stats(np.stack(recent)),
                        })
                        decision_idx += 1
                        recent = []
                if done:
                    success = True
                    break
                t += 1
            for row in ep_rows:
                row["success"] = int(success)
            rows.extend(ep_rows)
        if args.vla_type == "openvla":
            env.close()
        print(f"[zs] {i + 1}/{len(tasks)} {task['task'][:40]} rows={len(rows)}",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    n_ep = len(set((r["task"], r["episode_idx"]) for r in rows))
    print(json.dumps({
        "model": args.model, "rows": len(rows),
        "success_rate": sum(r["success"] for r in rows) / max(1, n_ep),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

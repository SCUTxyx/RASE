#!/usr/bin/env python3
"""RASE action-level closed-loop demo with the distilled risk evaluator.

At every decision point (every 8 steps) both OFT models (A=oft_spatial,
B=oft_object) generate a candidate chunk; the candidate-level risk evaluator
scores P(success | proprio, chunk stats, instruction) for each candidate and
the higher-scoring chunk is executed.  Baselines: fixed A / fixed B (matrix
data) and episode-level selection (93.5% measured).

Usage:
  python rase_oft_action_loop.py \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --selector runs/oft_opportunity/oft_risk_model.npz \
    --vocab runs/oft_opportunity/oft_risk_vocab.json \
    --output runs/oft_opportunity/action_loop.json \
    --num-trials-per-task 4
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

CHUNK_KEYS = [
    "chunk_mean_pos", "chunk_mean_rot", "chunk_std_pos", "chunk_std_rot",
    "chunk_gripper_mean", "chunk_gripper_std", "chunk_total_disp",
    "chunk_norm_mean",
]


class DualModel:
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


class RiskModel:
    def __init__(self, npz_path: Path, vocab_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as z:
            self.mean = z["mean"]
            self.scale = z["scale"]
            self.weights = z["weights"]
            self.intercept = float(z["intercept"])
        self.vocab = json.loads(vocab_path.read_text())
        self._priors: dict[str, np.ndarray] = {}

    def _prior(self, model_name: str) -> np.ndarray:
        if model_name not in self._priors:
            stats = json.load(open(
                f"/root/autodl-tmp/RASE/ckpts/{model_name}/dataset_statistics.json"))
            key = list(stats.keys())[0]
            act = stats[key]["action"]
            mean = np.asarray(act["mean"], dtype=np.float64)[:7]
            span = np.asarray(act["q99"], dtype=np.float64)[:7] - np.asarray(
                act["q01"], dtype=np.float64)[:7]
            self._priors[model_name] = np.concatenate([mean, span])
        return self._priors[model_name]

    def _bigram(self, text: str) -> np.ndarray:
        text = text.lower()
        x = np.zeros(len(self.vocab), dtype=np.float64)
        for i in range(len(text) - 1):
            idx = self.vocab.get(text[i:i + 2])
            if idx is not None:
                x[idx] += 1.0
        return x

    def _feature(self, proprio: np.ndarray, chunk: list[np.ndarray],
                 task: str, model_name: str) -> np.ndarray:
        arr = np.asarray(chunk, dtype=np.float64)
        pos = arr[:, :3]
        stats = [
            arr[:, :3].mean(axis=0), arr[:, 3:6].mean(axis=0),
            arr[:, :3].std(axis=0), arr[:, 3:6].std(axis=0),
            float(arr[:, 6].mean()), float(arr[:, 6].std()),
            float(np.abs(np.diff(pos, axis=0)).sum()),
            float(np.linalg.norm(pos, axis=1).mean()),
        ]
        return np.concatenate([
            np.asarray(proprio, dtype=np.float64),
            np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in stats]),
            self._bigram(task),
            self._prior(model_name),
        ])

    def score(self, proprio: np.ndarray, chunk: list[np.ndarray], task: str,
              model_name: str) -> float:
        x = self._feature(proprio, chunk, task, model_name)
        xs = (x - self.mean) / self.scale
        logit = self.intercept + xs @ self.weights
        return float(1.0 / (1.0 + np.exp(-logit)))


class TaskPrior:
    """Episode-level instruction selector: P(B better | task text)."""

    def __init__(self, npz_path: Path, vocab_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as z:
            self.mean = z["mean"]
            self.scale = z["scale"]
            self.weights = z["weights"]
            self.intercept = float(z["intercept"])
        self.vocab = json.loads(vocab_path.read_text())

    def choose_b(self, task: str) -> float:
        text = task.lower()
        x = np.zeros(len(self.vocab), dtype=np.float64)
        for i in range(len(text) - 1):
            idx = self.vocab.get(text[i:i + 2])
            if idx is not None:
                x[idx] += 1.0
        xs = (x - self.mean) / self.scale
        logit = self.intercept + xs @ self.weights
        return float(1.0 / (1.0 + np.exp(-logit)))


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    observation = {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist_img, resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
             obs["robot0_gripper_qpos"])
        ),
    }
    return observation


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=4)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--abstain-margin", type=float, default=0.03,
                        help="if |s_b - s_a| < margin, follow the task prior "
                             "(episode-level instruction selector)")
    parser.add_argument("--prior", type=Path, default=None,
                        help="task-prior npz (with_mean/scale/weights/intercept); "
                             "required when abstain-margin > 0")
    parser.add_argument("--prior-table", type=Path, default=None,
                        help="JSON {task: 'A'|'B'}: per-task primary model "
                             "(monitor-and-switch mode)")
    parser.add_argument("--switch-threshold", type=float, default=0.55,
                        help="monitor-and-switch: if primary candidate score "
                             "drops below this, try the other model (persistent)")
    parser.add_argument("--score-log", type=Path, default=None,
                        help="append per-decision-point scores to this JSONL")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]
    benchmark_dict = benchmark.get_benchmark_dict()
    resize_size = get_image_resize_size(argparse.Namespace(model_family="openvla"))
    dual = DualModel()
    risk = RiskModel(args.selector, args.vocab)
    prior = TaskPrior(args.prior, args.vocab) if args.prior is not None else None
    prior_table = json.loads(args.prior_table.read_text()) if args.prior_table else None
    print("[risk] evaluator loaded", flush=True)

    def primary_model(task_text: str) -> str:
        if prior_table is not None:
            return "oft_object" if prior_table.get(task_text, "A") == "B" else "oft_spatial"
        if prior is not None:
            return "oft_object" if prior.choose_b(task_text) > 0.5 else "oft_spatial"
        return "oft_spatial"

    report = {"schema": "rase-soft-action-loop/v1", "tasks": {}}
    n_ok = n_ep = 0
    n_choose_a = n_choose_b = 0
    n_decisions = 0
    n_switches = 0
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
        current = primary_model(task_description)
        ok = 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            queue: list[np.ndarray] = []
            t = 0
            max_steps = TASK_MAX_STEPS[task["suite"]]
            success = False
            ep_switches = 0
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, info = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue
                observation = prepare_observation(obs, resize_size)
                if len(queue) == 0:
                    chunk_cur = dual.act(current, observation, task_description)
                    s_cur = risk.score(observation["state"], chunk_cur,
                                       task_description, current)
                    n_decisions += 1
                    if s_cur < args.switch_threshold:
                        other = ("oft_object" if current == "oft_spatial"
                                 else "oft_spatial")
                        chunk_oth = dual.act(other, observation, task_description)
                        s_oth = risk.score(observation["state"], chunk_oth,
                                           task_description, other)
                        if s_oth > s_cur + 0.02:
                            current = other
                            chunk_cur = chunk_oth
                            ep_switches += 1
                    if args.score_log is not None:
                        with args.score_log.open("a") as fh:
                            fh.write(json.dumps({
                                "task": task["task"], "suite": task["suite"],
                                "episode": ep, "decision": t,
                                "s_cur": s_cur, "current": current,
                                "switches": ep_switches,
                            }) + "\n")
                    if current == "oft_object":
                        queue = list(chunk_cur)
                        n_choose_b += 1
                    else:
                        queue = list(chunk_cur)
                        n_choose_a += 1
                action = process_action(queue.pop(0))
                obs, _, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1
            n_switches += ep_switches
            ok += int(success)
            n_ok += int(success)
            n_ep += 1
        env.close()
        report["tasks"][task["task"][:60]] = {
            "suite": task["suite"], "success": ok, "episodes": args.num_trials_per_task,
            "A_rate": task["A"], "B_rate": task["B"],
            "oracle_rate": max(task["A"], task["B"]),
        }
        print(f"[{i + 1}/{len(tasks)}] {task['task'][:45]} -> {ok}/{args.num_trials_per_task}",
              flush=True)

    report["overall"] = {
        "success": n_ok, "episodes": n_ep,
        "rate": n_ok / n_ep if n_ep else None,
    }
    report["selection_stats"] = {
        "decisions": n_decisions,
        "choose_A": n_choose_a, "choose_B": n_choose_b,
        "choose_B_fraction": n_choose_b / max(1, n_decisions),
        "episode_switches": n_switches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

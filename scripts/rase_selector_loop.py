#!/usr/bin/env python3
"""Stage D: pure risk-driven closed loop with the uncertainty-aware selector
(no lookup, no task router).  Uses an OPD-v2 risk model (any C0..C4 variant;
the npz must carry feature_version) and selector_risk.decide.

Decision point: both candidate models generate a chunk; risk model predicts
mu (mean success) and sigma (uncertainty) per candidate; the selector decides
continue / switch / fallback / abort with LCB/UCB, dwell and emergency rules.

Baselines are compared in the analysis step (A-only, B-only, best-fixed,
random, oracle).

Usage (server, oft env):
  python rase_selector_loop.py \
    --selector runs/oft_opportunity/opd_v2_C2.npz \
    --vocab runs/oft_opportunity/oft_risk_vocab.json \
    --matrix runs/oft_opportunity/oft_matrix_analysis.json \
    --output runs/oft_opportunity/selector_loop.json \
    --num-trials-per-task 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rase_common import build_bigram_vocab  # noqa: E402
from selector_risk import (  # noqa: E402
    SelectorConfig, SelectorState, decide, ridge_uncertainty,
)

TASK_MAX_STEPS = {
    "libero_spatial": 520, "libero_object": 520, "libero_goal": 520,
    "libero_10": 520, "libero_90": 400,
}


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class DualModel:
    def __init__(self, models: list[str], ckpts_root: str) -> None:
        ROOT = Path("/root/autodl-tmp/openvla-oft")
        sys.path.insert(0, str(ROOT))
        from experiments.robot.openvla_utils import (  # noqa
            get_vla, get_processor, get_action_head, get_proprio_projector,
            get_vla_action,
        )
        from experiments.robot.libero.libero_utils import (  # noqa
            get_libero_env, get_libero_dummy_action, get_libero_image,
            get_libero_wrist_image, quat2axisangle,
        )
        from experiments.robot.robot_utils import (  # noqa
            get_image_resize_size, normalize_gripper_action,
            invert_gripper_action,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa
        from libero.libero import benchmark  # noqa
        self._ofn = locals()
        self.models = models
        self.residents: dict[str, dict] = {}
        for name in models:
            cfg = argparse.Namespace(
                pretrained_checkpoint=f"{ckpts_root}/{name}",
                model_family="openvla", use_l1_regression=True,
                use_diffusion=False, num_diffusion_steps_inference=50,
                use_film=False, num_images_in_input=2, use_proprio=True,
                center_crop=True, num_open_loop_steps=NUM_ACTIONS_CHUNK,
                lora_rank=32,
                unnorm_key=list(json.load(open(
                    f"{ckpts_root}/{name}/dataset_statistics.json"
                )).keys())[0],
                load_in_8bit=False, load_in_4bit=False,
            )
            vla = get_vla(cfg)
            head = get_action_head(cfg, vla.llm_dim)
            proj = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
            proc = get_processor(cfg)
            self.residents[name] = {"cfg": cfg, "vla": vla, "head": head,
                                    "proj": proj, "proc": proc}
            print(f"[dual] loaded {name}", flush=True)

    def act(self, name: str, observation: dict, task: str) -> list[np.ndarray]:
        m = self.residents[name]
        return self._ofn["get_vla_action"](
            m["cfg"], m["vla"], m["proc"], observation, task,
            action_head=m["head"], proprio_projector=m["proj"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="oft_spatial,oft_object")
    parser.add_argument("--ckpts-root", default="/root/autodl-tmp/RASE/ckpts")
    parser.add_argument("--num-trials-per-task", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--dwell", type=int, default=2)
    parser.add_argument("--emergency", type=float, default=0.35)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    args = parser.parse_args()

    # ---- risk model artifacts ----
    z = load_npz(args.selector)
    mean = z["mean"]; scale = z["scale"]; weights = z["weights"]
    intercept = float(z["intercept"])
    feature_version = str(z.get("feature_version", b"?"))
    if isinstance(feature_version, bytes):
        feature_version = feature_version.decode()
    vocab = json.loads(args.vocab.read_text())
    print(f"[sel] model feature_version={feature_version}", flush=True)

    models = [m.strip() for m in args.models.split(",")]
    dual = DualModel(models, args.ckpts_root)
    cfg_sel = SelectorConfig(beta=args.beta, delta=args.delta, dwell=args.dwell,
                             emergency=args.emergency)
    st = SelectorState(current=models[0])

    # ---- env + rollout ----
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_dummy_action,
    )
    from experiments.robot.libero.libero_utils import (
        get_libero_image, get_libero_wrist_image, quat2axisangle,
    )
    from experiments.robot.robot_utils import (
        get_image_resize_size, normalize_gripper_action, invert_gripper_action,
    )
    from experiments.robot.openvla_utils import resize_image_for_policy

    matrix = json.loads(args.matrix.read_text())
    tasks = [t for t in matrix["per_task"][: args.tasks]]
    benchmark_dict = benchmark.get_benchmark_dict()
    resize = get_image_resize_size(argparse.Namespace(model_family="openvla"))

    def prep(obs) -> dict:
        return {
            "full_image": resize_image_for_policy(get_libero_image(obs), resize),
            "wrist_image": resize_image_for_policy(
                get_libero_wrist_image(obs), resize),
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                 obs["robot0_gripper_qpos"])),
        }

    def feats_of(observation: dict, chunk: np.ndarray, task: str,
                 model: str) -> np.ndarray:
        # B1 layout: [s_t proprio(8), canonical chunk(24), bigram(V)]
        # identity-free: model name is NOT a feature.
        from rase_common import canonical_chunk_features
        arr = np.asarray(chunk, dtype=np.float64)
        parts = [np.asarray(observation["state"], dtype=np.float64)]
        parts.append(canonical_chunk_features(arr))
        t = task.lower()
        x = np.zeros(len(vocab))
        for i in range(len(t) - 1):
            idx = vocab.get(t[i:i + 2])
            if idx is not None:
                x[idx] += 1.0
        parts.append(x)
        return np.concatenate(parts)

    report = {"schema": "rase-stage-d-selector-loop/v1", "tasks": {}}
    n_ok = n_ep = 0
    n_switches = n_abstains = n_aborts = 0
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
        ok = 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            queue: list[np.ndarray] = []
            current = models[0]
            t = 0
            max_steps = TASK_MAX_STEPS[task["suite"]]
            success = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, info = env.step(
                        get_libero_dummy_action("openvla"))
                    t += 1
                    continue
                observation = prep(obs)
                if len(queue) == 0:
                    chunks = {m: dual.act(m, observation, task_description)
                              for m in models}
                    X = np.stack([
                        feats_of(observation, np.asarray(chunks[m]),
                                 task_description, m) for m in models])
                    mu = {m: float(1.0 / (1.0 + np.exp(-(
                        intercept + ((X[j] - mean) / scale) @ weights))))
                        for j, m in enumerate(models)}
                    sig = ridge_uncertainty(
                        X, mean, scale, weights, intercept)
                    sigma = {m: float(sig[j]) for j, m in enumerate(models)}
                    chosen, info = decide(mu, sigma, cfg_sel, st)
                    if info["mode"] == "switch":
                        n_switches += 1
                    elif info["mode"] == "abstain":
                        n_abstains += 1
                    if chosen == "abort":
                        n_aborts += 1
                        break
                    current = chosen if chosen in models else current
                    queue = list(chunks[current])
                action = normalize_gripper_action(queue.pop(0), binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1
            ok += int(success)
            n_ok += int(success)
            n_ep += 1
        env.close()
        report["tasks"][task["task"][:60]] = {
            "suite": task["suite"], "success": ok,
            "episodes": args.num_trials_per_task,
            "A_rate": task["A"], "B_rate": task["B"],
            "oracle_rate": max(task["A"], task["B"]),
        }
        print(f"[{i + 1}/{len(tasks)}] {task['task'][:45]} -> {ok}/"
              f"{args.num_trials_per_task}", flush=True)

    report["overall"] = {"success": n_ok, "episodes": n_ep,
                         "rate": n_ok / n_ep if n_ep else None}
    report["selector_stats"] = {
        "switches": n_switches, "abstains": n_abstains, "aborts": n_aborts,
        "decisions_log": st.history[-50:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

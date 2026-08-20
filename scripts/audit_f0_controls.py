#!/usr/bin/env python3
"""PRE-C0-R0 Step 1.4: F0 Control Experiments.

Runs alpha-scaling (0.0-1.25), sign-flip, random norm-matched, and dim-shuffled
corrections on spatial dev episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def generate_control_deltas(c_vec, seed=42):
    norm_c = np.linalg.norm(c_vec)
    rng = np.random.RandomState(seed)
    random_dir = rng.randn(7)
    random_dir = random_dir / np.linalg.norm(random_dir) * norm_c
    perm = rng.permutation(7)
    shuffled = c_vec[perm]

    deltas = dict(
        c_original=c_vec.copy(),
        sign_flip=-c_vec.copy(),
        random_norm_matched=random_dir.astype(np.float32),
        dim_shuffled=shuffled.astype(np.float32),
    )
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        deltas["alpha_{:.2f}".format(alpha)] = (c_vec * alpha).astype(np.float32)
    return deltas


def run_single_arm(entry, bundle, delta_vec, max_steps, arm_label, arm_config=""):
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    task_id = entry["task_id"]
    seed_val = entry["seed"]
    init_state = entry["init_state_id"] % 50

    seed_everything(seed_val)
    bundle["policy"].reset()

    handle = make_libero_env_for_task(task_id, init_state_id=init_state,
                                       seed=seed_val, libero_flavor="clean")
    instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")

    obs = observation_from_libero_env(handle.vector_env.envs[0])
    for t in range(max_steps):
        student_action = select_env_action(bundle, obs, task=instruction)

        if delta_vec is not None:
            delta_clipped = np.clip(delta_vec, -0.5, 0.5)
            mixed = np.clip(student_action.flatten() + delta_clipped, -1.0, 1.0)
            action = mixed.reshape(1, -1)
        else:
            action = student_action

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            handle.close()
            return dict(success=success_from_info(info), steps=t + 1,
                        arm=arm_label, arm_config=arm_config,
                        task_id=task_id, seed=int(seed_val),
                        init_state_id=int(init_state),
                        suite=entry.get("suite", "libero_spatial"))

    handle.close()
    return dict(success=False, steps=max_steps, arm=arm_label,
                arm_config=arm_config, task_id=task_id,
                seed=int(seed_val), init_state_id=int(init_state),
                suite=entry.get("suite", "libero_spatial"))


def build_manifest(protocol, suite, n_per_task, base_seed):
    task_ids = protocol["splits"][suite]["dev"]
    manifest = []
    for task_id in task_ids[:2]:
        for ep_i in range(n_per_task):
            for init_state in range(5):
                seed_val = (base_seed * 31 + ep_i * 7 + init_state * 97) % (2**31)
                manifest.append(dict(
                    task_id=task_id,
                    init_state_id=init_state % 50,
                    seed=seed_val,
                    suite=suite,
                ))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "runs/route_c_final/protocol_frozen.json")
    parser.add_argument("--f0-vector", type=Path,
                        default=ROOT / "runs/pre_c0_r0/f0_constant_vector.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=20260807)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    f0_data = json.loads(args.f0_vector.read_text(encoding="utf-8"))
    c_vec = np.array(f0_data["f0_constant_vector_c"], dtype=np.float32)
    print("F0 c = [{}] |c|={:.6f}\n".format(
        ", ".join("{:+.6f}".format(x) for x in c_vec),
        np.linalg.norm(c_vec)))

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    from rase.collect.forked_rollout import load_smolvla_policy_bundle
    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    manifest = build_manifest(protocol, args.suite, args.n_per_task, args.base_seed)
    print("Manifest: {} episodes".format(len(manifest)))

    control_deltas = generate_control_deltas(c_vec, seed=42)

    print("Control arms:")
    for label, delta in control_deltas.items():
        print("  {}: [{}] |delta|={:.4f}".format(
            label,
            ", ".join("{:+.4f}".format(x) for x in delta),
            np.linalg.norm(delta)))

    arm_order = [
        ("B0_baseline", None, ""),
    ] + [
        ("alpha_{:.2f}".format(a), control_deltas["alpha_{:.2f}".format(a)],
         "alpha={:.2f}".format(a))
        for a in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    ] + [
        ("sign_flip", control_deltas["sign_flip"], "sign_flip"),
        ("random", control_deltas["random_norm_matched"], "random_norm_matched"),
        ("dim_shuffled", control_deltas["dim_shuffled"], "dim_shuffled"),
    ]

    summary = {}
    for arm_label, delta, arm_config in arm_order:
        print("\n--- {} ---".format(arm_label))
        arm_results = []
        for i, entry in enumerate(manifest):
            r = run_single_arm(entry, bundle, delta, args.max_steps,
                               arm_label, arm_config=arm_config)
            arm_results.append(r)
            n_success = sum(1 for rr in arm_results if rr["success"])
            print("  [{}/{}] {} s={} success={} ({}/{})".format(
                i + 1, len(manifest), entry["task_id"], entry["seed"],
                r["success"], n_success, len(arm_results)))

        arm_path = output_dir / "f0_controls_{}.jsonl".format(
            arm_label.replace(".", "_"))
        with open(arm_path, "w") as f:
            for r in arm_results:
                f.write(json.dumps(r) + "\n")
        n_success = sum(1 for r in arm_results if r["success"])
        summary[arm_label] = dict(N=len(arm_results), success=n_success,
                                   rate=n_success / max(len(arm_results), 1))
        print("  {}: {}/{} = {:.1%}".format(
            arm_label, n_success, len(arm_results),
            n_success / max(len(arm_results), 1)))

    print("\n" + "=" * 60)
    print("CONTROLS SUMMARY")
    print("=" * 60)
    for arm_label, _, _ in arm_order:
        if arm_label in summary:
            s = summary[arm_label]
            print("{:<20} {:>5} {:>8} {:>9.1%}".format(
                arm_label, s["N"], s["success"], s["rate"]))

    summary_path = output_dir / "f0_controls_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nSaved to: {}".format(summary_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

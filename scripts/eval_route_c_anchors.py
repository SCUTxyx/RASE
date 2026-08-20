#!/usr/bin/env python3
"""S1: Held-out anchor recovery assessment.

For each R0 episode, picks a random teacher_recovery snapshot and computes:
  C0: ||student_action - teacher_action||  (baseline, should be > 0)
  C3: ||(student_action + delta_a) - teacher_action||  (plugin)
  C1: ||teacher_action - teacher_action|| = 0  (OFT upper bound)

Computes rescue rate: fraction of snapshots where C3 < C0.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.recovery.residual_plugin import ResidualRecoveryPlugin


def load_episodes(data_dir: Path, mode: str = "R0") -> list[dict]:
    episodes = []
    subdir = data_dir / mode
    if not subdir.is_dir():
        print(f"WARNING: {subdir} does not exist")
        return episodes
    for f in sorted(subdir.glob("*.json")):
        data = json.loads(f.read_text())
        episodes.append(data)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"Checkpoint hidden_dim={cfg.get('hidden_dim', '?')} "
          f"obs_feature_dim={cfg.get('obs_feature_dim', '?')} "
          f"history_window={cfg.get('history_window', '?')}")

    plugin = ResidualRecoveryPlugin(
        proprio_dim=cfg.get("proprio_dim", 8),
        action_dim=cfg.get("action_dim", 7),
        history_window=cfg.get("history_window", 8),
        obs_feature_dim=cfg.get("obs_feature_dim", 144),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        delta_clip=cfg.get("delta_clip", 0.5),
    )
    plugin.load_state_dict(ckpt["model_state_dict"])
    plugin.to(args.device)
    plugin.eval()

    episodes = load_episodes(args.data_dir, "R0")
    print(f"Loaded {len(episodes)} R0 episodes")

    random.seed(args.seed)
    anchors = []

    for ep_idx, ep in enumerate(episodes):
        task_id = ep.get("task_id", "unknown")
        recovery_steps = ep.get("teacher_recovery", [])
        if not recovery_steps:
            print(f"  Episode {ep_idx} ({task_id}): no recovery steps, skipping")
            continue

        # Pick random anchor timestep
        anchor_idx = random.randint(0, len(recovery_steps) - 1)
        step = recovery_steps[anchor_idx]

        obs_features = np.array(step["obs_features"], dtype=np.float32).reshape(-1)[:144]
        history = np.array(step["history_before"], dtype=np.float32)
        student_act = np.array(step["action"], dtype=np.float32).reshape(-1)[:7]
        teacher_act = np.array(step["teacher_action"], dtype=np.float32).reshape(-1)[:7]

        h_t = torch.from_numpy(history).float().unsqueeze(0).to(args.device)
        o_t = torch.from_numpy(obs_features).float().unsqueeze(0).to(args.device)
        a_t = torch.from_numpy(student_act).float().unsqueeze(0).to(args.device)

        with torch.no_grad():
            delta_a = plugin(h_t, o_t, a_t).squeeze(0).cpu().numpy()
        plugin_act = student_act + delta_a

        # C0: student → teacher distance (baseline)
        c0_l2 = float(np.linalg.norm(student_act - teacher_act))
        # C1: teacher → teacher = 0 (upper bound)
        c1_l2 = 0.0
        # C3: student+plugin → teacher distance
        c3_l2 = float(np.linalg.norm(plugin_act - teacher_act))

        # Cosine similarity: student→teacher vs plugin→teacher
        student_vec = student_act - teacher_act
        plugin_vec = plugin_act - teacher_act
        c0_norm = np.linalg.norm(student_vec)
        c3_norm = np.linalg.norm(plugin_vec)
        cosine = float(np.dot(student_vec, plugin_vec) / max(c0_norm * c3_norm, 1e-8))

        improvement = (c0_l2 - c3_l2) / max(c0_l2, 1e-8) * 100

        anchor = {
            "episode_idx": ep_idx,
            "task_id": task_id,
            "anchor_idx": anchor_idx,
            "total_recovery_steps": len(recovery_steps),
            "c0_student_l2": c0_l2,
            "c1_teacher_l2": c1_l2,
            "c3_plugin_l2": c3_l2,
            "improvement_pct": improvement,
            "cosine_similarity": cosine,
            "rescue": c3_l2 < c0_l2,
            "c0_has_headroom": c0_l2 > 0.01,
        }
        anchors.append(anchor)

        print(f"  [{ep_idx+1}/{len(episodes)}] {task_id} "
              f"C0={c0_l2:.4f} C3={c3_l2:.4f} "
              f"improvement={improvement:+.1f}% "
              f"rescue={anchor['rescue']}")

    if not anchors:
        print("No anchors found!")
        return 1

    n_rescue = sum(1 for a in anchors if a["rescue"])
    n_headroom = sum(1 for a in anchors if a["c0_has_headroom"])
    n_better_than_teacher = sum(1 for a in anchors if a["c3_plugin_l2"] < a["c1_teacher_l2"])

    rescue_rate = n_rescue / len(anchors)
    c0_vals = [a["c0_student_l2"] for a in anchors]
    c3_vals = [a["c3_plugin_l2"] for a in anchors]
    imp_vals = [a["improvement_pct"] for a in anchors]

    rescue_pass = rescue_rate >= 0.6
    headroom_pass = n_headroom == len(anchors)
    no_data_leak = n_better_than_teacher == 0

    print()
    print("=" * 60)
    print("HELD-OUT ANCHOR RESULTS")
    print("=" * 60)
    print(f"  Anchors: {len(anchors)}/{len(episodes)} episodes")
    print(f"  C0 (student) mean={np.mean(c0_vals):.4f} median={np.median(c0_vals):.4f}")
    print(f"  C3 (plugin)  mean={np.mean(c3_vals):.4f} median={np.median(c3_vals):.4f}")
    print(f"  Improvement:  mean={np.mean(imp_vals):.1f}% median={np.median(imp_vals):.1f}%")
    print()
    print(f"  Rescue rate: {n_rescue}/{len(anchors)} = {rescue_rate:.1%}  "
          f"{'PASS' if rescue_pass else 'FAIL'} (>=60%)")
    print(f"  Headroom:    {n_headroom}/{len(anchors)}  "
          f"{'PASS' if headroom_pass else 'FAIL'} (C0>0)")
    print(f"  No data leak: {no_data_leak}  "
          f"{'PASS' if no_data_leak else 'FAIL'} (C3>=C1)")

    gate_pass = rescue_pass and headroom_pass and no_data_leak
    print(f"\n  ANCHOR GATE: {'PASS' if gate_pass else 'FAIL'}")

    results = {
        "gate_pass": gate_pass,
        "rescue_rate": rescue_rate,
        "n_rescue": n_rescue,
        "n_total": len(anchors),
        "headroom_all": headroom_pass,
        "no_data_leak": no_data_leak,
        "c0_mean": float(np.mean(c0_vals)),
        "c0_median": float(np.median(c0_vals)),
        "c3_mean": float(np.mean(c3_vals)),
        "c3_median": float(np.median(c3_vals)),
        "improvement_mean_pct": float(np.mean(imp_vals)),
        "improvement_median_pct": float(np.median(imp_vals)),
        "anchors": anchors,
        "checkpoint": str(args.checkpoint),
    }
    (output_dir / "recovery_anchor_results.json").write_text(
        json.dumps(results, indent=2) + "\n")

    # Also save as jsonl
    with open(output_dir / "recovery_anchor_results.jsonl", "w") as f:
        for a in anchors:
            f.write(json.dumps(a) + "\n")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

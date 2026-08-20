#!/usr/bin/env python3
"""Phase 5.3: Student / Plugin / OFT headroom replay.

For each R0 episode's teacher_recovery steps, compute:
- Student error: ||student_act - teacher_act|| (baseline)
- Plugin error: ||student_act + delta_a - teacher_act|| (our model)
- Recovery rate: what fraction of the gap does the plugin close?
"""

from __future__ import annotations

import argparse
import json
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
        episodes.append(json.load(open(f)))
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"Checkpoint config: {json.dumps(cfg, indent=2)}")

    # Build plugin from checkpoint config
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

    # Load episodes
    episodes = load_episodes(args.data_dir, "R0")
    print(f"Loaded {len(episodes)} R0 episodes")

    all_errors = {"student_l2": [], "plugin_l2": [], "improvement_pct": []}
    episode_metrics = []

    for ep_idx, ep in enumerate(episodes):
        task_id = ep.get("task_id", "unknown")
        recovery_steps = ep.get("teacher_recovery", [])
        if not recovery_steps:
            print(f"  Episode {ep_idx} ({task_id}): no recovery steps")
            continue

        ep_student_l2 = []
        ep_plugin_l2 = []

        for step in recovery_steps:
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

            student_l2 = float(np.linalg.norm(student_act - teacher_act))
            plugin_l2 = float(np.linalg.norm(plugin_act - teacher_act))

            ep_student_l2.append(student_l2)
            ep_plugin_l2.append(plugin_l2)

            all_errors["student_l2"].append(student_l2)
            all_errors["plugin_l2"].append(plugin_l2)
            improvement = (student_l2 - plugin_l2) / max(student_l2, 1e-8)
            all_errors["improvement_pct"].append(improvement * 100)

        mean_student = float(np.mean(ep_student_l2))
        mean_plugin = float(np.mean(ep_plugin_l2))
        mean_imp = (mean_student - mean_plugin) / max(mean_student, 1e-8) * 100
        ep_summary = {
            "task_id": task_id,
            "n_steps": len(recovery_steps),
            "mean_student_l2": mean_student,
            "mean_plugin_l2": mean_plugin,
            "mean_improvement_pct": mean_imp,
        }
        episode_metrics.append(ep_summary)

    if not all_errors["student_l2"]:
        print("No recovery steps found!")
        return 1

    # ── Aggregate ──
    overall = {}
    for k, vals in all_errors.items():
        overall[f"mean_{k}"] = float(np.mean(vals))
        overall[f"median_{k}"] = float(np.median(vals))
        overall[f"std_{k}"] = float(np.std(vals))

    print("\n" + "=" * 60)
    print("HEADROOM REPLAY RESULTS")
    print("=" * 60)
    print(f"  Episodes: {len(episode_metrics)}, Total steps: {len(all_errors['student_l2'])}")
    print(f"  Student L2 (vs OFT): mean={overall['mean_student_l2']:.4f}  median={overall['median_student_l2']:.4f}")
    print(f"  Plugin  L2 (vs OFT): mean={overall['mean_plugin_l2']:.4f}  median={overall['median_plugin_l2']:.4f}")
    print(f"  Improvement:         mean={overall['mean_improvement_pct']:.1f}%  median={overall['median_improvement_pct']:.1f}%")
    print()

    n_improved = sum(1 for ep in episode_metrics if ep["mean_improvement_pct"] > 0)
    n_hurt = len(episode_metrics) - n_improved
    print(f"  Episodes improved: {n_improved}/{len(episode_metrics)}")

    for ep in episode_metrics:
        sign = "+" if ep["mean_improvement_pct"] > 0 else ""
        print(f"    {ep['task_id']}: {sign}{ep['mean_improvement_pct']:.1f}% ({ep['n_steps']} steps)")

    headroom_pass = overall["median_improvement_pct"] > 5.0
    print()
    print(f"  HEADROOM GATE: {'PASS' if headroom_pass else 'FAIL'} "
          f"(median improvement {overall['median_improvement_pct']:.1f}% {'>=' if headroom_pass else '<'} 5.0%)")

    results = {
        "overall": overall,
        "per_episode": episode_metrics,
        "headroom_pass": headroom_pass,
        "checkpoint": str(args.checkpoint),
    }
    (output_dir / "headroom_replay.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if headroom_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PRE-C0-R0 Step 1.1: Extract the F0 constant delta vector.

Loads the trained F0 plugin (all-zero obs features, 128-D),
feeds it representative inputs (history variations + zero obs_features),
and confirms constancy (or lack thereof). Outputs c = mean(delta_pred)
and per-dim std to confirm.
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

from rase.recovery.residual_plugin import load_plugin


def load_r0_delta_targets(data_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load all R0 episodes and extract delta_target and student_action vectors."""
    deltas = []
    actions = []
    r0_dir = data_dir / "R0"
    for ep_path in sorted(r0_dir.glob("*.json")):
        ep = json.loads(ep_path.read_text(encoding="utf-8"))
        for step in ep.get("teacher_recovery", []):
            dt = np.asarray(step.get("delta_target", np.zeros(7)), dtype=np.float32).flatten()[:7]
            # Handle nested list ([[...]]) vs flat list
            if len(dt) != 7 and dt.size >= 7:
                dt = dt[:7]
            if dt.size != 7:
                # Nested: take first element
                dt_flat = np.asarray(dt, dtype=np.float32).flatten()
                dt = dt_flat[:7]
            a = np.asarray(step.get("action", np.zeros(7)), dtype=np.float32).flatten()[:7]
            if len(a) != 7 and a.size >= 7:
                a = a[:7]
            if a.size != 7:
                a_flat = np.asarray(a, dtype=np.float32).flatten()
                a = a_flat[:7]
            deltas.append(dt)
            actions.append(a)
    return deltas, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-ckpt", type=Path,
                        default=ROOT / "runs/route_c_controls/F0/plugin_best.pt")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "runs/route_c_r0_scaled")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plugin = load_plugin(str(args.plugin_ckpt))
    plugin.eval()
    # predict_delta creates CPU tensors internally — keep plugin on CPU
    # or we'd hit device mismatch. For an ~400KB model this is fine.

    print(f"Plugin config: obs_feature_dim={plugin.obs_feature_dim}, "
          f"proprio_dim={plugin.proprio_dim}, action_dim={plugin.action_dim}, "
          f"history_window={plugin.history_window}, delta_clip={plugin.delta_clip}")

    deltas, actions = load_r0_delta_targets(args.data_dir)
    print(f"Loaded {len(deltas)} (delta, action) pairs from R0 data")

    if len(deltas) == 0:
        print("ERROR: No R0 data found. Checking alternative paths...")
        # Try libero_10
        alt = ROOT / "runs/route_c_r0_10"
        if (alt / "R0").is_dir():
            deltas, actions = load_r0_delta_targets(alt)
            print(f"  Loaded {len(deltas)} from {alt}")

    if len(deltas) == 0:
        print("FATAL: No R0 data available.")
        return 1

    # ── Test 1: Zero-everything input (should produce the constant vector)
    history_zeros = np.zeros((plugin.history_window, 8 + 7 + 1 + 7), dtype=np.float32)
    obs_zeros = np.zeros(plugin.obs_feature_dim, dtype=np.float32)
    action_zeros = np.zeros(plugin.action_dim, dtype=np.float32)

    with torch.no_grad():
        delta_zero = plugin.predict_delta(history_zeros, obs_zeros, action_zeros)
    print(f"\nZero-input delta: {delta_zero}")

    # ── Test 2: Real history from R0 data, zero obs_features (F0 deployment)
    all_deltas = []
    step = max(len(deltas) // 100, 1)  # batch every 1% of samples for efficiency

    # Build representative histories from R0 data
    for i, (delta_t, student_a) in enumerate(zip(deltas[::step], actions[::step])):
        # Build a plausible history (pad with repeats of student_action)
        hist = np.zeros((plugin.history_window, 8 + 7 + 1 + 7), dtype=np.float32)
        a_pad = np.zeros(7, dtype=np.float32)
        a_pad[:min(len(student_a), 7)] = student_a[:7]
        for h in range(plugin.history_window):
            hist[h, 8:15] = a_pad      # student_action
            hist[h, 16:23] = a_pad     # executed_action
            hist[h, 15] = float(h) / plugin.history_window  # progress

        with torch.no_grad():
            pred = plugin.predict_delta(hist, obs_zeros, a_pad)
        all_deltas.append(pred)

    # Also test with actual R0 histories
    r0_dir = args.data_dir / "R0"
    real_hist_deltas = []
    for ep_path in sorted(r0_dir.glob("*.json"))[:4]:
        ep = json.loads(ep_path.read_text(encoding="utf-8"))
        for step_data in ep.get("teacher_recovery", [])[:20]:
            hist_data = step_data.get("history_before", None)
            a = np.asarray(step_data.get("action", np.zeros(7)), dtype=np.float32).flatten()[:7]
            if len(a) != 7 and a.size >= 7:
                a = a[:7]
            if a.size != 7:
                continue

            if hist_data is not None and np.asarray(hist_data).size > 0:
                hist_arr = np.asarray(hist_data, dtype=np.float32)
                if hist_arr.ndim == 2 and hist_arr.shape[0] >= plugin.history_window:
                    hist_arr = hist_arr[:plugin.history_window, :]
                else:
                    hist_arr = np.zeros((plugin.history_window, 23), dtype=np.float32)
            else:
                hist_arr = np.zeros((plugin.history_window, 23), dtype=np.float32)

            delta_pred = plugin.predict_delta(hist_arr, obs_zeros,
                                              np.asarray(a, dtype=np.float32))
            real_hist_deltas.append(delta_pred)

    all_deltas_arr = np.stack(all_deltas, axis=0) if all_deltas else np.zeros((0, 7))

    if real_hist_deltas:
        real_hist_arr = np.stack(real_hist_deltas, axis=0)
        all_combined = np.concatenate([all_deltas_arr, real_hist_arr], axis=0)
    else:
        all_combined = all_deltas_arr

    # ── Statistics ──────────────────────────────────────────────────
    mean_delta = all_combined.mean(axis=0)
    std_delta = all_combined.std(axis=0)
    max_delta = all_combined.max(axis=0)
    min_delta = all_combined.min(axis=0)
    range_delta = max_delta - min_delta

    # Per-dim analysis
    dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    per_dim = {}
    for d in range(7):
        per_dim[f"dim_{d}_{dim_names[d]}"] = {
            "mean": float(mean_delta[d]),
            "std": float(std_delta[d]),
            "min": float(min_delta[d]),
            "max": float(max_delta[d]),
            "range": float(range_delta[d]),
            "constancy": "CONSTANT" if float(std_delta[d]) < 1e-4 else
                         "NEAR_CONSTANT" if float(std_delta[d]) < 1e-3 else
                         "VARIABLE",
        }

    # Cosine between zero-input delta and mean across all
    znorm = np.linalg.norm(delta_zero)
    mnorm = np.linalg.norm(mean_delta)
    if znorm > 1e-8 and mnorm > 1e-8:
        cos_zero_vs_mean = float(np.dot(delta_zero, mean_delta) / (znorm * mnorm))
    else:
        cos_zero_vs_mean = 1.0

    # Overall constancy assessment
    max_std = float(std_delta.max())
    constancy_verdict = (
        "CONSTANT" if max_std < 1e-4 else
        "NEARLY_CONSTANT" if max_std < 1e-3 else
        "HISTORY_CONDITIONED" if max_std < 1e-2 else
        "STRONGLY_VARIABLE"
    )

    results = {
        "f0_constant_vector_c": [float(x) for x in mean_delta],
        "f0_norm": float(np.linalg.norm(mean_delta)),
        "f0_zero_input_delta": [float(x) for x in delta_zero],
        "f0_zero_input_norm": float(np.linalg.norm(delta_zero)),
        "n_samples_used": int(all_combined.shape[0]),
        "per_dim": per_dim,
        "global_std_vector": [float(x) for x in std_delta],
        "max_std_across_dims": max_std,
        "cosine_zero_vs_mean": cos_zero_vs_mean,
        "constancy_verdict": constancy_verdict,
        "real_history_deltas_n": len(real_hist_deltas),
    }

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"F0 Vector Audit Results")
    print(f"{'='*60}")
    print(f"c = [{', '.join(f'{x:+.6f}' for x in mean_delta)}]")
    print(f"|c| = {results['f0_norm']:.6f}")
    print(f"Zero-input delta = [{', '.join(f'{x:+.6f}' for x in delta_zero)}]")
    print(f"cos(delta_zero, mean_delta) = {cos_zero_vs_mean:.6f}")
    print(f"\nPer-dim constancy:")
    for d in range(7):
        info = per_dim[f"dim_{d}_{dim_names[d]}"]
        print(f"  {dim_names[d]}: mean={info['mean']:+.6f} std={info['std']:.6f} "
              f"range={info['range']:.6f} [{info['constancy']}]")
    print(f"\nConstancy verdict: {constancy_verdict}")
    print(f"N samples: {all_combined.shape[0]}")

    out_path = output_dir / "f0_constant_vector.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

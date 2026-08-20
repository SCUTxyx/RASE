#!/usr/bin/env python3
"""Route C: train standalone residual recovery plugin.

Losses:
  - Huber(delta_plugin, clip(aT - aS))  [main]
  - action smoothness  [aux]
  - magnitude regularization  [aux]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.recovery.residual_plugin import ResidualRecoveryPlugin, make_recovery_plugin, save_plugin, load_plugin
from rase.recovery.schema import RecoveryDatasetSchemaError, validate_recovery_sample, is_legacy_zero_feature_sample


# ── dataset ─────────────────────────────────────────────────────────────

class RecoveryPluginDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        history_window: int = 8,
        delta_clip: float = 0.5,
        mode: str = "train",
        split_file: Path | None = None,
        obs_feature_dim: int = 144,
        feature_level: str = "F2",
        allow_legacy: bool = False,
    ):
        self.samples: list[dict] = []
        self.history_window = history_window
        self.delta_clip = delta_clip
        self.obs_feature_dim = obs_feature_dim
        self.feature_level = feature_level
        self.split_mode = mode
        self.allow_legacy = allow_legacy
        self._history_dim = 8 + 7 + 1 + 7  # proprio(8) + action(7) + progress(1) + action(7)

        r0_dir = data_dir / "R0"
        if not r0_dir.is_dir():
            raise FileNotFoundError(f"R0 directory not found: {r0_dir}")

        episodes = sorted(r0_dir.glob("*.json"))

        split_info = None
        if split_file and split_file.is_file():
            split_info = json.loads(split_file.read_text(encoding="utf-8"))
            valid_tasks: set[str] = set()
            for suite_tasks in split_info.get("splits", {}).values():
                valid_tasks.update(suite_tasks.get(mode, []))
        else:
            valid_tasks = set()

        for ep_path in episodes:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
            # Filter by mode if split info is in the sample
            sample_split = ep.get("split", "")
            if mode in ("train", "dev"):
                if sample_split and sample_split != mode:
                    continue
            # Protocol-based task filtering (only for train/dev/test modes)
            if self.split_mode in ("train", "dev", "test") and valid_tasks:
                task = self._extract_task_id(ep_path)
                if task not in valid_tasks:
                    continue

            recovery_steps = ep.get("teacher_recovery", [])

            # Simple format: single delta sample (from R3 conversion)
            if not recovery_steps and "delta_target" in ep:
                self.samples.append({
                    "delta_target": ep.get("delta_target", np.zeros(7)),
                    "student_action": ep.get("student_action", np.zeros(7)),
                    "obs_features": ep.get("obs_features", None),
                    "step_index": 0,
                    "episode_id": ep_path.stem,
                    "split_label": ep.get("split", ""),
                })
                continue

            # Full episode format: multiple recovery steps
            for i, step in enumerate(recovery_steps):
                delta_t = np.asarray(step.get("delta_target", np.zeros(7)), dtype=np.float32).flatten()[:7]
                student_action = np.asarray(step.get("action", np.zeros(7)), dtype=np.float32).flatten()[:7]
                obs_feat = step.get("obs_features", None)
                history_before = step.get("history_before", None)
                history_mask = step.get("history_mask", None)
                self.samples.append({
                    "delta_target": delta_t.tolist(),
                    "student_action": student_action.tolist(),
                    "obs_features": obs_feat,
                    "history_before": history_before,
                    "history_mask": history_mask,
                    "step_index": i,
                    "episode_id": ep_path.stem,
                    "split_label": ep.get("split", ""),
                })

    def _extract_task_id(self, path: Path) -> str:
        parts = path.stem.split("_")
        # Filename format: {suite}_{suite_parts}_{task_num}_{mode}_s{seed}.json
        # e.g. libero_spatial_libero_spatial_000001_N0_s744597088
        # Protocol key: libero_spatial_000001 → parts[0]_parts[1]_parts[4]
        if len(parts) >= 5:
            return parts[0] + "_" + parts[1] + "_" + parts[4]
        return path.stem

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Read obs_features from data - MUST be present and non-zero
        obs_feat_data = s.get("obs_features", None)
        if obs_feat_data is not None and np.asarray(obs_feat_data).size > 0:
            obs_feat = np.asarray(obs_feat_data, dtype=np.float32).flatten()
            if obs_feat.size != self.obs_feature_dim:
                padded = np.zeros(self.obs_feature_dim, dtype=np.float32)
                n = min(obs_feat.size, self.obs_feature_dim)
                padded[:n] = obs_feat[:n]
                obs_feat = padded
        else:
            if not self.allow_legacy:
                raise RecoveryDatasetSchemaError(
                    f"obs_features missing or empty in sample {idx} "
                    f"(episode {s.get('episode_id', '?')}, step {s.get('step_index', '?')}). "
                    "Legacy zero-feature samples cannot be used for training. "
                    "Re-collect data with the updated collector, or pass "
                    "--allow-legacy-zero-features for diagnostic-only runs."
                )
            obs_feat = np.zeros(self.obs_feature_dim, dtype=np.float32)

        # Read history_before from data if available, otherwise zeros (backward compat)
        hist_data = s.get("history_before", None)
        if hist_data is not None and np.asarray(hist_data).size > 0:
            hist_arr = np.asarray(hist_data, dtype=np.float32)
            if hist_arr.ndim == 2:
                # (window, dim) format from updated collector
                pass
            elif hist_arr.ndim == 1:
                hist_arr = hist_arr.reshape(self.history_window, -1)
            if hist_arr.shape[0] != self.history_window:
                resized = np.zeros((self.history_window, self._history_dim), dtype=np.float32)
                n = min(hist_arr.shape[0], self.history_window)
                d = min(hist_arr.shape[1], self._history_dim)
                resized[:n, :d] = hist_arr[:n, :d]
                hist_arr = resized
        else:
            hist_arr = np.zeros((self.history_window, self._history_dim), dtype=np.float32)

        return {
            "history": torch.tensor(hist_arr, dtype=torch.float32),
            "obs_features": torch.tensor(obs_feat, dtype=torch.float32),
            "student_action": torch.tensor(np.asarray(s["student_action"], dtype=np.float32).flatten()[:7], dtype=torch.float32),
            "delta_target": torch.tensor(np.asarray(s["delta_target"], dtype=np.float32).flatten()[:7], dtype=torch.float32),
            "step_index": s["step_index"],
        }


# ── train ──────────────────────────────────────────────────────────────

def train_one_segment(
    plugin: ResidualRecoveryPlugin,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    segment: int,
    writer: SummaryWriter | None = None,
    lambda_smooth: float = 0.01,
    lambda_mag: float = 0.001,
    lambda_early: float = 2.0,
) -> dict:
    plugin.train()
    total_loss = 0.0
    total_huber = 0.0
    total_smooth = 0.0
    total_mag = 0.0
    n = 0

    for batch in dataloader:
        history = batch["history"].to(device)
        obs_feat = batch["obs_features"].to(device)
        student_action = batch["student_action"].to(device)
        delta_target = batch["delta_target"].to(device)
        step_idx = batch["step_index"].to(device)

        delta_pred = plugin(history, obs_feat, student_action)

        huber = F.huber_loss(delta_pred, delta_target, delta=0.5)

        weight = torch.where(step_idx < 8, lambda_early, 1.0)
        huber_weighted = (huber * weight).mean() if huber.dim() > 0 else huber

        smooth = 0.0
        if delta_pred.size(0) > 1:
            smooth = F.mse_loss(delta_pred[1:], delta_pred[:-1])

        mag = delta_pred.abs().mean()

        loss = huber_weighted + lambda_smooth * smooth + lambda_mag * mag

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plugin.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_huber += huber_weighted.item()
        total_smooth += smooth if isinstance(smooth, float) else smooth.item()
        total_mag += mag.item()
        n += 1

    n = max(n, 1)
    metrics = {"loss": total_loss / n, "huber": total_huber / n,
               "smooth": total_smooth / n, "mag": total_mag / n}
    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, segment)
    return metrics


def evaluate_on_segment(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                         device: torch.device) -> dict:
    plugin.eval()
    total_huber = 0.0
    total_zero = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)

            delta_pred = plugin(history, obs_feat, student_action)
            huber = F.huber_loss(delta_pred, delta_target, delta=0.5)
            zero_err = F.mse_loss(delta_target, torch.zeros_like(delta_target))

            total_huber += huber.item()
            total_zero += zero_err.item()
            n += 1

    n = max(n, 1)
    return {"huber": total_huber / n, "zero_baseline": total_zero / n}


def compute_residual_change(plugin_before: ResidualRecoveryPlugin,
                             plugin_after: ResidualRecoveryPlugin,
                             dataloader: DataLoader, device: torch.device) -> float:
    with torch.no_grad():
        diffs = []
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            d1 = plugin_before(history, obs_feat, student_action)
            d2 = plugin_after(history, obs_feat, student_action)
            diffs.append((d2 - d1).abs().mean().item())
            if len(diffs) > 16:
                break
    return float(np.mean(diffs))


# ── overfit diagnostics (D0-B) ───────────────────────────────────────

def compute_per_dim_stats(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                           device: torch.device) -> dict:
    plugin.eval()
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)
            delta_pred = plugin(history, obs_feat, student_action)
            all_pred.append(delta_pred.cpu().numpy())
            all_target.append(delta_target.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    per_dim = {}
    for d in range(min(pred.shape[1], target.shape[1])):
        per_dim[f"dim_{d}"] = {
            "target_mean": float(np.mean(target[:, d])),
            "target_std": float(np.std(target[:, d])),
            "pred_mean": float(np.mean(pred[:, d])),
            "pred_std": float(np.std(pred[:, d])),
            "mean_abs_error": float(np.mean(np.abs(pred[:, d] - target[:, d]))),
        }
    return {"per_dim": per_dim, "n_samples": int(pred.shape[0])}


def compute_cosine_sim(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                        device: torch.device, eps: float = 1e-4) -> dict:
    plugin.eval()
    cosines: list[float] = []
    n_zero = 0
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)
            delta_pred = plugin(history, obs_feat, student_action)

            pred_np = delta_pred.cpu().numpy()
            target_np = delta_target.cpu().numpy()
            for i in range(pred_np.shape[0]):
                t = target_np[i]
                tnorm = np.linalg.norm(t)
                if tnorm < eps:
                    n_zero += 1
                    continue
                p = pred_np[i]
                pnorm = np.linalg.norm(p)
                if pnorm < eps:
                    cosines.append(0.0)
                else:
                    cos = float(np.dot(p, t) / (pnorm * tnorm))
                    cosines.append(np.clip(cos, -1.0, 1.0))
    if not cosines:
        return {"mean_cosine": None, "median_cosine": None, "n_nonzero": 0, "n_zero_target": n_zero,
                "note": "all targets near zero"}
    return {
        "mean_cosine": float(np.mean(cosines)),
        "median_cosine": float(np.median(cosines)),
        "n_nonzero": len(cosines),
        "n_zero_target": n_zero,
    }


def compute_norm_ratio(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                        device: torch.device) -> dict:
    plugin.eval()
    pred_norms = []
    target_norms = []
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)
            delta_pred = plugin(history, obs_feat, student_action)

            pred_n = torch.norm(delta_pred, dim=-1).cpu().numpy()
            target_n = torch.norm(delta_target, dim=-1).cpu().numpy()
            pred_norms.extend(pred_n.tolist())
            target_norms.extend(target_n.tolist())

    mean_pred = float(np.mean(pred_norms))
    mean_target = float(np.mean(target_norms))
    ratio = mean_pred / max(mean_target, 1e-8)
    return {
        "mean_pred_norm": mean_pred,
        "mean_target_norm": mean_target,
        "norm_ratio": ratio,
        "in_range": 0.5 <= ratio <= 1.5,
    }


def compute_executed_action_change(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                                    device: torch.device, action_clip: float = 1.0) -> dict:
    """Simulate the full execution pipeline: pred → delta_clip → action_space_clip."""
    plugin.eval()
    ratios = []
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)
            delta_pred = plugin(history, obs_feat, student_action)

            # Apply delta_clip (same as plugin's forward)
            delta_pred_clipped = torch.clamp(delta_pred,
                                              -plugin.delta_clip, plugin.delta_clip)
            # Apply action-space clip: student + delta → clamped to [-1, 1]
            final_action = torch.clamp(student_action + delta_pred_clipped,
                                       -float(action_clip), float(action_clip))
            action_change = (final_action - student_action).abs().sum(dim=-1)

            for i in range(len(action_change)):
                target_norm = float(torch.norm(delta_target[i]))
                if target_norm > 1e-6:
                    ratios.append(float(action_change[i]) / target_norm)

    if not ratios:
        return {"executed_action_change_ratio": 0.0, "n": 0, "note": "all zero targets"}
    return {
        "executed_action_change_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "n": len(ratios),
    }


def verify_save_reload_parity(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                               device: torch.device, tmp_path: str) -> dict:
    save_plugin(plugin, tmp_path)
    reloaded = load_plugin(tmp_path).to(device)
    reloaded.eval()
    plugin.eval()
    max_abs_diff = 0.0
    n_compared = 0
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            d1 = plugin(history, obs_feat, student_action)
            d2 = reloaded(history, obs_feat, student_action)
            diff = (d1 - d2).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            n_compared += d1.size(0)
    return {"save_reload_max_abs_diff": max_abs_diff, "parity_ok": max_abs_diff == 0.0,
            "n_compared": n_compared}


def compute_zero_baseline_mse(dataloader: DataLoader) -> float:
    """MSE of predicting zero delta (baseline)."""
    total_mse = 0.0
    n = 0
    for batch in dataloader:
        delta_target = batch["delta_target"]
        total_mse += F.mse_loss(torch.zeros_like(delta_target), delta_target).item()
        n += 1
    return total_mse / max(n, 1)


def compute_mse_reduction(plugin: ResidualRecoveryPlugin, dataloader: DataLoader,
                           device: torch.device, zero_mse: float) -> float:
    """Percentage reduction in MSE relative to zero baseline."""
    plugin.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            obs_feat = batch["obs_features"].to(device)
            student_action = batch["student_action"].to(device)
            delta_target = batch["delta_target"].to(device)
            delta_pred = plugin(history, obs_feat, student_action)
            total_mse += F.mse_loss(delta_pred, delta_target).item()
            n += 1
    model_mse = total_mse / max(n, 1)
    return (1.0 - model_mse / max(zero_mse, 1e-8)) * 100.0


def run_overfit_diagnostics(plugin: ResidualRecoveryPlugin,
                             dataloader: DataLoader, device: torch.device,
                             output_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("D0-B Overfit Diagnostics")
    print("=" * 60)

    zero_mse = compute_zero_baseline_mse(dataloader)
    print(f"Zero-baseline MSE: {zero_mse:.6f}")

    mse_reduction = compute_mse_reduction(plugin, dataloader, device, zero_mse)
    print(f"MSE reduction: {mse_reduction:.1f}% (target >= 90%)")

    per_dim = compute_per_dim_stats(plugin, dataloader, device)
    print("Per-dim stats:")
    for k, v in per_dim["per_dim"].items():
        print(f"  {k}: target[{v['target_mean']:+.4f}±{v['target_std']:.4f}] "
              f"pred[{v['pred_mean']:+.4f}±{v['pred_std']:.4f}] "
              f"MAE={v['mean_abs_error']:.4f}")

    cosine = compute_cosine_sim(plugin, dataloader, device)
    print(f"Cosine similarity (nonzero targets only):")
    if cosine.get("mean_cosine") is not None:
        print(f"  mean={cosine['mean_cosine']:.4f} median={cosine['median_cosine']:.4f} "
              f"n_nonzero={cosine['n_nonzero']} n_zero_target={cosine['n_zero_target']}")
    else:
        print(f"  {cosine.get('note', 'N/A')}")

    norm = compute_norm_ratio(plugin, dataloader, device)
    print(f"Norm ratio: {norm['norm_ratio']:.4f} "
          f"(pred={norm['mean_pred_norm']:.4f} target={norm['mean_target_norm']:.4f}) "
          f"in_range={norm['in_range']}")

    exec_change = compute_executed_action_change(plugin, dataloader, device)
    print(f"Executed action change ratio: {exec_change['executed_action_change_ratio']:.4f} "
          f"median={exec_change.get('median_ratio', 'N/A')} "
          f"n={exec_change['n']}")

    tmp_path = str(output_dir / "_parity_check.pt")
    parity = verify_save_reload_parity(plugin, dataloader, device, tmp_path)
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()
    print(f"Save/reload parity: max_abs_diff={parity['save_reload_max_abs_diff']:.8f} "
          f"ok={parity['parity_ok']}")

    # ── Gate ───────────────────────────────────────────────────────
    gate_checks = {
        "mse_reduction_ok": mse_reduction >= 90.0,
        "cosine_mean_ok": (cosine.get("mean_cosine") or 0.0) >= 0.8,
        "cosine_median_ok": (cosine.get("median_cosine") or 0.0) >= 0.8,
        "norm_ratio_ok": norm["in_range"],
        "executed_action_change_ok": exec_change["executed_action_change_ratio"] >= 0.5,
        "save_reload_parity_ok": parity["parity_ok"],
    }
    gate_pass = all(gate_checks.values())

    result = {
        "zero_baseline_mse": zero_mse,
        "mse_reduction_pct": mse_reduction,
        "per_dim_stats": per_dim,
        "cosine_sim": cosine,
        "norm_ratio": norm,
        "executed_action_change": exec_change,
        "save_reload_parity": parity,
        "gate_checks": gate_checks,
        "gate_pass": gate_pass,
    }

    print("\n--- Gate ---")
    for check_name, ok in gate_checks.items():
        print(f"  {check_name}: {'PASS' if ok else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if gate_pass else 'FAIL'}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory with R0/N0/F0 subdirs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--n-segments", type=int, default=16)
    parser.add_argument("--steps-per-segment", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-level", type=str, default="F2",
                        choices=["F0", "F1", "F2"],
                        help="F0=zeros only, F1=proprio+action+stats, F2=SmolVLA latent+all")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "overfit"],
                        help="training mode: standard or overfit diagnostics")
    parser.add_argument("--allow-legacy-zero-features", action="store_true",
                        help="allow legacy zero-feature samples (diagnostic only, "
                             "do NOT use for production checkpoints)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="override plugin hidden_dim (default: from protocol)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]

    # Determine feature dimension based on level
    from rase.collect.smolvla_feature_extractor import F2_FEATURE_DIM, SMOLVLA_LATENT_DIM
    if args.feature_level == "F2":
        obs_feature_dim = F2_FEATURE_DIM  # 144
    elif args.feature_level == "F1":
        obs_feature_dim = SMOLVLA_LATENT_DIM  # same size, but first 128 are zeros
    else:
        obs_feature_dim = SMOLVLA_LATENT_DIM  # 128 (zeros only)

    ds_mode = "all" if args.mode == "overfit" else "train"
    ds_val_mode = "all" if args.mode == "overfit" else "dev"
    train_ds = RecoveryPluginDataset(args.data_dir, mode=ds_mode,
                                      split_file=args.protocol,
                                      obs_feature_dim=obs_feature_dim,
                                      feature_level=args.feature_level,
                                      allow_legacy=args.allow_legacy_zero_features)
    val_ds = RecoveryPluginDataset(args.data_dir, mode=ds_val_mode,
                                    split_file=args.protocol,
                                    obs_feature_dim=obs_feature_dim,
                                    feature_level=args.feature_level,
                                    allow_legacy=args.allow_legacy_zero_features)
    if len(train_ds) == 0:
        print("ERROR: No training samples found in R0 directory.")
        print(f"  data_dir = {args.data_dir}")
        print("  Please run the data collection script first:")
        print(f"    conda run -p /root/autodl-tmp/envs/smolvla python scripts/collect_route_c_demos.py \\")
        print(f"      --suite libero_spatial --output-dir {args.data_dir} --protocol {args.protocol}")
        print(f"      --n-episodes-per-task 4 --mode all")
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else train_loader

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
          f"feature_level={args.feature_level}, obs_feature_dim={obs_feature_dim}")

    plugin_kwargs = dict(
        proprio_dim=protocol["action_schema"]["proprio_dim"],
        action_dim=protocol["action_schema"]["action_dim"],
        history_window=plugin_conf["plugin_history_window"],
        delta_clip=plugin_conf["delta_clip_per_dim"],
        obs_feature_dim=obs_feature_dim,
    )
    if args.hidden_dim is not None:
        plugin_kwargs["hidden_dim"] = args.hidden_dim
        print(f"  hidden_dim override: {args.hidden_dim}")

    plugin = make_recovery_plugin(**plugin_kwargs).to(device)

    before_params = {k: v.clone() for k, v in plugin.state_dict().items()}
    plugin_before = make_recovery_plugin(
        history_window=plugin_conf["plugin_history_window"],
        delta_clip=plugin_conf["delta_clip_per_dim"],
        obs_feature_dim=obs_feature_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(plugin.parameters(), lr=args.lr)

    segment_results = []
    for seg in range(1, args.n_segments + 1):
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard" / f"seg_{seg:02d}"))
        metrics = train_one_segment(plugin, train_loader, optimizer, device, seg, writer=writer)
        val_metrics = evaluate_on_segment(plugin, val_loader, device)

        has_nan = any(math.isnan(v) for v in metrics.values())
        seg_result = {"segment": seg, "train": metrics, "val": val_metrics, "nan": has_nan}
        segment_results.append(seg_result)
        print(f"  seg {seg:2d}: huber={metrics['huber']:.4f} val_huber={val_metrics['huber']:.4f} nan={has_nan}")
        writer.close()

    residual_change = compute_residual_change(plugin_before, plugin, val_loader, device)
    plugin_before.to(device)
    plugin_before.load_state_dict({k: v.to(device) for k, v in before_params.items() if k in plugin_before.state_dict()})
    residual_change = compute_residual_change(plugin_before, plugin, val_loader, device)

    final_val = evaluate_on_segment(plugin, val_loader, device)

    ckpt_path = output_dir / "plugin_best.pt"
    save_plugin(plugin, str(ckpt_path))

    if args.mode == "overfit":
        # Run comprehensive diagnostics
        diag_loader = DataLoader(train_ds, batch_size=args.batch_size)  # diagnostic on train set
        diag_results = run_overfit_diagnostics(plugin, diag_loader, device, output_dir)
        diag_path = output_dir / "overfit_diagnostics.json"
        diag_path.write_text(json.dumps(diag_results, indent=2) + "\n", encoding="utf-8")
        per_dim_path = output_dir / "per_dim_stats.json"
        per_dim_path.write_text(json.dumps(diag_results["per_dim_stats"], indent=2) + "\n",
                                 encoding="utf-8")
        gate_pass = diag_results["gate_pass"]

        gate = {
            "segments": segment_results,
            "final_val_huber": final_val["huber"],
            "residual_change": residual_change,
            "overfit_pass": gate_pass,
            "diagnostics": {k: v for k, v in diag_results.items() if k != "per_dim_stats"},
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        }
    else:
        # Legacy gate
        overfit_pass = (final_val["huber"] < 0.1 and residual_change > 1e-6 and
                        not any(r.get("nan", False) for r in segment_results))
        gate_pass = overfit_pass
        gate = {
            "segments": segment_results,
            "final_val_huber": final_val["huber"],
            "residual_change": residual_change,
            "overfit_pass": overfit_pass,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        }

    gate["gate_pass"] = gate_pass
    gate_path = output_dir / "plugin_overfit_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Training done. Huber={final_val['huber']:.4f}, change={residual_change:.6f}, "
          f"overfit={gate_pass} mode={args.mode}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

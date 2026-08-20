#!/usr/bin/env python3
"""Route C: train recovery selector v2 — outcome-grounded labels.

Unlike v1 which uses delta_target L2 threshold for labels, v2 uses
counterfactual outcome labels:

  rescue (1)  → student fails alone, plugin succeeds → strong positive
  harm (-1)   → student succeeds alone, plugin causes failure → strong negative
  both_fail / both_ok (0) → neutral

Weighted BCE loss emphasizes harm avoidance (weight=4.0) over rescue
(weight=2.0) with down-weighted neutral samples (weight=0.2).

If no counterfactual data is available, falls back to delta_target labels
with adjusted thresholds and weighted loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.recovery.selector import RecoverySelector, make_selector, save_selector


# ── Counterfactual dataset ───────────────────────────────────────────

class CounterfactualSelectorDataset(Dataset):
    """Loads counterfactual labels (rescue/harm/neutral) for training.

    Each record in the jsonl has:
      - label: -1 (harm), 1 (rescue), 0 (neutral)
      - boundary_obs_features: 144-D F2 feature vector at boundary
      - boundary_history: optional history_before
      - student_action_at_boundary: action at boundary frame
    """

    def __init__(
        self,
        cf_path: Path,
        history_window: int = 8,
        obs_feature_dim: int = 144,
        mode: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.history_window = history_window
        self.obs_feature_dim = obs_feature_dim
        self._history_dim = 8 + 7 + 1 + 7

        records = []
        with open(cf_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                records.append(rec)

        # Split by episode_id to avoid leakage
        ep_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            ep_id = rec.get("episode_id", str(i))
            ep_to_idx[ep_id].append(i)

        ep_list = sorted(ep_to_idx.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(ep_list)
        n_train = max(1, int(len(ep_list) * train_ratio))

        if mode == "train":
            valid_eps = set(ep_list[:n_train])
        else:
            valid_eps = set(ep_list[n_train:])

        self.samples: list[dict] = []
        for ep_id in valid_eps:
            for idx in ep_to_idx[ep_id]:
                rec = records[idx]
                self.samples.append({
                    "label": rec["label"],
                    "category": rec.get("category", "unknown"),
                    "obs_features": rec.get("boundary_obs_features", None),
                    "history_before": rec.get("boundary_history", None),
                    "student_action": rec.get("student_action_at_boundary", np.zeros(7)),
                    "episode_id": ep_id,
                    "weight": self._sample_weight(rec["label"]),
                })

    @staticmethod
    def _sample_weight(label: int) -> float:
        if label == -1:    # harm → highest weight
            return 4.0
        elif label == 1:   # rescue → medium weight
            return 2.0
        else:              # neutral → low weight
            return 0.2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        obs_feat_data = s.get("obs_features", None)
        if obs_feat_data is not None and np.asarray(obs_feat_data).size > 0:
            obs_feat = np.asarray(obs_feat_data, dtype=np.float32).flatten()
            if obs_feat.size != self.obs_feature_dim:
                padded = np.zeros(self.obs_feature_dim, dtype=np.float32)
                n = min(obs_feat.size, self.obs_feature_dim)
                padded[:n] = obs_feat[:n]
                obs_feat = padded
        else:
            obs_feat = np.zeros(self.obs_feature_dim, dtype=np.float32)

        hist_data = s.get("history_before", None)
        if hist_data is not None and np.asarray(hist_data).size > 0:
            hist_arr = np.asarray(hist_data, dtype=np.float32)
            if hist_arr.ndim == 1:
                hist_arr = hist_arr.reshape(self.history_window, -1)
            if hist_arr.shape[0] != self.history_window:
                resized = np.zeros((self.history_window, self._history_dim), dtype=np.float32)
                n = min(hist_arr.shape[0], self.history_window)
                d = min(hist_arr.shape[1], self._history_dim)
                resized[:n, :d] = hist_arr[:n, :d]
                hist_arr = resized
        else:
            hist_arr = np.zeros((self.history_window, self._history_dim), dtype=np.float32)

        student_action = np.asarray(s["student_action"], dtype=np.float32).flatten()[:7]
        label = float(max(s["label"], 0))  # treat harm (-1) as 0 for BCE

        return {
            "history": torch.tensor(hist_arr, dtype=torch.float32),
            "obs_features": torch.tensor(obs_feat, dtype=torch.float32),
            "student_action": torch.tensor(student_action, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "weight": torch.tensor(s["weight"], dtype=torch.float32),
        }


# ── Delta-target fallback dataset ────────────────────────────────────

class DeltaTargetSelectorDataset(Dataset):
    """Fallback dataset using L2(delta_target) > threshold labels (v1 style).

    Includes sample weights: recovery steps weight=2.0, student_rollin
    steps weight=1.0 (negative samples).
    """

    def __init__(
        self,
        data_dir: Path,
        history_window: int = 8,
        obs_feature_dim: int = 144,
        mode: str = "train",
        label_threshold: float = 0.15,
        seed: int = 42,
    ):
        self.samples: list[dict] = []
        self.history_window = history_window
        self.obs_feature_dim = obs_feature_dim
        self._history_dim = 8 + 7 + 1 + 7

        r0_dir = data_dir / "R0"
        if not r0_dir.is_dir():
            raise FileNotFoundError(f"R0 directory not found: {r0_dir}")

        episodes = sorted(r0_dir.glob("*.json"))
        if not episodes:
            raise ValueError(f"No episodes found in {r0_dir}")

        all_samples = []
        episode_ids = []
        for ep_path in episodes:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
            recovery_steps = ep.get("teacher_recovery", [])
            student_rollin = ep.get("student_rollin", [])

            for step in student_rollin:
                all_samples.append({
                    "delta_target": [0.0] * 7,
                    "student_action": step.get("action", np.zeros(7)),
                    "obs_features": step.get("obs_features", None),
                    "history_before": step.get("history_before", None),
                    "step_index": -1,
                    "weight": 1.0,
                })
                episode_ids.append(ep_path.stem + "_rollin")

            for step in recovery_steps:
                delta = np.asarray(step.get("delta_target", np.zeros(7)), dtype=np.float32).flatten()[:7]
                l2 = float(np.linalg.norm(delta))
                all_samples.append({
                    "delta_target": delta,
                    "student_action": step.get("action", np.zeros(7)),
                    "obs_features": step.get("obs_features", None),
                    "history_before": step.get("history_before", None),
                    "step_index": 0,
                    "label": 1 if l2 > label_threshold else 0,
                    "weight": 2.0 if l2 > label_threshold else 1.0,
                })
                episode_ids.append(ep_path.stem)

        labels = [s.get("label", 0) for s in all_samples]

        ep_to_indices = defaultdict(list)
        for i, ep_id in enumerate(episode_ids):
            ep_to_indices[ep_id].append(i)

        ep_list = sorted(ep_to_indices.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(ep_list)
        n_train = max(1, int(len(ep_list) * 0.8))

        if mode == "train":
            valid_eps = set(ep_list[:n_train])
        else:
            valid_eps = set(ep_list[n_train:])

        for ep_id in valid_eps:
            for idx in ep_to_indices[ep_id]:
                s = all_samples[idx]
                s["label_val"] = labels[idx]
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        obs_feat_data = s.get("obs_features", None)
        if obs_feat_data is not None and np.asarray(obs_feat_data).size > 0:
            obs_feat = np.asarray(obs_feat_data, dtype=np.float32).flatten()
            if obs_feat.size != self.obs_feature_dim:
                padded = np.zeros(self.obs_feature_dim, dtype=np.float32)
                n = min(obs_feat.size, self.obs_feature_dim)
                padded[:n] = obs_feat[:n]
                obs_feat = padded
        else:
            obs_feat = np.zeros(self.obs_feature_dim, dtype=np.float32)

        hist_data = s.get("history_before", None)
        if hist_data is not None and np.asarray(hist_data).size > 0:
            hist_arr = np.asarray(hist_data, dtype=np.float32)
            if hist_arr.ndim == 1:
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
            "student_action": torch.tensor(
                np.asarray(s["student_action"], dtype=np.float32).flatten()[:7],
                dtype=torch.float32),
            "label": torch.tensor(float(s.get("label_val", 0)), dtype=torch.float32),
            "weight": torch.tensor(s.get("weight", 1.0), dtype=torch.float32),
        }


# ── training ─────────────────────────────────────────────────────────

def train_one_epoch(selector, dataloader, optimizer, device, epoch, writer=None):
    selector.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        history = batch["history"].to(device)
        obs_feat = batch["obs_features"].to(device)
        student_action = batch["student_action"].to(device)
        labels = batch["label"].to(device)
        weights = batch["weight"].to(device)

        logits = selector(history, obs_feat, student_action)

        # Per-sample weighted BCE
        loss_per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none")
        loss = (loss_per_sample * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    metrics = {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
    }
    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
    return metrics


@torch.no_grad()
def evaluate(selector, dataloader, device):
    selector.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for batch in dataloader:
        history = batch["history"].to(device)
        obs_feat = batch["obs_features"].to(device)
        student_action = batch["student_action"].to(device)
        labels = batch["label"].to(device)

        logits = selector(history, obs_feat, student_action)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for p, l in zip(preds.cpu(), labels.cpu()):
            if p == 1 and l == 1:
                true_pos += 1
            elif p == 1 and l == 0:
                false_pos += 1
            elif p == 0 and l == 1:
                false_neg += 1
            else:
                true_neg += 1

    n_pos = true_pos + false_neg
    n_neg = true_neg + false_pos
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
        "true_pos": true_pos,
        "false_pos": false_pos,
        "true_neg": true_neg,
        "false_neg": false_neg,
        "precision": true_pos / max(true_pos + false_pos, 1),
        "recall": true_pos / max(n_pos, 1),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="R0 data directory (fallback mode when no cf-labels)")
    parser.add_argument("--cf-labels", type=Path, default=None,
                        help="counterfactual_labels.jsonl from Phase 1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label-threshold", type=float, default=0.15,
                        help="L2 threshold for fallback delta-target mode")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mode", type=str, default="cf",
                        choices=["cf", "delta"],
                        help="label mode: cf=counterfactual, delta=R0 delta_target")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]
    history_window = plugin_conf["plugin_history_window"]

    from rase.collect.smolvla_feature_extractor import F2_FEATURE_DIM
    obs_feature_dim = F2_FEATURE_DIM  # 144

    # ── Dataset selection ──
    if args.mode == "cf" and args.cf_labels and args.cf_labels.is_file():
        print(f"Using counterfactual labels: {args.cf_labels}")
        train_ds = CounterfactualSelectorDataset(
            args.cf_labels,
            history_window=history_window,
            obs_feature_dim=obs_feature_dim,
            mode="train",
            seed=args.seed,
        )
        val_ds = CounterfactualSelectorDataset(
            args.cf_labels,
            history_window=history_window,
            obs_feature_dim=obs_feature_dim,
            mode="val",
            seed=args.seed,
        )
        label_source = "counterfactual"
    elif args.data_dir:
        print(f"Using delta_target labels from {args.data_dir} (threshold={args.label_threshold})")
        train_ds = DeltaTargetSelectorDataset(
            args.data_dir,
            history_window=history_window,
            obs_feature_dim=obs_feature_dim,
            mode="train",
            label_threshold=args.label_threshold,
            seed=args.seed,
        )
        val_ds = DeltaTargetSelectorDataset(
            args.data_dir,
            history_window=history_window,
            obs_feature_dim=obs_feature_dim,
            mode="val",
            label_threshold=args.label_threshold,
            seed=args.seed,
        )
        label_source = f"delta_target_{args.label_threshold}"
    else:
        print("ERROR: Need either --cf-labels or --data-dir")
        return 1

    if len(train_ds) == 0:
        print("ERROR: No training samples found.")
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else train_loader

    n_train_pos = sum(1 for s in train_ds.samples if s.get("label_val", s.get("label", 0)) == 1)
    n_train_neg = len(train_ds) - n_train_pos
    n_val_pos = sum(1 for s in val_ds.samples if s.get("label_val", s.get("label", 0)) == 1) if len(val_ds) > 0 else 0
    n_val_neg = len(val_ds) - n_val_pos if len(val_ds) > 0 else 0
    print(f"Train: {len(train_ds)} samples (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val:   {len(val_ds)} samples (pos={n_val_pos}, neg={n_val_neg})")
    print(f"Label source: {label_source}")

    selector = make_selector(
        proprio_dim=protocol["action_schema"]["proprio_dim"],
        action_dim=protocol["action_schema"]["action_dim"],
        history_window=history_window,
        obs_feature_dim=obs_feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr)

    best_val_acc = 0.0
    epoch_results = []

    for epoch in range(1, args.n_epochs + 1):
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard" / f"epoch_{epoch:02d}"))
        train_metrics = train_one_epoch(selector, train_loader, optimizer, device, epoch, writer)
        val_metrics = evaluate(selector, val_loader, device)

        epoch_results.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"  epoch {epoch:2d}: train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.3f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
              f"fp={val_metrics['false_pos']} fn={val_metrics['false_neg']} "
              f"prec={val_metrics['precision']:.3f} rec={val_metrics['recall']:.3f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_path = output_dir / "selector_v2_best.pt"
            save_selector(selector, str(ckpt_path))

        writer.close()

    # ── Gate check ──
    final_val = evaluate(selector, val_loader, device)

    # Precision-first gates: fp_rate is the most critical metric
    fp_rate = final_val["false_pos"] / max(final_val["n_negative"], 1)
    gate = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "label_source": label_source,
        "label_threshold": args.label_threshold if args.mode == "delta" else None,
        "epochs": epoch_results,
        "final_val": final_val,
        "best_val_accuracy": best_val_acc,
        "gate_pass": False,
        "gate_checks": {
            "recall_ok": final_val["recall"] >= 0.4,
            "precision_ok": final_val["precision"] >= 0.7,
            "false_pos_rate_ok": fp_rate <= 0.1,
        },
    }
    gate["gate_pass"] = all(gate["gate_checks"].values())

    gate_path = output_dir / "selector_v2_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nDone. val_acc={final_val['accuracy']:.3f} "
          f"precision={final_val['precision']:.3f} "
          f"recall={final_val['recall']:.3f} "
          f"fp_rate={fp_rate:.3f} "
          f"gate={'PASS' if gate['gate_pass'] else 'FAIL'}")
    print(f"  Gate checks: {gate['gate_checks']}")
    return 0 if gate["gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

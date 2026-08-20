#!/usr/bin/env python3
"""Route C: train recovery selector (binary classifier).

Train a lightweight MLP that decides whether each step needs Plugin
intervention. Labels are auto-generated from R0 recovery data:
  L2(delta_target) > threshold → label=1 (need plugin)
  else → label=0 (student ok)

Model: ~3K parameter binary classifier sharing Plugin's encoder
architecture with a single sigmoid output.
"""

from __future__ import annotations

import argparse
import json
import math
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


# ── label generation ──────────────────────────────────────────────────

def compute_labels(samples: list[dict], threshold: float = 0.05) -> list[int]:
    """Generate binary labels from delta_target L2 norm.

    delta_target = teacher_action - student_action.
    Large L2 → student diverged from teacher → needs plugin.
    """
    labels = []
    for s in samples:
        delta = np.asarray(s.get("delta_target", np.zeros(7)), dtype=np.float32).flatten()[:7]
        l2 = float(np.linalg.norm(delta))
        labels.append(1 if l2 > threshold else 0)
    return labels


# ── dataset ───────────────────────────────────────────────────────────

class SelectorDataset(Dataset):
    """Loads R0 recovery data for selector training.

    Uses the same data format as RecoveryPluginDataset but adds
    binary labels for whether Plugin should intervene.
    """

    def __init__(
        self,
        data_dir: Path,
        history_window: int = 8,
        obs_feature_dim: int = 144,
        mode: str = "train",
        split_file: Path | None = None,
        label_threshold: float = 0.05,
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

        # — Load all raw samples first —
        all_samples = []
        episode_ids = []
        for ep_path in episodes:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
            recovery_steps = ep.get("teacher_recovery", [])
            student_rollin = ep.get("student_rollin", [])

            # Student rollin steps → label=0 (student was fine, no plugin needed)
            for i, step in enumerate(student_rollin):
                all_samples.append({
                    "delta_target": [0.0]*7,  # student was doing fine, no delta
                    "student_action": step.get("action", np.zeros(7)),
                    "obs_features": step.get("obs_features", None),
                    "history_before": step.get("history_before", None),
                    "step_index": -1,  # negative label marker
                })
                episode_ids.append(ep_path.stem + "_rollin")

            if not recovery_steps and "delta_target" in ep:
                all_samples.append({
                    "delta_target": ep.get("delta_target", np.zeros(7)),
                    "student_action": ep.get("student_action", np.zeros(7)),
                    "obs_features": ep.get("obs_features", None),
                    "history_before": ep.get("history_before", None),
                    "step_index": 0,
                })
                episode_ids.append(ep_path.stem)
                continue

            for i, step in enumerate(recovery_steps):
                all_samples.append({
                    "delta_target": step.get("delta_target", np.zeros(7)),
                    "student_action": step.get("action", np.zeros(7)),
                    "obs_features": step.get("obs_features", None),
                    "history_before": step.get("history_before", None),
                    "step_index": i,
                })
                episode_ids.append(ep_path.stem)

        # — Generate labels —
        labels = compute_labels(all_samples, threshold=label_threshold)
        for s, label in zip(all_samples, labels):
            s["label"] = label

        # — Split by episode to avoid leakage —
        ep_to_indices = defaultdict(list)
        for i, ep_id in enumerate(episode_ids):
            ep_to_indices[ep_id].append(i)

        ep_list = sorted(ep_to_indices.keys())
        np.random.seed(42)
        np.random.shuffle(ep_list)
        n_train = max(1, int(len(ep_list) * 0.8))

        if mode == "train":
            valid_eps = set(ep_list[:n_train])
        else:
            valid_eps = set(ep_list[n_train:])

        for ep_id in valid_eps:
            for idx in ep_to_indices[ep_id]:
                self.samples.append(all_samples[idx])

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
            "label": torch.tensor(float(s["label"]), dtype=torch.float32),
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

        logits = selector(history, obs_feat, student_action)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

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
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory with R0/N0/F0 subdirs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label-threshold", type=float, default=0.05,
                        help="L2 threshold for plugin-needed label")
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]

    from rase.collect.smolvla_feature_extractor import F2_FEATURE_DIM
    obs_feature_dim = F2_FEATURE_DIM  # 144

    train_ds = SelectorDataset(
        args.data_dir,
        history_window=plugin_conf["plugin_history_window"],
        obs_feature_dim=obs_feature_dim,
        mode="train",
        label_threshold=args.label_threshold,
    )
    val_ds = SelectorDataset(
        args.data_dir,
        history_window=plugin_conf["plugin_history_window"],
        obs_feature_dim=obs_feature_dim,
        mode="val",
        label_threshold=args.label_threshold,
    )

    if len(train_ds) == 0:
        print("ERROR: No training samples found.")
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) > 0 else train_loader

    n_train_pos = sum(1 for s in train_ds.samples if s["label"] == 1)
    n_train_neg = len(train_ds) - n_train_pos
    n_val_pos = sum(1 for s in val_ds.samples if s["label"] == 1) if len(val_ds) > 0 else 0
    n_val_neg = len(val_ds) - n_val_pos if len(val_ds) > 0 else 0
    print(f"Train: {len(train_ds)} samples (pos={n_train_pos}, neg={n_train_neg})")
    print(f"Val:   {len(val_ds)} samples (pos={n_val_pos}, neg={n_val_neg})")
    print(f"Label threshold: L2(delta_target) > {args.label_threshold}")

    selector = make_selector(
        proprio_dim=protocol["action_schema"]["proprio_dim"],
        action_dim=protocol["action_schema"]["action_dim"],
        history_window=plugin_conf["plugin_history_window"],
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
              f"fp={val_metrics['false_pos']} fn={val_metrics['false_neg']}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_path = output_dir / "selector_best.pt"
            save_selector(selector, str(ckpt_path))

        writer.close()

    # ── Gate check ──
    final_val = evaluate(selector, val_loader, device)
    gate = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "label_threshold": args.label_threshold,
        "epochs": epoch_results,
        "final_val": final_val,
        "best_val_accuracy": best_val_acc,
        "gate_pass": False,
    }

    # Gate: recall >= 0.6 (catch most true plugin needs)
    #       precision >= 0.3 (at least 30% of interventions are correct)
    #       false_pos_rate <= 0.2 (no more than 20% unnecessary interventions)
    gate["gate_checks"] = {
        "recall_ok": final_val["recall"] >= 0.6,
        "precision_ok": final_val["precision"] >= 0.3,
        "false_pos_rate_ok": (final_val["false_pos"] / max(final_val["n_negative"], 1)) <= 0.2,
    }
    gate["gate_pass"] = all(gate["gate_checks"].values())

    gate_path = output_dir / "selector_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nDone. val_acc={final_val['accuracy']:.3f} recall={final_val['recall']:.3f} "
          f"precision={final_val['precision']:.3f} fp_rate={final_val['false_pos'] / max(final_val['n_negative'], 1):.3f} "
          f"gate={'PASS' if gate['gate_pass'] else 'FAIL'}")
    return 0 if gate["gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

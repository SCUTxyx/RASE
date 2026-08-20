#!/usr/bin/env python3
"""PRE-C0-R2: train and validate a lean F0 activation gate.

Validation is grouped by task (or source episode as a fallback), never by
random snapshots.  This avoids placing two snapshots from the same rollout in
both train and validation.  The decision threshold is selected from out-of-fold
predictions using rescue utility with an explicit harm constraint; raw accuracy
is reported but is not a gate because an always-off classifier is already very
accurate on imbalanced data.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LeanActivationGate(nn.Module):
    """Small activation classifier using aligned history and lean features."""

    def __init__(
        self,
        proprio_dim: int = 8,
        action_dim: int = 7,
        history_window: int = 8,
        obs_feature_dim: int = 16,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.history_window = history_window
        self.obs_feature_dim = obs_feature_dim
        self.hidden_dim = hidden_dim

        history_input_dim = history_window * (proprio_dim + action_dim + 1 + action_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feature_dim, hidden_dim), nn.ReLU()
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(action_dim + 1, 8), nn.ReLU()
        )
        layers: list[nn.Module] = []
        in_dim = hidden_dim * 2 + action_dim + 8
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, history, obs_features, student_action, plugin_delta, delta_norm):
        h_enc = self.history_encoder(history.view(history.size(0), -1))
        o_enc = self.obs_encoder(obs_features.view(obs_features.size(0), -1))
        d_enc = self.delta_encoder(torch.cat([
            plugin_delta.view(plugin_delta.size(0), -1),
            delta_norm.view(delta_norm.size(0), -1),
        ], dim=-1))
        return self.head(torch.cat([h_enc, o_enc, student_action, d_enc], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict(self, history, obs_features, student_action, plugin_delta, delta_norm) -> float:
        device = next(self.parameters()).device
        h = torch.as_tensor(np.asarray(history, dtype=np.float32), device=device).unsqueeze(0)
        o = torch.as_tensor(np.asarray(obs_features, dtype=np.float32), device=device).unsqueeze(0)
        a = torch.as_tensor(np.asarray(student_action, dtype=np.float32), device=device).reshape(1, -1)
        d = torch.as_tensor(np.asarray(plugin_delta, dtype=np.float32), device=device).reshape(1, -1)
        n = torch.tensor([[float(delta_norm)]], dtype=torch.float32, device=device)
        return torch.sigmoid(self.forward(h, o, a, d, n)).item()


def load_labels(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_key(row: dict) -> str:
    if row.get("episode_id"):
        return str(row["episode_id"])
    return f"{row.get('task_id')}:{row.get('init_state_id')}:{row.get('seed')}"


def build_batch(rows: list[dict], device: str) -> dict[str, torch.Tensor]:
    histories, obs, actions, deltas, norms, labels, weights = [], [], [], [], [], [], []
    for row in rows:
        feat = row["features"]
        history_value = feat["history_flat"] if "history_flat" in feat else feat["history"]
        obs_value = feat["obs_features_lean"] if "obs_features_lean" in feat else feat["obs_features"]
        histories.append(np.asarray(history_value, np.float32).reshape(-1))
        obs.append(np.asarray(obs_value, np.float32))
        actions.append(np.asarray(feat["student_action"], np.float32))
        deltas.append(np.asarray(feat["plugin_delta"], np.float32))
        norms.append(float(feat.get("delta_norm", 0.0)))
        labels.append(float(row["label"]))
        weights.append(2.0 if row.get("label_type") == "negative_harm" else 1.0)
    return {
        "history": torch.from_numpy(np.stack(histories)).to(device),
        "obs_features": torch.from_numpy(np.stack(obs)).to(device),
        "student_action": torch.from_numpy(np.stack(actions)).to(device),
        "plugin_delta": torch.from_numpy(np.stack(deltas)).to(device),
        "delta_norm": torch.tensor(norms, dtype=torch.float32, device=device).unsqueeze(-1),
        "label": torch.tensor(labels, dtype=torch.float32, device=device),
        "weight": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def model_logits(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["history"], batch["obs_features"], batch["student_action"],
                 batch["plugin_delta"], batch["delta_norm"])


def fit_split(train_rows, val_rows, *, device, epochs, lr, hidden_dim, seed, patience):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LeanActivationGate(hidden_dim=hidden_dim).to(device)
    train = build_batch(train_rows, device)
    val = build_batch(val_rows, device)
    n_pos = max(1, sum(int(r["label"] == 1) for r in train_rows))
    n_neg = max(1, len(train_rows) - n_pos)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(n_neg / n_pos), device=device), reduction="none"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        losses = loss_fn(model_logits(model, train), train["label"])
        loss = (losses * train["weight"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model_logits(model, val), val["label"]).mean().item()
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model_logits(model, val)).cpu().numpy()
    return model, probs, best_epoch, best_loss, history


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p)
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / n_pos)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def threshold_metrics(rows, probs, threshold, harm_cost=2.0, neutral_cost=0.05):
    y = np.asarray([int(r["label"]) for r in rows])
    types = np.asarray([r.get("label_type", "") for r in rows])
    selected = probs >= threshold
    pos = y == 1
    neg = ~pos
    harm = types == "negative_harm"
    neutral = neg & ~harm
    tp, tn = int((selected & pos).sum()), int((~selected & neg).sum())
    fp, fn = int((selected & neg).sum()), int((~selected & pos).sum())
    recall = tp / max(1, int(pos.sum()))
    specificity = tn / max(1, int(neg.sum()))
    precision = tp / max(1, tp + fp)
    harm_fpr = int((selected & harm).sum()) / max(1, int(harm.sum()))
    utility = (tp - harm_cost * int((selected & harm).sum())
               - neutral_cost * int((selected & neutral).sum())) / max(1, len(rows))
    return {
        "threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "rescue_recall": float(recall), "precision": float(precision),
        "harm_activation_rate": float(harm_fpr), "utility": float(utility),
        "activation_rate": float(selected.mean()),
    }


def make_folds(rows: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    tasks = sorted({str(r["task_id"]) for r in rows})
    folds = []
    if len(tasks) >= 3:
        for task in tasks:
            val = [r for r in rows if str(r["task_id"]) == task]
            train = [r for r in rows if str(r["task_id"]) != task]
            if train and val and any(r["label"] == 1 for r in train):
                folds.append((task, train, val))
    if folds:
        return folds
    groups = sorted({group_key(r) for r in rows})
    for fold_idx in range(min(4, len(groups))):
        val_groups = set(groups[fold_idx::4])
        val = [r for r in rows if group_key(r) in val_groups]
        train = [r for r in rows if group_key(r) not in val_groups]
        if train and val and any(r["label"] == 1 for r in train):
            folds.append((f"episode_fold_{fold_idx}", train, val))
    return folds


def train_final(rows, *, device, epochs, lr, hidden_dim, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LeanActivationGate(hidden_dim=hidden_dim).to(device)
    batch = build_batch(rows, device)
    n_pos = max(1, sum(int(r["label"] == 1) for r in rows))
    n_neg = max(1, len(rows) - n_pos)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(n_neg / n_pos), device=device), reduction="none"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    losses_out = []
    for _ in range(max(1, epochs)):
        model.train(); optimizer.zero_grad()
        losses = loss_fn(model_logits(model, batch), batch["label"])
        loss = (losses * batch["weight"]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses_out.append(float(loss.item()))
    model.eval()
    return model, losses_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-rescue-recall", type=float, default=0.5)
    parser.add_argument("--max-harm-fpr", type=float, default=0.25)
    parser.add_argument("--harm-cost", type=float, default=2.0)
    args = parser.parse_args()

    rows = load_labels(args.labels_path)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    n_pos = sum(int(r["label"] == 1) for r in rows)
    n_harm = sum(r.get("label_type") == "negative_harm" for r in rows)
    tasks = sorted({str(r["task_id"]) for r in rows})
    groups = {group_key(r) for r in rows}
    print(f"Loaded {len(rows)} snapshots: rescue={n_pos}, harm={n_harm}, tasks={len(tasks)}, episodes={len(groups)}")

    folds = make_folds(rows)
    if len(folds) < 2:
        raise RuntimeError("Need at least two leakage-free validation folds")
    oof_rows, oof_probs, best_epochs, fold_reports = [], [], [], []
    for fold_idx, (name, train_rows, val_rows) in enumerate(folds):
        _, probs, best_epoch, best_loss, _ = fit_split(
            train_rows, val_rows, device=args.device, epochs=args.epochs,
            lr=args.lr, hidden_dim=args.hidden_dim, seed=args.seed + fold_idx,
            patience=args.patience,
        )
        oof_rows.extend(val_rows); oof_probs.extend(probs.tolist()); best_epochs.append(best_epoch + 1)
        fold_reports.append({
            "fold": name, "n_train": len(train_rows), "n_val": len(val_rows),
            "n_val_positive": sum(int(r["label"] == 1) for r in val_rows),
            "best_epoch": best_epoch, "best_val_loss": best_loss,
        })
        print(f"Fold {name}: train={len(train_rows)} val={len(val_rows)} best_epoch={best_epoch} loss={best_loss:.4f}")

    probs = np.asarray(oof_probs, dtype=np.float64)
    y = np.asarray([int(r["label"]) for r in oof_rows])
    all_thresholds = [threshold_metrics(oof_rows, probs, t, args.harm_cost)
                      for t in np.linspace(0.05, 0.95, 91)]
    qualified = [m for m in all_thresholds
                 if m["rescue_recall"] >= args.min_rescue_recall
                 and m["harm_activation_rate"] <= args.max_harm_fpr]
    pool = qualified or all_thresholds
    chosen = max(pool, key=lambda m: (m["utility"], m["balanced_accuracy"], -m["activation_rate"]))
    prevalence = float(y.mean()) if len(y) else 0.0
    ap, auc = average_precision(y, probs), roc_auc(y, probs)
    gate_pass = bool(
        len(rows) >= 50 and n_pos >= 6 and n_harm >= 3 and len(tasks) >= 3
        and ap > prevalence and chosen["balanced_accuracy"] >= 0.55
        and chosen["rescue_recall"] >= args.min_rescue_recall
        and chosen["harm_activation_rate"] <= args.max_harm_fpr
        and chosen["utility"] > 0
    )

    final_epochs = int(max(20, np.median(best_epochs)))
    model, final_losses = train_final(
        rows, device=args.device, epochs=final_epochs, lr=args.lr,
        hidden_dim=args.hidden_dim, seed=args.seed,
    )
    config = {
        "proprio_dim": 8, "action_dim": 7, "history_window": 8,
        "obs_feature_dim": 16, "hidden_dim": args.hidden_dim,
        "num_layers": 1, "dropout": 0.1,
    }
    report = {
        "schema_version": "rase-activation-gate-training/v2",
        "gate_pass": gate_pass,
        "n_snapshots": len(rows), "n_rescue": n_pos, "n_harm": n_harm,
        "n_tasks": len(tasks), "n_episode_groups": len(groups), "tasks": tasks,
        "prevalence": prevalence, "average_precision": ap, "roc_auc": auc,
        "selected": chosen, "qualified_threshold_count": len(qualified),
        "folds": fold_reports, "final_epochs": final_epochs,
        "feature_schema_versions": sorted({str(r.get("feature_schema_version", "legacy")) for r in rows}),
        "oof_predictions": [
            {
                "snapshot_id": row.get("snapshot_id"),
                "task_id": row.get("task_id"),
                "episode_id": group_key(row),
                "label": int(row["label"]),
                "label_type": row.get("label_type"),
                "probability": float(prob),
            }
            for row, prob in zip(oof_rows, probs)
        ],
        "threshold_sweep": all_thresholds,
    }
    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "activation_threshold": chosen["threshold"], "cv_metrics": report,
        "final_train_loss": final_losses,
    }, out / "gate_checkpoint.pt")
    (out / "gate_training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {out / 'gate_checkpoint.pt'}; gate_pass={gate_pass}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

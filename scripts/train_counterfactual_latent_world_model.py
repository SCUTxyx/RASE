#!/usr/bin/env python3
"""Train an action-conditioned latent world model with counterfactual heads.

Input is JSONL with one deployment-defined decision boundary per row:
  task_id, state_key, latent, action, next_latent, operator_success
Optional: proprio, split.  ``operator_success`` maps operator names to 0/1.

The dynamics loss makes the representation a world model rather than a plain
outcome classifier.  Risk and operator heads answer two different questions:
whether CONTINUE is likely to fail and which intervention is recoverable.
Task-held-out OOF predictions are used for every model-selection claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def vec(row: dict, key: str) -> np.ndarray:
    value = np.asarray(row.get(key, []), dtype=np.float32).reshape(-1)
    if value.size == 0:
        raise ValueError(f"{row.get('state_key')}: missing {key}")
    return value


class Normalizer:
    def __init__(self, values: np.ndarray):
        self.mean = values.mean(0)
        self.std = values.std(0).clip(1e-5)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class CounterfactualLatentWM(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, proprio_dim: int,
                 n_operators: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        input_dim = latent_dim + action_dim + proprio_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.operator_head = nn.Linear(hidden_dim, n_operators)

    def forward(self, x):
        h = self.trunk(x)
        return self.delta_head(h), self.risk_head(h).squeeze(-1), self.operator_head(h)


def arrays(rows: list[dict], operators: list[str], norms=None):
    lat = np.stack([vec(r, "latent") for r in rows])
    act = np.stack([vec(r, "action") for r in rows])
    prop_size = (int(norms[2].mean.size) if norms is not None and norms[2] is not None
                 else max((np.asarray(r.get("proprio", [])).size for r in rows), default=0))
    prop = np.stack([np.pad(np.asarray(r.get("proprio", []), np.float32).reshape(-1),
                            (0, prop_size - np.asarray(r.get("proprio", [])).size)) for r in rows])
    nxt = np.stack([vec(r, "next_latent") for r in rows])
    if lat.shape != nxt.shape:
        raise ValueError(f"latent/next_latent shape mismatch: {lat.shape} vs {nxt.shape}")
    y_op = np.asarray([[float(r["operator_success"][op]) for op in operators] for r in rows], np.float32)
    y_risk = 1.0 - y_op[:, operators.index("CONTINUE")]
    if norms is None:
        norms = (Normalizer(lat), Normalizer(act), Normalizer(prop) if prop_size else None)
    lnorm, anorm, pnorm = norms
    x = np.concatenate([lnorm(lat), anorm(act), pnorm(prop) if pnorm else prop], axis=1)
    delta = lnorm(nxt) - lnorm(lat)
    return {"x": x, "delta": delta, "risk": y_risk, "operators": y_op}, norms


def tensorize(data: dict, device: str):
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in data.items()}


def fit_one(train_rows, val_rows, operators, *, device, seed, epochs, lr,
            hidden_dim, patience, dynamics_weight):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_np, norms = arrays(train_rows, operators)
    val_np, _ = arrays(val_rows, operators, norms)
    train, val = tensorize(train_np, device), tensorize(val_np, device)
    latent_dim = train_np["delta"].shape[1]
    action_dim = vec(train_rows[0], "action").size
    proprio_dim = train_np["x"].shape[1] - latent_dim - action_dim
    model = CounterfactualLatentWM(latent_dim, action_dim, proprio_dim,
                                   len(operators), hidden_dim).to(device)
    pos = train["operators"].sum(0).clamp_min(1)
    neg = (len(train_rows) - pos).clamp_min(1)
    pos_weight = (neg / pos).clamp(0.25, 4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best, best_state, stale = float("inf"), None, 0
    for _epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        pred_delta, risk_logits, op_logits = model(train["x"])
        dyn_loss = F.smooth_l1_loss(pred_delta, train["delta"])
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, train["risk"])
        op_loss = F.binary_cross_entropy_with_logits(op_logits, train["operators"],
                                                     pos_weight=pos_weight)
        loss = dynamics_weight * dyn_loss + risk_loss + op_loss
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            vd, vr, vo = model(val["x"])
            vloss = (dynamics_weight * F.smooth_l1_loss(vd, val["delta"])
                     + F.binary_cross_entropy_with_logits(vr, val["risk"])
                     + F.binary_cross_entropy_with_logits(vo, val["operators"])).item()
        if vloss < best - 1e-5:
            best, best_state, stale = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pd, pr, po = model(val["x"])
    return model, norms, {
        "delta": pd.cpu().numpy(), "risk": torch.sigmoid(pr).cpu().numpy(),
        "operators": torch.sigmoid(po).cpu().numpy(), "target_delta": val_np["delta"],
        "best_val_loss": best}


def fit_full(rows, operators, *, device, seed, epochs, lr, hidden_dim, dynamics_weight):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data_np, norms = arrays(rows, operators)
    data = tensorize(data_np, device)
    latent_dim = data_np["delta"].shape[1]
    action_dim = vec(rows[0], "action").size
    proprio_dim = data_np["x"].shape[1] - latent_dim - action_dim
    model = CounterfactualLatentWM(latent_dim, action_dim, proprio_dim,
                                   len(operators), hidden_dim).to(device)
    pos = data["operators"].sum(0).clamp_min(1)
    neg = (len(rows) - pos).clamp_min(1)
    pos_weight = (neg / pos).clamp(.25, 4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        pred_delta, risk_logits, op_logits = model(data["x"])
        loss = (dynamics_weight * F.smooth_l1_loss(pred_delta, data["delta"])
                + F.binary_cross_entropy_with_logits(risk_logits, data["risk"])
                + F.binary_cross_entropy_with_logits(op_logits, data["operators"],
                                                     pos_weight=pos_weight))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval()
    config = {"latent_dim": latent_dim, "action_dim": action_dim,
              "proprio_dim": proprio_dim, "n_operators": len(operators),
              "hidden_dim": hidden_dim}
    norm_state = {"latent": norms[0].state_dict(), "action": norms[1].state_dict(),
                  "proprio": norms[2].state_dict() if norms[2] else None}
    return model, config, norm_state


def average_precision(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0: return float("nan")
    order = np.argsort(-p); ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / y.sum())


def roc_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a > b) + .5 * (a == b) for a in pos for b in neg]))


def selector_metrics(rows, operators, probs, margin: float, std=None, z=1.64, costs=None):
    costs = costs or {}
    y = np.asarray([[float(r["operator_success"][op]) for op in operators] for r in rows])
    base_idx = operators.index("CONTINUE")
    conservative = probs - (z * std if std is not None else 0.0)
    utility = conservative - np.asarray([float(costs.get(op, 0.0)) for op in operators])
    choices = []
    for i in range(len(rows)):
        advantage = utility[i] - utility[i, base_idx]
        best = int(np.argmax(advantage))
        choices.append(best if best != base_idx and advantage[best] >= margin else base_idx)
    choices = np.asarray(choices)
    selected_success = y[np.arange(len(y)), choices]
    base_success = y[:, base_idx]
    rescue = int(((base_success == 0) & (selected_success == 1)).sum())
    harm = int(((base_success == 1) & (selected_success == 0)).sum())
    return {"success_rate": float(selected_success.mean()), "base_rate": float(base_success.mean()),
            "delta_pp": float(100 * (selected_success.mean() - base_success.mean())),
            "rescue": rescue, "harm": harm, "intervention_rate": float((choices != base_idx).mean()),
            "choices": [operators[i] for i in choices]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--dynamics-weight", type=float, default=1.0)
    ap.add_argument("--selector-margin", type=float, default=0.10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260808)
    args = ap.parse_args()
    rows = read_jsonl(args.dataset)
    operators = sorted(set.intersection(*(set(r["operator_success"]) for r in rows)))
    operators = ["CONTINUE"] + [op for op in operators if op != "CONTINUE"]
    tasks = sorted({str(r["task_id"]) for r in rows})
    if len(tasks) < 3: raise SystemExit("Need at least 3 tasks for task-held-out OOF evaluation")
    oof_probs = np.zeros((len(rows), len(operators)), np.float32)
    oof_std = np.zeros((len(rows), len(operators)), np.float32)
    oof_risk = np.zeros(len(rows), np.float32)
    oof_dyn, persistence = [], []
    row_index = {id(row): i for i, row in enumerate(rows)}
    folds = []
    for fold, task in enumerate(tasks):
        val_rows = [r for r in rows if str(r["task_id"]) == task]
        train_rows = [r for r in rows if str(r["task_id"]) != task]
        member_predictions = []
        member_risk = []
        for member in range(args.ensemble_size):
            _model, _norms, pred = fit_one(
                train_rows, val_rows, operators, device=args.device,
                seed=args.seed + fold * 100 + member, epochs=args.epochs, lr=args.lr,
                hidden_dim=args.hidden_dim, patience=args.patience,
                dynamics_weight=args.dynamics_weight)
            member_predictions.append(pred["operators"]); member_risk.append(pred["risk"])
            oof_dyn.append(float(np.mean(pred["delta"] ** 2)))
            persistence.append(float(np.mean(pred["target_delta"] ** 2)))
        mean_prob = np.mean(member_predictions, 0)
        std_prob = np.std(member_predictions, 0)
        for local, row in enumerate(val_rows):
            idx = row_index[id(row)]; oof_probs[idx] = mean_prob[local]
            oof_std[idx] = std_prob[local]
            oof_risk[idx] = np.mean(member_risk, 0)[local]
        folds.append({"held_out_task": task, "n": len(val_rows)})
    y_op = np.asarray([[float(r["operator_success"][op]) for op in operators] for r in rows])
    y_risk = 1 - y_op[:, operators.index("CONTINUE")]
    selector = selector_metrics(rows, operators, oof_probs, args.selector_margin, std=oof_std)
    report = {
        "schema_version": "rase-counterfactual-latent-wm/v1", "n": len(rows),
        "tasks": tasks, "operators": operators, "folds": folds,
        "dynamics_mse": float(np.mean(oof_dyn)),
        "persistence_mse": float(np.mean(persistence)),
        "dynamics_improvement": float(1 - np.mean(oof_dyn) / max(np.mean(persistence), 1e-12)),
        "risk_ap": average_precision(y_risk, oof_risk), "risk_auc": roc_auc(y_risk, oof_risk),
        "operator_brier": {op: float(np.mean((oof_probs[:, i] - y_op[:, i]) ** 2))
                           for i, op in enumerate(operators)},
        "selector_oof": selector,
        "gates": {
            "world_model_predictive": bool(np.mean(oof_dyn) <= .9 * np.mean(persistence)),
            "risk_discriminative": bool(roc_auc(y_risk, oof_risk) >= .65),
            "selector_positive": bool(selector["delta_pp"] >= 5 and selector["harm"] <= max(1, .05 * len(rows))),
        },
    }
    report["status"] = "ready_for_closed_loop_dev" if all(report["gates"].values()) else "not_ready"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof_predictions.jsonl").write_text("".join(
        json.dumps({"state_key": row["state_key"], "task_id": row["task_id"],
                    "risk": float(oof_risk[i]),
                    "operator_probabilities": {op: float(oof_probs[i, j]) for j, op in enumerate(operators)},
                    "selected_operator": selector["choices"][i]}) + "\n"
        for i, row in enumerate(rows)))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Deployment artifacts are trained only after all task-held-out predictions
    # are frozen.  They must never be used to revise the OOF gates above.
    final_epochs = min(args.epochs, 150)
    for member in range(args.ensemble_size):
        model, config, normalizers = fit_full(
            rows, operators, device=args.device, seed=args.seed + 10_000 + member,
            epochs=final_epochs, lr=args.lr, hidden_dim=args.hidden_dim,
            dynamics_weight=args.dynamics_weight)
        torch.save({"schema_version": "rase-counterfactual-latent-wm-checkpoint/v1",
                    "state_dict": model.state_dict(), "config": config,
                    "normalizers": normalizers, "operators": operators,
                    "selector_margin": args.selector_margin,
                    "training_rows": len(rows), "training_tasks": tasks,
                    "member": member, "seed": args.seed + 10_000 + member},
                   args.output_dir / f"member_{member:02d}.pt")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready_for_closed_loop_dev" else 2


if __name__ == "__main__":
    raise SystemExit(main())

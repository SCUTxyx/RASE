#!/usr/bin/env python3
"""Task-held-out OOF evaluation for the R6-C no-world-model dynamic risk baseline.

Each row is a boundary of a trajectory group (state_key:policy_id:seed_index).
The fixed controller is CONTINUE_SOURCE or ENTER_PERSISTENT_OFT with a
two-boundary dwell: a boundary is "risky" when the ensemble 1.64-sigma LCB of
source success is below the calibrated threshold; ENTER_PERSISTENT_OFT fires
only after ``dwell`` consecutive risky boundaries within a trajectory.

Modes mirror R6-B0: ``per_vla`` (train/eval one policy), ``zero_shot``
(train on a source policy, eval on a target policy), ``loo`` (train on all
policies but the target, eval on the target).  Every mode keeps all rows of a
trajectory group and of a true task in one outer fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RiskNet(nn.Module):
    def __init__(self, *, n_policies: int, use_policy_id: bool,
                 history_dim: int, language_dim: int = 0) -> None:
        super().__init__()
        self.use_policy_id = use_policy_id
        self.vision = nn.Sequential(
            nn.Conv2d(6, 24, 5, 2, 2), nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.GroupNorm(12, 96), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proprio = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU())
        self.action = nn.Sequential(nn.Linear(20, 48), nn.LayerNorm(48), nn.GELU())
        self.history = nn.Sequential(nn.Linear(history_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.progress = nn.Sequential(nn.Linear(1, 8), nn.LayerNorm(8), nn.GELU())
        self.language = (nn.Sequential(nn.Linear(language_dim, 48), nn.LayerNorm(48), nn.GELU())
                         if language_dim else None)
        self.policy = nn.Embedding(n_policies, 16) if use_policy_id else None
        fused = 96 + 32 + 48 + 32 + 8 + (48 if language_dim else 0) + (16 if use_policy_id else 0)
        self.core = nn.Sequential(
            nn.Linear(fused, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 96), nn.GELU(),
        )
        self.source = nn.Linear(96, 2)
        self.persistent = nn.Linear(96, 1)
        self.cost = nn.Linear(96, 3)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor,
                action: torch.Tensor, history: torch.Tensor,
                progress: torch.Tensor, language: torch.Tensor,
                policy: torch.Tensor) -> dict[str, torch.Tensor]:
        b = image.shape[0]
        visual = self.vision(image.reshape(b, 6, image.shape[-2], image.shape[-1]))
        pieces = [visual, self.proprio(proprio), self.action(action),
                  self.history(history), self.progress(progress)]
        if self.language is not None:
            pieces.append(self.language(language))
        if self.policy is not None:
            pieces.append(self.policy(policy))
        value = self.core(torch.cat(pieces, dim=-1))
        raw = self.source(value)
        alpha = F.softplus(raw[:, 0]) + 1e-4
        beta = F.softplus(raw[:, 1]) + 1e-4
        return {
            "alpha_raw": raw[:, 0], "beta_raw": raw[:, 1],
            "source_mean": alpha / (alpha + beta),
            "persistent_logit": self.persistent(value).squeeze(-1),
            "cost_quantiles": torch.cumsum(F.softplus(self.cost(value)), dim=-1),
        }


def beta_binomial_nll(alpha_raw: torch.Tensor, beta_raw: torch.Tensor,
                      successes: torch.Tensor, trials: torch.Tensor) -> torch.Tensor:
    alpha = F.softplus(alpha_raw) + 1e-4
    beta = F.softplus(beta_raw) + 1e-4
    failures = trials - successes
    log_comb = torch.lgamma(trials + 1) - torch.lgamma(successes + 1) - torch.lgamma(failures + 1)
    log_beta = (torch.lgamma(successes + alpha) + torch.lgamma(failures + beta)
                - torch.lgamma(trials + alpha + beta) - torch.lgamma(alpha)
                - torch.lgamma(beta) + torch.lgamma(alpha + beta))
    return -(log_comb + log_beta).mean()


def folds(tasks: list[str], count: int, seed: int) -> list[set[str]]:
    values = sorted(set(tasks))
    random.Random(seed).shuffle(values)
    return [set(values[index::count]) for index in range(count)]


def calibration_tasks(train_tasks: set[str], fold: int, count: int = 6) -> set[str]:
    values = sorted(train_tasks)
    offset = (fold * count) % len(values)
    rotated = values[offset:] + values[:offset]
    return set(rotated[: min(count, max(1, len(values) // 3))])


def normalize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[fit].mean(0)
    std = values[fit].std(0)
    std[std < 1e-5] = 1.0
    return (values - mean) / std, mean, std


def train_member(data: dict[str, np.ndarray], fit_idx: np.ndarray, *,
                 use_policy_id: bool, use_language: bool, seed: int,
                 epochs: int, device: str) -> RiskNet:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    task_values = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0xB0075A9)
    sampled = [rng.choice(task_values) for _ in task_values]
    boot = np.concatenate([fit_idx[data["task_id"][fit_idx] == task] for task in sampled])
    prop, _, _ = normalize(data["proprio"], boot)
    action, _, _ = normalize(data["action_summary"], boot)
    history, _, _ = normalize(data["history"], boot)
    progress, _, _ = normalize(data["elapsed_progress"].reshape(-1, 1), boot)
    language_dim = int(data["language_hash"].shape[1]) if use_language else 0
    model = RiskNet(n_policies=int(data["policy_index"].max()) + 1,
                    use_policy_id=use_policy_id, history_dim=data["history"].shape[1],
                    language_dim=language_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    q = torch.tensor([0.1, 0.5, 0.9], device=device)
    image = torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0, device=device)
    p = torch.as_tensor(prop[boot], device=device)
    a = torch.as_tensor(action[boot], device=device)
    h = torch.as_tensor(history[boot], device=device)
    g = torch.as_tensor(progress[boot], device=device)
    language = torch.as_tensor(data["language_hash"][boot], device=device) if use_language else torch.empty((len(boot), 0), device=device)
    policy = torch.as_tensor(data["policy_index"][boot], device=device)
    succ = torch.as_tensor(data["source_success"][boot], device=device)
    trials = torch.as_tensor(data["source_trials"][boot], device=device)
    persistent = torch.as_tensor(data["persistent_success"][boot], device=device)
    target_cost = torch.log1p(torch.as_tensor(data["persistent_teacher_steps"][boot], device=device))
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(image, p, a, h, g, language, policy)
        source_loss = beta_binomial_nll(out["alpha_raw"], out["beta_raw"], succ, trials)
        persistent_loss = F.binary_cross_entropy_with_logits(out["persistent_logit"], persistent)
        error = target_cost[:, None] - out["cost_quantiles"]
        cost_loss = torch.maximum(q * error, (q - 1) * error).mean()
        loss = source_loss + 0.35 * persistent_loss + 0.12 * cost_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    model._r6_prop_mean = torch.as_tensor(data["proprio"][boot].mean(0), device=device)
    prop_std = data["proprio"][boot].std(0); prop_std[prop_std < 1e-5] = 1.0
    model._r6_prop_std = torch.as_tensor(prop_std, device=device)
    model._r6_action_mean = torch.as_tensor(data["action_summary"][boot].mean(0), device=device)
    action_std = data["action_summary"][boot].std(0); action_std[action_std < 1e-5] = 1.0
    model._r6_action_std = torch.as_tensor(action_std, device=device)
    model._r6_history_mean = torch.as_tensor(data["history"][boot].mean(0), device=device)
    history_std = data["history"][boot].std(0); history_std[history_std < 1e-5] = 1.0
    model._r6_history_std = torch.as_tensor(history_std, device=device)
    model._r6_progress_mean = torch.as_tensor(data["elapsed_progress"][boot].mean(), device=device)
    progress_std = float(data["elapsed_progress"][boot].std())
    model._r6_progress_std = torch.as_tensor(max(progress_std, 1e-5), device=device)
    return model


@torch.no_grad()
def predict(model: RiskNet, data: dict[str, np.ndarray], idx: np.ndarray, device: str) -> dict[str, np.ndarray]:
    image = torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device)
    prop = (torch.as_tensor(data["proprio"][idx], device=device) - model._r6_prop_mean) / model._r6_prop_std
    action = (torch.as_tensor(data["action_summary"][idx], device=device) - model._r6_action_mean) / model._r6_action_std
    history = (torch.as_tensor(data["history"][idx], device=device) - model._r6_history_mean) / model._r6_history_std
    progress = (torch.as_tensor(data["elapsed_progress"][idx], device=device) - model._r6_progress_mean) / model._r6_progress_std
    progress = progress.reshape(-1, 1)
    language = (torch.as_tensor(data["language_hash"][idx], device=device)
                if model.language is not None else torch.empty((len(idx), 0), device=device))
    policy = torch.as_tensor(data["policy_index"][idx], device=device)
    out = model(image, prop, action, history, progress, language, policy)
    return {"source": out["source_mean"].cpu().numpy(),
            "persistent": torch.sigmoid(out["persistent_logit"]).cpu().numpy(),
            "cost": torch.expm1(out["cost_quantiles"]).cpu().numpy()}


def group_boundaries(data: dict[str, np.ndarray], idx: np.ndarray) -> list[tuple[int, int]]:
    """Map each row to its trajectory group; return (group_id, order) per row."""
    order = np.argsort(idx, kind="stable")
    group_ids, group_orders = [], []
    seen: dict[str, int] = {}
    for position in order:
        group = str(data["group_id"][position])
        counter = seen.get(group, 0)
        seen[group] = counter + 1
        group_ids.append(group)
        group_orders.append(counter)
    return group_ids, group_orders


def controller_metrics(data: dict[str, np.ndarray], idx: np.ndarray,
                       lcb: np.ndarray, threshold: float, dwell: int) -> dict[str, Any]:
    """Two-boundary dwell controller, evaluated per trajectory group.

    Baseline is ENTER_PERSISTENT_OFT at the first boundary (t=0) of every
    trajectory.  A group enters persistent OFT at the first boundary where the
    risky condition (lcb < threshold) holds for ``dwell`` consecutive recorded
    boundaries; otherwise the source runs to completion.
    """
    group_ids, _ = group_boundaries(data, idx)
    by_group: dict[str, list[tuple[int, float]]] = {}
    for local, group in zip(range(len(idx)), group_ids):
        row = int(idx[local])
        by_group.setdefault(group, []).append((row, float(lcb[local])))
    entered = 0
    false_continue = 0
    controller_success = 0
    controller_teacher = 0.0
    baseline_success = 0
    baseline_teacher = 0.0
    n_trajectories = 0
    trajectory_records = []
    for group, members in by_group.items():
        members.sort(key=lambda item: item[0])  # elapsed_source_steps ascending
        rows = [item[0] for item in members]
        lcbs = np.asarray([item[1] for item in members])
        persistent = data["persistent_success"][rows].astype(bool)
        psteps = data["persistent_teacher_steps"][rows].astype(float)
        source_success = bool(data["source_success"][rows[0]])
        baseline_success += int(persistent[0])
        baseline_teacher += float(psteps[0])
        n_trajectories += 1
        streak = 0
        enter_at = None
        for position, row in enumerate(rows):
            if lcbs[position] < threshold:
                streak += 1
                if streak >= dwell:
                    enter_at = position
                    break
            else:
                streak = 0
        if enter_at is not None:
            entered += 1
            controller_success += int(persistent[enter_at])
            controller_teacher += float(psteps[enter_at])
            decided_continue = False
        else:
            decided_continue = True
            controller_success += int(source_success)
            controller_teacher += 0.0
            if (not source_success) and persistent[0]:
                false_continue += 1
        trajectory_records.append({
            "group_id": group, "state_key": str(data["state_key"][rows[0]]),
            "task_id": str(data["task_id"][rows[0]]),
            "n_boundaries": len(rows),
            "entered_persistent": enter_at is not None,
            "enter_elapsed": int(data["elapsed_source_steps"][rows[enter_at]]) if enter_at is not None else None,
            "source_success": bool(source_success),
            "persistent_success_at_0": bool(persistent[0]),
            "controller_success": bool(controller_success),
            "decided_continue": decided_continue,
            "controller_teacher_steps": float(controller_teacher),
            "baseline_teacher_steps": float(psteps[0]),
        })
    return {
        "episodes": float(n_trajectories),
        "entered": float(entered),
        "successes": float(controller_success),
        "baseline_successes": float(baseline_success),
        "success_gap": float((controller_success - baseline_success) / n_trajectories),
        "false_continue": float(false_continue),
        "false_continue_rate": float(false_continue / max(1, baseline_success)),
        "teacher_steps": float(controller_teacher),
        "baseline_teacher_steps": float(baseline_teacher),
        "savings": float(1.0 - controller_teacher / max(1.0, baseline_teacher)),
        "trajectories": trajectory_records,
    }


def select_threshold(data: dict[str, np.ndarray], idx: np.ndarray, lcb: np.ndarray,
                     dwell: int) -> tuple[float, dict[str, Any]]:
    best = None
    values = sorted(set([-0.01, 1.01, *np.linspace(0, 1, 101).tolist(), *lcb.tolist()]))
    for threshold in values:
        metrics = controller_metrics(data, idx, lcb, threshold, dwell)
        if metrics["success_gap"] < -0.05 or metrics["false_continue_rate"] > 0.05:
            continue
        rank = (metrics["savings"], metrics["success_gap"], -threshold)
        if best is None or rank > best[0]: best = (rank, float(threshold), metrics)
    if best is None:
        threshold = 1.01
        return threshold, controller_metrics(data, idx, lcb, threshold, dwell)
    return best[1], best[2]


def eligible_masks(data: dict[str, np.ndarray], mode: str, target: str | None,
                   source: str | None) -> tuple[np.ndarray, np.ndarray, bool]:
    policy = data["policy_id"]
    if mode == "per_vla":
        if not target: raise ValueError("per_vla requires --target-policy")
        mask = policy == target; return mask, mask, False
    if mode == "zero_shot":
        if not target or not source: raise ValueError("zero_shot requires target and source")
        return policy == source, policy == target, False
    if mode == "loo":
        if not target: raise ValueError("loo requires --target-policy")
        return policy != target, policy == target, False
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["per_vla", "zero_shot", "loo"], required=True)
    parser.add_argument("--target-policy")
    parser.add_argument("--source-policy")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260810)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dwell", type=int, default=2)
    parser.add_argument("--lcb-z", type=float, default=1.6448536269514722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-language", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset/report lock mismatch")
    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    if args.use_language and "language_hash" not in data:
        raise ValueError("--use-language requires language_hash in the dataset")
    train_policy_mask, eval_policy_mask, use_policy_id = eligible_masks(
        data, args.mode, args.target_policy, args.source_policy)
    task_folds = folds(data["task_id"].tolist(), args.folds, args.fold_seed)
    predictions: list[dict[str, Any]] = []
    fold_reports = []
    for fold, validation_tasks in enumerate(task_folds):
        train_tasks = set(data["task_id"].tolist()) - validation_tasks
        cal_tasks = calibration_tasks(train_tasks, fold)
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.where(train_policy_mask & np.isin(data["task_id"], list(fit_tasks)))[0]
        cal_idx = np.where(train_policy_mask & np.isin(data["task_id"], list(cal_tasks)))[0]
        val_idx = np.where(eval_policy_mask & np.isin(data["task_id"], list(validation_tasks)))[0]
        if min(len(fit_idx), len(cal_idx), len(val_idx)) == 0:
            raise ValueError(f"fold {fold} has an empty partition")
        models = [train_member(data, fit_idx, use_policy_id=use_policy_id,
                               use_language=args.use_language,
                               seed=args.seed + fold * 1009 + member * 7919,
                               epochs=args.epochs, device=args.device)
                  for member in range(args.members)]
        cal_pred = [predict(model, data, cal_idx, args.device) for model in models]
        cal_source = np.stack([value["source"] for value in cal_pred])
        cal_lcb = np.clip(cal_source.mean(0) - args.lcb_z * cal_source.std(0), 0, 1)
        val_pred = [predict(model, data, val_idx, args.device) for model in models]
        source = np.stack([value["source"] for value in val_pred])
        persistent = np.stack([value["persistent"] for value in val_pred])
        cost = np.stack([value["cost"] for value in val_pred])
        lcb = np.clip(source.mean(0) - args.lcb_z * source.std(0), 0, 1)
        threshold, cal_metrics = select_threshold(data, cal_idx, cal_lcb, args.dwell)
        metrics = controller_metrics(data, val_idx, lcb, threshold, args.dwell)
        fold_reports.append({
            "fold": fold, "threshold": threshold,
            "fit_rows": len(fit_idx), "calibration_rows": len(cal_idx),
            "validation_rows": len(val_idx), "validation_groups": metrics["episodes"],
            "calibration_metrics": {key: value for key, value in cal_metrics.items() if key != "trajectories"},
            "validation_metrics": {key: value for key, value in metrics.items() if key != "trajectories"},
        })
        for local, index in enumerate(val_idx):
            predictions.append({
                "index": int(index), "fold": fold,
                "group_id": str(data["group_id"][index]),
                "state_key": str(data["state_key"][index]),
                "task_id": str(data["task_id"][index]), "policy_id": str(data["policy_id"][index]),
                "elapsed_source_steps": int(data["elapsed_source_steps"][index]),
                "source_mean": float(source[:, local].mean()), "source_std": float(source[:, local].std()),
                "source_lcb": float(lcb[local]),
                "persistent_mean": float(persistent[:, local].mean()),
                "cost_q10": float(cost[:, local, 0].mean()), "cost_q50": float(cost[:, local, 1].mean()),
                "cost_q90": float(cost[:, local, 2].mean()),
                "threshold": float(threshold),
                "risky": bool(lcb[local] < threshold),
            })
        del models
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    predictions.sort(key=lambda row: (row["group_id"], row["elapsed_source_steps"]))
    all_idx = np.asarray([row["index"] for row in predictions], dtype=int)
    all_lcb = np.asarray([row["source_lcb"] for row in predictions], dtype=float)
    threshold = float(np.asarray([row["threshold"] for row in predictions]).mean())
    metrics = controller_metrics(data, all_idx, all_lcb, threshold, args.dwell)
    by_policy = {}
    for policy in sorted(set(data["policy_id"][all_idx].tolist())):
        mask = data["policy_id"][all_idx] == policy
        sub = controller_metrics(data, all_idx[mask], all_lcb[mask], threshold, args.dwell)
        by_policy[policy] = {key: value for key, value in sub.items() if key != "trajectories"}
    gate = (metrics["success_gap"] >= -0.05 and metrics["false_continue_rate"] <= 0.05
            and metrics["savings"] >= 0.20 and all(
                value["success_gap"] >= -0.05
                and value["false_continue_rate"] <= 0.05
                and value["savings"] >= 0.20
                for value in by_policy.values()
            ))
    result = {
        "schema_version": "rase-r6c-dynamic-risk-oof/v1", "status": "complete",
        "scientific_scope": "no-world-model dynamic boundary risk; two-boundary dwell",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "mode": args.mode, "target_policy": args.target_policy, "source_policy": args.source_policy,
        "seed": args.seed, "fold_seed": args.fold_seed, "folds": args.folds,
        "members": args.members, "epochs": args.epochs, "dwell": args.dwell, "lcb_z": args.lcb_z,
        "use_language": args.use_language,
        "metrics": {key: value for key, value in metrics.items() if key != "trajectories"},
        "metrics_by_policy": by_policy,
        "seed_gate_passed": gate, "fold_reports": fold_reports,
        "trajectory_records": metrics["trajectories"],
        "predictions": predictions,
        "parameter_count": int(sum(p.numel() for p in RiskNet(
            n_policies=int(data["policy_index"].max()) + 1,
            use_policy_id=use_policy_id, history_dim=data["history"].shape[1],
            language_dim=int(data["language_hash"].shape[1]) if args.use_language else 0).parameters())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ["mode", "target_policy", "seed", "metrics", "metrics_by_policy", "seed_gate_passed", "parameter_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task-held-out R7-A source-failure representation probe.

This stage predicts one target only: whether the declared source VLA ultimately fails when it
continues from the frozen t0 reset state.  It refuses to run unless the frozen
label-support audit passed and the dataset checksum matches its report.  OFT,
selector, cost and world-model targets are intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.light_risk_student import SourceRiskStudent  # noqa: E402
from rase.risk.r7_source_protocol import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    calibration_tasks,
    task_folds,
)
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(((positive[:, None] > negative[None, :]).mean()
                  + 0.5 * (positive[:, None] == negative[None, :]).mean()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray,
                               bins: int = 10) -> float:
    total, result = len(labels), 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (probability >= lower) & ((probability < upper) if upper < 1.0 else
                                        (probability <= upper))
        if mask.any():
            result += float(mask.sum()) / total * abs(
                float(labels[mask].mean()) - float(probability[mask].mean()))
    return result


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(labels.mean())
    return {
        "rows": int(len(labels)), "prevalence": prevalence,
        "auroc": binary_auc(labels, probability),
        "average_precision": average_precision(labels, probability),
        "ap_above_prevalence": average_precision(labels, probability) - prevalence,
        "ece_10_equal_width": expected_calibration_error(labels, probability),
        "brier": float(np.mean((probability - labels) ** 2)),
    }


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit temperature and bias on outer-train calibration tasks only."""
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.25, max_iter=80,
                                  line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated = x / log_temperature.exp().clamp(0.05, 20.0) + bias
        loss = F.binary_cross_entropy_with_logits(calibrated, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0)), float(bias.detach())


def normalize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = values[fit].mean(0), values[fit].std(0)
    std[std < 1e-5] = 1.0
    return (values - mean) / std, mean, std


def train_member(data: dict[str, np.ndarray], fit_idx: np.ndarray, *, seed: int,
                 epochs: int, device: str) -> tuple[SourceRiskStudent, dict[str, np.ndarray]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tasks = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0xB0075A9)
    sampled_tasks = [rng.choice(tasks) for _ in tasks]
    boot = np.concatenate([fit_idx[data["task_id"][fit_idx] == task]
                           for task in sampled_tasks])
    proprio, prop_mean, prop_std = normalize(data["proprio"], boot)
    action, action_mean, action_std = normalize(data["action_summary"], boot)
    encoder = TinyUniversalStateEncoder(
        image_size=96, proprio_dim=8, text_embed_dim=data["language_hash"].shape[1],
        hidden_dim=128, output_dim=128, dropout=0.1, input_mode="image",
    )
    model = SourceRiskStudent(
        encoder, action_dim=data["action_summary"].shape[1], fused_dim=128,
        head_hidden=128, n_members=1, dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    image = torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0, device=device)
    prop = torch.as_tensor(proprio[boot], device=device)
    act = torch.as_tensor(action[boot], device=device)
    text = torch.as_tensor(data["language_hash"][boot], device=device)
    target = torch.as_tensor(data["source_failure"][boot], device=device)
    positives = float(target.sum().item())
    pos_weight = torch.tensor(
        min(8.0, max(0.125, (len(target) - positives) / max(1.0, positives))),
        device=device,
    )
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logit = model(image, prop, act, text)["source_failure_logit"][0]
        loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, {
        "prop_mean": prop_mean, "prop_std": prop_std,
        "action_mean": action_mean, "action_std": action_std,
    }


@torch.no_grad()
def predict(model: SourceRiskStudent, stats: dict[str, np.ndarray],
            data: dict[str, np.ndarray], idx: np.ndarray, device: str) -> np.ndarray:
    image = torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device)
    prop = torch.as_tensor(
        (data["proprio"][idx] - stats["prop_mean"]) / stats["prop_std"], device=device)
    action = torch.as_tensor(
        (data["action_summary"][idx] - stats["action_mean"]) / stats["action_std"],
        device=device,
    )
    text = torch.as_tensor(data["language_hash"][idx], device=device)
    return model(image, prop, action, text)["source_failure_logit"][0].cpu().numpy()


def task_bootstrap(labels: np.ndarray, probability: np.ndarray, task_id: np.ndarray,
                   *, seed: int, samples: int) -> dict[str, dict[str, float]]:
    tasks = sorted(set(task_id.tolist()))
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "average_precision": [], "ece": []}
    for _ in range(samples):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        idx = np.concatenate([np.flatnonzero(task_id == task) for task in sampled])
        row = metrics(labels[idx], probability[idx])
        for name, value in (("auroc", row["auroc"]),
                            ("average_precision", row["average_precision"]),
                            ("ece", row["ece_10_equal_width"])):
            if math.isfinite(value):
                values[name].append(value)
    return {name: {
        "mean": float(np.mean(value)),
        "lower_95": float(np.quantile(value, 0.025)),
        "upper_95": float(np.quantile(value, 0.975)),
    } for name, value in values.items() if value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--exact-repeat-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audit = json.loads(args.label_audit.read_text())
    repeat_audit = json.loads(args.exact_repeat_audit.read_text())
    report = json.loads(args.dataset_report.read_text())
    expected_rows = int(audit.get("states", -1))
    if (audit.get("status") != "PASS" or expected_rows not in (191, 192)
            or int(audit.get("tasks", -1)) != 48
            or not all(bool(value) for value in (audit.get("gate") or {}).values())):
        raise ValueError("R7 label-support gate is not PASS")
    exclusion_sha = audit.get("exclusion_manifest_sha256")
    if expected_rows == 191 and not exclusion_sha:
        raise ValueError("191-state R7 audit must bind a frozen exclusion manifest")
    if expected_rows == 192 and exclusion_sha:
        raise ValueError("192-state R7 audit must not declare an exclusion")
    if (repeat_audit.get("status") != "PASS"
            or int(repeat_audit.get("audited_records", -1)) != 16
            or repeat_audit.get("label_audit_sha256") != sha256(args.label_audit)):
        raise ValueError("R7 exact-repeat stability gate is not PASS or is unbound")
    if report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("R7 dataset checksum differs from frozen report")
    if report.get("label_audit_sha256") != sha256(args.label_audit):
        raise ValueError("R7 dataset report does not bind this label audit")
    if report.get("exact_repeat_audit_sha256") != sha256(args.exact_repeat_audit):
        raise ValueError("R7 dataset report does not bind this exact-repeat audit")
    if (int(report.get("rows", -1)) != expected_rows
            or int(report.get("tasks", -1)) != 48
            or report.get("exclusion_manifest_sha256") != exclusion_sha):
        raise ValueError("R7 dataset report row/task/exclusion contract mismatch")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if (len(data["source_failure"]) != expected_rows
            or len(set(data["task_id"].tolist())) != 48):
        raise ValueError(f"R7 source-risk protocol requires {expected_rows} rows / 48 tasks")
    policies = sorted(set(data["policy_id"].tolist()))
    if len(policies) != 1:
        raise ValueError("single-policy R7 probe requires exactly one policy_id")
    policy_id = str(policies[0])
    if str(report.get("policy_id") or policy_id) != policy_id:
        raise ValueError("dataset report / policy_id mismatch")
    if str(audit.get("policy_id") or policy_id) != policy_id:
        raise ValueError("label audit / policy_id mismatch")

    fold_sets = task_folds(data["task_id"], data["suite"], count=args.folds,
                           seed=args.fold_seed)
    oof_logit = np.full(len(data["source_failure"]), np.nan, dtype=np.float64)
    oof_probability = np.full_like(oof_logit, np.nan)
    fold_reports = []
    all_tasks = set(data["task_id"].tolist())
    for fold, validation_tasks in enumerate(fold_sets):
        train_tasks = all_tasks - validation_tasks
        cal_tasks = calibration_tasks(train_tasks, data["task_id"], data["suite"],
                                      fold=fold, seed=args.fold_seed)
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.flatnonzero(np.isin(data["task_id"], list(fit_tasks)))
        cal_idx = np.flatnonzero(np.isin(data["task_id"], list(cal_tasks)))
        val_idx = np.flatnonzero(np.isin(data["task_id"], list(validation_tasks)))
        if not len(fit_idx) or not len(cal_idx) or not len(val_idx):
            raise ValueError(f"empty R7 split in fold {fold}")
        for partition, indices in (("fit", fit_idx), ("calibration", cal_idx)):
            if len(np.unique(data["source_failure"][indices])) != 2:
                raise ValueError(f"fold {fold} {partition} partition lacks both labels")
        member_cal, member_val = [], []
        for member in range(args.members):
            member_seed = args.seed + fold * 1009 + member * 7919
            model, stats = train_member(data, fit_idx, seed=member_seed,
                                        epochs=args.epochs, device=args.device)
            member_cal.append(predict(model, stats, data, cal_idx, args.device))
            member_val.append(predict(model, stats, data, val_idx, args.device))
        cal_logit = np.mean(member_cal, axis=0)
        val_logit = np.mean(member_val, axis=0)
        temperature, bias = fit_platt(cal_logit, data["source_failure"][cal_idx])
        val_probability = 1.0 / (1.0 + np.exp(-(val_logit / temperature + bias)))
        oof_logit[val_idx] = val_logit
        oof_probability[val_idx] = val_probability
        fold_reports.append({
            "fold": fold, "fit_tasks": sorted(fit_tasks),
            "calibration_tasks": sorted(cal_tasks),
            "validation_tasks": sorted(validation_tasks),
            "fit_rows": int(len(fit_idx)), "calibration_rows": int(len(cal_idx)),
            "validation_rows": int(len(val_idx)),
            "temperature": temperature, "bias": bias,
            "validation_metrics": metrics(data["source_failure"][val_idx], val_probability),
        })
    if not np.isfinite(oof_probability).all():
        raise AssertionError("OOF predictions are incomplete")
    labels = data["source_failure"].astype(np.float64)
    overall = metrics(labels, oof_probability)
    by_suite = {
        name: metrics(labels[data["suite"] == name], oof_probability[data["suite"] == name])
        for name in sorted(set(data["suite"].tolist()))
    }
    bootstrap = task_bootstrap(labels, oof_probability, data["task_id"], seed=args.seed,
                               samples=args.bootstrap_samples)
    point_gate = {
        "auroc_at_least_0p75": overall["auroc"] >= 0.75,
        "bootstrap_auroc_lower_at_least_0p65": bootstrap["auroc"]["lower_95"] >= 0.65,
        "ap_above_prevalence_at_least_0p10": overall["ap_above_prevalence"] >= 0.10,
        "ece_at_most_0p10": overall["ece_10_equal_width"] <= 0.10,
        "all_suite_auroc_above_0p60": all(row["auroc"] > 0.60 for row in by_suite.values()),
    }
    result = {
        "schema_version": "rase-r7a-source-risk-probe/v1",
        "scientific_scope": "development task-held-out source-failure representation probe",
        "status": "PASS" if all(point_gate.values()) else "FAIL",
        "policy_id": policy_id,
        "seed": args.seed, "fold_seed": args.fold_seed, "folds": args.folds,
        "members": args.members, "epochs": args.epochs,
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "label_audit": str(args.label_audit.resolve()),
        "exact_repeat_audit": str(args.exact_repeat_audit.resolve()),
        "exact_repeat_audit_sha256": sha256(args.exact_repeat_audit),
        "target": "source final failure only",
        "representation": "canonical_lightweight_initial_10_step_proposal",
        "metrics": overall, "metrics_by_suite": by_suite,
        "task_bootstrap": bootstrap, "gate": point_gate,
        "fold_reports": fold_reports,
        "forbidden_and_absent": ["OFT labels", "teacher cost", "selector target",
                                 "future frames", "world-model features"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        state_key=data["state_key"], task_id=data["task_id"], suite=data["suite"],
        source_failure=labels.astype(np.float32), raw_oof_logit=oof_logit.astype(np.float32),
        calibrated_oof_probability=oof_probability.astype(np.float32),
    )
    print(json.dumps({"status": result["status"], "metrics": overall,
                      "bootstrap_auroc": bootstrap["auroc"], "gate": point_gate},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

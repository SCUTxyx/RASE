#!/usr/bin/env python3
"""Task-held-out R8-B local fallback-recoverability hazard probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.light_risk_student import RecoverabilityHazardStudent  # noqa: E402
from rase.risk.r7_source_protocol import calibration_tasks, task_folds  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
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


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probability >= lower) & ((probability < upper) if index + 1 < bins
                                        else (probability <= upper))
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probability[mask].mean()))
    return result


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prevalence = float(labels.mean())
    ap = average_precision(labels, probability)
    return {
        "rows": int(len(labels)), "positives": int(labels.sum()),
        "prevalence": prevalence, "auroc": auc(labels, probability),
        "average_precision": ap, "ap_above_prevalence": ap - prevalence,
        "ece_10_equal_width": ece(labels, probability),
        "brier": float(np.mean((probability - labels) ** 2)),
    }


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Platt calibration requires both labels")
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.25, max_iter=80,
                                  line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = x / log_temperature.exp().clamp(0.05, 20.0) + bias
        loss = F.binary_cross_entropy_with_logits(value, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0)), float(bias.detach())


def build_transitions(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(source["group_id"]):
        grouped[str(group)].append(index)
    feature_index, current, following, start_elapsed = [], [], [], []
    for indices in grouped.values():
        by_elapsed = {int(source["elapsed_source_steps"][i]): i for i in indices}
        if set(by_elapsed) != {0, 8, 16}:
            continue
        for start, end in ((0, 8), (8, 16)):
            first, second = by_elapsed[start], by_elapsed[end]
            first_trials = int(source["persistent_trials"][first])
            second_trials = int(source["persistent_trials"][second])
            first_successes = int(source["persistent_successes"][first])
            second_successes = int(source["persistent_successes"][second])
            if (first_trials < 1 or second_trials < 1
                    or first_successes not in (0, first_trials)
                    or second_successes not in (0, second_trials)):
                continue
            feature_index.append(first)
            current.append(float(first_successes == first_trials))
            following.append(float(second_successes == second_trials))
            start_elapsed.append(start)
    idx = np.asarray(feature_index, dtype=np.int64)
    current_array = np.asarray(current, dtype=np.float32)
    next_array = np.asarray(following, dtype=np.float32)
    elapsed = np.asarray(start_elapsed, dtype=np.int64)
    data = {key: source[key][idx] for key in (
        "image", "proprio", "action_summary", "history", "language_hash",
        "state_key", "task_id", "suite", "group_id", "cohort_role",
        "policy_id", "policy_index",
    )}
    data.update({
        "elapsed_source_steps": elapsed,
        "elapsed_context": np.stack([elapsed / 16.0, (elapsed == 8).astype(np.float32)],
                                    axis=-1).astype(np.float32),
        "current_recoverable": current_array,
        "next_recoverable": next_array,
        "loss_hazard": (current_array * (1.0 - next_array)).astype(np.float32),
    })
    return data


def normalize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = values[fit].mean(0), values[fit].std(0)
    std[std < 1e-5] = 1.0
    return (values - mean) / std, mean, std


def positive_weight(target: torch.Tensor) -> torch.Tensor:
    positives = float(target.sum().item())
    return torch.tensor(min(8.0, max(0.125, (len(target) - positives)
                                                / max(1.0, positives))),
                        device=target.device)


def train_member(data: dict[str, np.ndarray], fit_idx: np.ndarray, *, seed: int,
                 epochs: int, device: str, policy_conditioning: str
                 ) -> tuple[RecoverabilityHazardStudent, dict[str, np.ndarray]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tasks = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0x8A2B007)
    sampled = [rng.choice(tasks) for _ in tasks]
    boot = np.concatenate([fit_idx[data["task_id"][fit_idx] == task] for task in sampled])
    proprio, prop_mean, prop_std = normalize(data["proprio"], boot)
    action, action_mean, action_std = normalize(data["action_summary"], boot)
    history, hist_mean, hist_std = normalize(data["history"], boot)
    encoder = TinyUniversalStateEncoder(
        image_size=96, proprio_dim=8, text_embed_dim=data["language_hash"].shape[1],
        hidden_dim=128, output_dim=128, dropout=0.1, input_mode="image",
    )
    n_policies = int(data["policy_index"].max()) + 1 if policy_conditioning == "id" else 0
    model = RecoverabilityHazardStudent(
        encoder, action_dim=data["action_summary"].shape[1],
        history_dim=data["history"].shape[1], n_policies=n_policies,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    tensors = {
        "image": torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0,
                                 device=device),
        "proprio": torch.as_tensor(proprio[boot], device=device),
        "action": torch.as_tensor(action[boot], device=device),
        "history": torch.as_tensor(history[boot], device=device),
        "elapsed": torch.as_tensor(data["elapsed_context"][boot], device=device),
        "text": torch.as_tensor(data["language_hash"][boot], device=device),
        "policy": torch.as_tensor(data["policy_index"][boot], dtype=torch.long,
                                  device=device),
        "current": torch.as_tensor(data["current_recoverable"][boot], device=device),
        "next": torch.as_tensor(data["next_recoverable"][boot], device=device),
        "hazard": torch.as_tensor(data["loss_hazard"][boot], device=device),
    }
    safe_mask = tensors["current"].bool()
    if len(torch.unique(tensors["hazard"][safe_mask])) != 2:
        raise ValueError("bootstrap fit sample lacks both conditional hazard labels")
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["image"], tensors["proprio"], tensors["action"],
            tensors["history"], tensors["elapsed"], tensors["text"],
            policy_index=tensors["policy"] if policy_conditioning == "id" else None,
        )
        current_loss = F.binary_cross_entropy_with_logits(
            output["current_recoverable_logit"], tensors["current"],
            pos_weight=positive_weight(tensors["current"]),
        )
        next_loss = F.binary_cross_entropy_with_logits(
            output["next_recoverable_logit"], tensors["next"],
            pos_weight=positive_weight(tensors["next"]),
        )
        hazard_loss = F.binary_cross_entropy_with_logits(
            output["loss_hazard_logit"][safe_mask], tensors["hazard"][safe_mask],
            pos_weight=positive_weight(tensors["hazard"][safe_mask]),
        )
        loss = hazard_loss + 0.5 * current_loss + 0.25 * next_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, {
        "prop_mean": prop_mean, "prop_std": prop_std,
        "action_mean": action_mean, "action_std": action_std,
        "hist_mean": hist_mean, "hist_std": hist_std,
    }


@torch.no_grad()
def predict(model: RecoverabilityHazardStudent, stats: dict[str, np.ndarray],
            data: dict[str, np.ndarray], idx: np.ndarray, device: str,
            policy_conditioning: str) -> dict[str, np.ndarray]:
    output = model(
        torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device),
        torch.as_tensor((data["proprio"][idx] - stats["prop_mean"])
                        / stats["prop_std"], device=device),
        torch.as_tensor((data["action_summary"][idx] - stats["action_mean"])
                        / stats["action_std"], device=device),
        torch.as_tensor((data["history"][idx] - stats["hist_mean"])
                        / stats["hist_std"], device=device),
        torch.as_tensor(data["elapsed_context"][idx], device=device),
        torch.as_tensor(data["language_hash"][idx], device=device),
        policy_index=(torch.as_tensor(data["policy_index"][idx], dtype=torch.long,
                                     device=device)
                      if policy_conditioning == "id" else None),
    )
    return {name: output[name].cpu().numpy() for name in (
        "current_recoverable_logit", "next_recoverable_logit", "loss_hazard_logit"
    )}


def task_bootstrap(labels: np.ndarray, probability: np.ndarray, tasks: np.ndarray,
                   *, seed: int, samples: int) -> dict[str, float]:
    unique = sorted(set(tasks.tolist()))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(tasks == task) for task in chosen])
        value = auc(labels[idx], probability[idx])
        if math.isfinite(value):
            values.append(value)
    return {"mean": float(np.mean(values)), "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--r8a0-audit", type=Path, required=True)
    parser.add_argument("--r8a1-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-seed", type=int, default=2026081207)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--policy-conditioning", choices=("none", "id"), default="id")
    parser.add_argument("--policy-id", choices=("pi05_libero", "pi0fast_libero"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    a0 = json.loads(args.r8a0_audit.read_text())
    a1 = json.loads(args.r8a1_audit.read_text())
    protocol = json.loads(args.protocol.read_text())
    dataset_hash = sha256(args.dataset)
    if report.get("dataset_sha256") != dataset_hash:
        raise ValueError("dataset/report hash mismatch")
    if a0.get("status") != "PASS" or a0.get("dataset_sha256") != dataset_hash:
        raise ValueError("R8-A0 is not PASS on this dataset")
    if a1.get("status") != "PASS":
        raise ValueError("R8-A1 label stability gate is not PASS")
    if protocol.get("status") != "frozen_before_r8a1_outcome":
        raise ValueError("R8-B protocol is not frozen")
    if protocol["input_contract"]["dataset_sha256"] != dataset_hash:
        raise ValueError("R8-B protocol is bound to a different dataset")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    data = build_transitions(source)
    if args.policy_id:
        keep = data["policy_id"] == args.policy_id
        data = {key: value[keep] for key, value in data.items()}
        if args.policy_conditioning != "none":
            raise ValueError("per-VLA comparator must disable policy ID conditioning")
    natural = data["cohort_role"] == "natural"
    folds = task_folds(data["task_id"], data["suite"], count=args.folds,
                       seed=args.fold_seed)
    all_tasks = set(data["task_id"].tolist())
    prediction_rows = []
    fold_reports = []
    for fold, validation_tasks in enumerate(folds):
        train_tasks = all_tasks - validation_tasks
        cal_tasks = calibration_tasks(train_tasks, data["task_id"], data["suite"],
                                      fold=fold, seed=args.fold_seed)
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.flatnonzero(np.isin(data["task_id"], list(fit_tasks)))
        cal_idx = np.flatnonzero(np.isin(data["task_id"], list(cal_tasks)) & natural)
        val_idx = np.flatnonzero(np.isin(data["task_id"], list(validation_tasks)) & natural)
        for name, idx in (("fit", fit_idx), ("calibration", cal_idx), ("validation", val_idx)):
            safe = data["current_recoverable"][idx].astype(bool)
            if not len(idx) or len(np.unique(data["loss_hazard"][idx][safe])) != 2:
                raise ValueError(f"fold {fold} {name} lacks both conditional hazard labels")
        member_cal: dict[str, list[np.ndarray]] = defaultdict(list)
        member_val: dict[str, list[np.ndarray]] = defaultdict(list)
        for member in range(args.members):
            member_seed = args.seed + fold * 1009 + member * 7919
            model, stats = train_member(
                data, fit_idx, seed=member_seed, epochs=args.epochs,
                device=args.device, policy_conditioning=args.policy_conditioning,
            )
            for key, value in predict(model, stats, data, cal_idx, args.device,
                                      args.policy_conditioning).items():
                member_cal[key].append(value)
            for key, value in predict(model, stats, data, val_idx, args.device,
                                      args.policy_conditioning).items():
                member_val[key].append(value)
        cal_logits = {key: np.mean(value, axis=0) for key, value in member_cal.items()}
        val_logits = {key: np.mean(value, axis=0) for key, value in member_val.items()}
        safe_cal = data["current_recoverable"][cal_idx].astype(bool)
        calibrations = {}
        probabilities = {}
        for target, logit_key, mask in (
            ("current_recoverable", "current_recoverable_logit", np.ones(len(cal_idx), bool)),
            ("next_recoverable", "next_recoverable_logit", np.ones(len(cal_idx), bool)),
            ("loss_hazard", "loss_hazard_logit", safe_cal),
        ):
            temperature, bias = fit_platt(cal_logits[logit_key][mask], data[target][cal_idx][mask])
            calibrations[target] = {"temperature": temperature, "bias": bias}
            probabilities[target] = 1.0 / (1.0 + np.exp(
                -(val_logits[logit_key] / temperature + bias)))
        for position, index in enumerate(val_idx):
            prediction_rows.append({
                "index": int(index), "fold": fold,
                "current_probability": float(probabilities["current_recoverable"][position]),
                "next_probability": float(probabilities["next_recoverable"][position]),
                "hazard_probability": float(probabilities["loss_hazard"][position]),
            })
        safe_val = data["current_recoverable"][val_idx].astype(bool)
        fold_reports.append({
            "fold": fold, "fit_tasks": sorted(fit_tasks),
            "calibration_tasks": sorted(cal_tasks), "validation_tasks": sorted(validation_tasks),
            "fit_rows": int(len(fit_idx)), "calibration_rows": int(len(cal_idx)),
            "validation_rows": int(len(val_idx)), "calibration": calibrations,
            "hazard_metrics": metrics(data["loss_hazard"][val_idx][safe_val],
                                      probabilities["loss_hazard"][safe_val]),
        })
    prediction_rows.sort(key=lambda row: row["index"])
    val_idx = np.asarray([row["index"] for row in prediction_rows], dtype=np.int64)
    current_probability = np.asarray([row["current_probability"] for row in prediction_rows])
    next_probability = np.asarray([row["next_probability"] for row in prediction_rows])
    hazard_probability = np.asarray([row["hazard_probability"] for row in prediction_rows])
    safe = data["current_recoverable"][val_idx].astype(bool)
    hazard_labels = data["loss_hazard"][val_idx][safe]
    hazard_scores = hazard_probability[safe]
    hazard_tasks = data["task_id"][val_idx][safe]
    overall_hazard = metrics(hazard_labels, hazard_scores)
    current_metrics = metrics(data["current_recoverable"][val_idx], current_probability)

    def breakdown(field: str) -> dict[str, dict[str, float | int]]:
        values = data[field][val_idx][safe]
        return {str(value): metrics(hazard_labels[values == value], hazard_scores[values == value])
                for value in sorted(set(values.tolist()))}

    by_policy = breakdown("policy_id")
    by_suite = breakdown("suite")
    by_horizon = breakdown("elapsed_source_steps")
    bootstrap = task_bootstrap(hazard_labels, hazard_scores, hazard_tasks,
                               seed=args.seed, samples=args.bootstrap_samples)
    frozen_gate = protocol["gates_per_seed"]
    gate = {
        "hazard_auroc": overall_hazard["auroc"] >= frozen_gate["hazard_auroc_min"],
        "hazard_bootstrap_lower": bootstrap["lower_95"] >= frozen_gate["hazard_task_bootstrap_auroc_lower95_min"],
        "hazard_ap_gain": overall_hazard["ap_above_prevalence"] >= frozen_gate["hazard_ap_above_prevalence_min"],
        "hazard_ece": overall_hazard["ece_10_equal_width"] <= frozen_gate["hazard_ece_max"],
        "current_recoverable_auroc": current_metrics["auroc"] >= frozen_gate["current_recoverable_auroc_min"],
        "each_policy_hazard_auroc": all(row["auroc"] > frozen_gate["each_policy_hazard_auroc_strictly_above"] for row in by_policy.values()),
        "each_horizon_hazard_auroc": all(row["auroc"] > frozen_gate["each_horizon_hazard_auroc_strictly_above"] for row in by_horizon.values()),
        "each_suite_hazard_auroc": all(row["auroc"] > frozen_gate["each_suite_hazard_auroc_strictly_above"] for row in by_suite.values()),
    }
    result = {
        "schema_version": "rase-r8b-local-recoverability-hazard-probe/v1",
        "status": "PASS" if all(gate.values()) else "FAIL",
        "scientific_scope": "development task-held-out no-world-model local hazard probe",
        "seed": args.seed, "members": args.members, "folds": args.folds,
        "policy_conditioning": args.policy_conditioning, "policy_filter": args.policy_id,
        "dataset_sha256": dataset_hash, "dataset_report_sha256": sha256(args.dataset_report),
        "r8a0_audit_sha256": sha256(args.r8a0_audit),
        "r8a1_audit_sha256": sha256(args.r8a1_audit),
        "protocol_sha256": sha256(args.protocol),
        "transition_rows": int(len(data["task_id"])),
        "natural_validation_rows": int(len(val_idx)),
        "conditional_hazard_rows": int(safe.sum()),
        "hazard_metrics": overall_hazard,
        "current_recoverable_metrics": current_metrics,
        "hazard_task_bootstrap_auroc": bootstrap,
        "hazard_metrics_by_policy": by_policy,
        "hazard_metrics_by_horizon": by_horizon,
        "hazard_metrics_by_suite": by_suite,
        "gate": gate, "fold_reports": fold_reports,
        "forbidden_and_absent": protocol["forbidden_features"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        state_key=data["state_key"][val_idx], task_id=data["task_id"][val_idx],
        suite=data["suite"][val_idx], policy_id=data["policy_id"][val_idx],
        group_id=data["group_id"][val_idx],
        elapsed_source_steps=data["elapsed_source_steps"][val_idx],
        current_recoverable=data["current_recoverable"][val_idx],
        next_recoverable=data["next_recoverable"][val_idx],
        loss_hazard=data["loss_hazard"][val_idx],
        current_probability=current_probability.astype(np.float32),
        next_probability=next_probability.astype(np.float32),
        hazard_probability=hazard_probability.astype(np.float32),
    )
    print(json.dumps({"status": result["status"], "hazard": overall_hazard,
                      "current": current_metrics, "bootstrap": bootstrap,
                      "gate": gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

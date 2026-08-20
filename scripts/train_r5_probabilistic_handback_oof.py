#!/usr/bin/env python3
"""Nested task-OOF training for the R5 probabilistic handback controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.canonical_action import summary_from_chunk  # noqa: E402
from rase.risk.probabilistic_handback_student import ProbabilisticHandbackStudent  # noqa: E402
from rase.risk.probabilistic_losses import beta_binomial_nll, binomial_nll_from_logits, quantile_pinball_loss  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402
from rase.risk.vla_action_adapters import create_vla_adapter  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol_summary(
    protocol: dict[str, Any], *, dataset: Path,
    require_opportunity_ready: bool = False,
) -> None:
    """Fail closed before fitting when a frozen data or opportunity gate is shut."""
    if protocol.get("protocol_gate_status") != "ready":
        raise ValueError(
            f"protocol gate is closed: {protocol.get('protocol_gate_reasons')}"
        )
    if Path(str(protocol["source_dataset"])).resolve() != dataset.resolve():
        raise ValueError("protocol summary refers to a different dataset path")
    if protocol.get("source_dataset_sha256") != sha256(dataset):
        raise ValueError("dataset changed after protocol summary was written")
    if require_opportunity_ready and protocol.get("probability_opportunity_gate_status") != "ready":
        raise ValueError(
            "probability opportunity gate is closed: "
            f"{protocol.get('probability_opportunity_gate_reasons')}"
        )


def task_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> list[dict[str, Any]]:
    tasks = sorted({str(row["task_id"]) for row in rows})
    if not 2 <= folds <= len(tasks):
        raise ValueError(f"folds must be in [2,{len(tasks)}]")
    rng = random.Random(seed)
    rng.shuffle(tasks)
    fold_tasks = [set(tasks[index::folds]) for index in range(folds)]
    result = []
    for index, validation_tasks in enumerate(fold_tasks):
        result.append({
            "fold": index,
            "validation_tasks": sorted(validation_tasks),
            "train": [row for row in rows if str(row["task_id"]) not in validation_tasks],
            "validation": [row for row in rows if str(row["task_id"]) in validation_tasks],
        })
    return result


def inner_split(rows: list[dict[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    tasks = sorted({str(row["task_id"]) for row in rows})
    calibration_count = max(1, min(2, len(tasks) // 3))
    offset = (fold * calibration_count) % len(tasks)
    rotated = tasks[offset:] + tasks[:offset]
    calibration_tasks = set(rotated[:calibration_count])
    fit = [row for row in rows if str(row["task_id"]) not in calibration_tasks]
    calibration = [row for row in rows if str(row["task_id"]) in calibration_tasks]
    if not fit or not calibration:
        raise ValueError("inner split produced an empty partition")
    return fit, calibration, sorted(calibration_tasks)


def feature_map(rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    return {id(row): np.asarray(row["latent"], np.float32) for row in rows}


def normalize(
    rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]], raw: dict[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    fit = np.stack([raw[id(row)] for row in fit_rows])
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    constant = std < 1e-5
    std = np.where(constant, 1.0, std)
    return {
        id(row): np.nan_to_num((raw[id(row)] - mean) / std).astype(np.float32)
        for row in rows
    }, {"fit_rows": len(fit_rows), "constant_dimensions": int(constant.sum())}


def build_batch(
    rows: list[dict[str, Any]], *, default_student_adapter: str,
    default_teacher_adapter: str, features: dict[int, np.ndarray],
    cost_scale: float, device: str,
) -> dict[str, torch.Tensor]:
    adapters: dict[str, Any] = {}

    def adapter(name: str) -> Any:
        if name not in adapters:
            adapters[name] = create_vla_adapter(name)
        return adapters[name]

    image = np.stack([features[id(row)] for row in rows])
    proprio = np.stack([np.asarray(row["proprio"], np.float32) for row in rows])
    student = np.stack([
        summary_from_chunk(adapter(str(row.get("student_vla_adapter", default_student_adapter))).to_canonical(
            np.asarray(row["student_action_chunk"], np.float32)
        )).numpy()
        for row in rows
    ])
    oft = np.stack([
        summary_from_chunk(adapter(str(row.get("teacher_vla_adapter", default_teacher_adapter))).to_canonical(
            np.asarray(row["oft_action"], np.float32).reshape(1, -1)
        )).numpy()
        for row in rows
    ])
    history = np.zeros((len(rows), 4, 6), np.float32)
    for index, row in enumerate(rows):
        elapsed = float(row["elapsed_oft_steps"])
        for step in range(4):
            history[index, step, 0] = max(0.0, elapsed - (3 - step) * 8.0) / 128.0
            history[index, step, 1:] = proprio[index, :5]
    successes = np.asarray([float(row["handback_success_count"]) for row in rows], np.float32)
    trials = np.asarray([float(row["handback_repeats"]) for row in rows], np.float32)
    persistent = np.asarray([float(bool(row["success_if_continue_oft"])) for row in rows], np.float32)
    source_mask = np.asarray([float(int(row["elapsed_oft_steps"]) == 0) for row in rows], np.float32)
    costs = np.asarray([
        float(row["remaining_teacher_steps"]) / max(cost_scale, 1e-8) for row in rows
    ], np.float32)
    values = {
        "image": image, "proprio": proprio, "student_action": student,
        "oft_action": oft, "history": history, "successes": successes,
        "trials": trials, "persistent": persistent, "source_mask": source_mask,
        "cost": costs,
    }
    return {key: torch.as_tensor(value, device=device) for key, value in values.items()}


def make_model(feature_dim: int, args: argparse.Namespace) -> ProbabilisticHandbackStudent:
    encoder = TinyUniversalStateEncoder(
        image_size=128, proprio_dim=8, text_embed_dim=0,
        hidden_dim=args.hidden_dim, output_dim=128,
        input_mode="latent", latent_dim=feature_dim,
    )
    return ProbabilisticHandbackStudent(
        encoder, action_dim=args.action_dim, fused_dim=args.hidden_dim,
        head_hidden=args.hidden_dim, n_cost_quantiles=3,
    ).to(args.device)


def slice_batch(batch: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    tensor_indices = torch.as_tensor(indices, device=next(iter(batch.values())).device)
    return {key: value.index_select(0, tensor_indices) for key, value in batch.items()}


def task_bootstrap(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for task in rng.choice(tasks, len(tasks), replace=True):
        output.extend(by_task[str(task)])
    return output


def train_member(
    model: ProbabilisticHandbackStudent, batch: dict[str, torch.Tensor], *,
    epochs: int, batch_size: int, learning_rate: float, seed: int,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=batch["image"].device)
    n = len(batch["successes"])
    persistent_supported = bool(batch["persistent"].min() < batch["persistent"].max())
    source_all = batch["source_mask"] > 0.5
    source_success_all = batch["successes"][source_all].sum()
    source_failure_all = (batch["trials"][source_all] - batch["successes"][source_all]).sum()
    source_supported = bool(source_success_all > 0 and source_failure_all > 0)
    for _ in range(epochs):
        chunks = np.array_split(rng.permutation(n), max(1, math.ceil(n / batch_size)))
        for indices in chunks:
            minibatch = slice_batch(batch, indices)
            model.train()
            optimizer.zero_grad()
            out = model(
                minibatch["image"], minibatch["proprio"],
                minibatch["student_action"], minibatch["oft_action"], minibatch["history"],
            )
            handback_loss = beta_binomial_nll(
                out["handback_alpha_raw"], out["handback_beta_raw"],
                minibatch["successes"], minibatch["trials"],
            )
            # A head with one-class support is present in the architecture but
            # is not trained or interpreted.  Optimizing BCE on all-positive
            # A16 persistent labels would create a misleading "validated" head.
            if persistent_supported:
                persistent_loss = F.binary_cross_entropy_with_logits(
                    out["persistent_logit"], minibatch["persistent"]
                )
            else:
                persistent_loss = out["persistent_logit"].sum() * 0.0
            source_indices = torch.nonzero(minibatch["source_mask"] > 0.5, as_tuple=False).flatten()
            source_success = minibatch["successes"].index_select(0, source_indices)
            source_trials = minibatch["trials"].index_select(0, source_indices)
            source_failures = source_trials - source_success
            if (
                len(source_indices) and source_supported
            ):
                source_loss = binomial_nll_from_logits(
                    out["source_risk_logit"].index_select(0, source_indices),
                    source_failures,
                    source_trials,
                )
            else:
                source_loss = out["source_risk_logit"].sum() * 0.0
            cost_loss = quantile_pinball_loss(
                out["remaining_cost_quantiles"], minibatch["cost"], quantiles
            )
            quantile_crossing = F.relu(
                out["remaining_cost_quantiles"][:, :-1]
                - out["remaining_cost_quantiles"][:, 1:]
            ).mean()
            loss = handback_loss + 0.25 * persistent_loss + 0.25 * source_loss + 0.5 * cost_loss + 0.1 * quantile_crossing
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def predict(models: list[ProbabilisticHandbackStudent], batch: dict[str, torch.Tensor], z: float) -> dict[str, np.ndarray]:
    means, variances, persistent, source, costs = [], [], [], [], []
    for model in models:
        model.eval()
        out = model(batch["image"], batch["proprio"], batch["student_action"], batch["oft_action"], batch["history"])
        alpha, beta = out["handback_alpha"], out["handback_beta"]
        mean = out["handback_mean"]
        variance = alpha * beta / ((alpha + beta).square() * (alpha + beta + 1.0))
        means.append(mean)
        variances.append(variance)
        persistent.append(torch.sigmoid(out["persistent_logit"]))
        source.append(torch.sigmoid(out["source_risk_logit"]))
        costs.append(out["remaining_cost_quantiles"])
    mean_stack = torch.stack(means)
    total_variance = torch.stack(variances).mean(0) + mean_stack.var(0, unbiased=False)
    mean = mean_stack.mean(0)
    return {
        "handback_mean": mean.cpu().numpy(),
        "handback_lcb": torch.clamp(mean - z * torch.sqrt(total_variance), 0.0, 1.0).cpu().numpy(),
        "epistemic_std": mean_stack.std(0, unbiased=False).cpu().numpy(),
        "aleatoric_std": torch.sqrt(torch.stack(variances).mean(0)).cpu().numpy(),
        "persistent": torch.stack(persistent).mean(0).cpu().numpy(),
        "source_risk": torch.stack(source).mean(0).cpu().numpy(),
        "cost_quantiles": torch.stack(costs).mean(0).cpu().numpy(),
    }


def controller_records(
    rows: list[dict[str, Any]], lcbs: np.ndarray, threshold: float, dwell: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, lcb in zip(rows, lcbs):
        grouped[str(row["state_key"])].append((row, float(lcb)))
    records = []
    for state, items in grouped.items():
        items.sort(key=lambda item: int(item[0]["elapsed_oft_steps"]))
        streak = 0
        chosen: dict[str, Any] | None = None
        for row, lcb in items:
            streak = streak + 1 if lcb >= threshold else 0
            if streak >= dwell:
                chosen = row
                break
        reference = items[0][0]
        persistent_success = float(bool(reference["success_if_continue_oft"]))
        persistent_cost = float(reference["persistent_executed_oft_steps"])
        if chosen is None:
            empirical_success = persistent_success
            conservative_success = persistent_success
            cost = persistent_cost
            handback = False
            elapsed = None
        else:
            empirical_success = float(chosen["handback_success_probability"])
            conservative_success = float(bool(chosen["success_if_handback_now"]))
            cost = min(float(chosen["elapsed_oft_steps"]), persistent_cost)
            handback = True
            elapsed = int(chosen["elapsed_oft_steps"])
        records.append({
            "state_key": state, "task_id": str(reference["task_id"]),
            "persistent_success": persistent_success,
            "empirical_policy_success": empirical_success,
            "conservative_policy_success": conservative_success,
            "expected_harm": float(handback) * persistent_success * (1.0 - empirical_success),
            "conservative_harm": bool(handback and persistent_success and not conservative_success),
            "persistent_cost": persistent_cost, "policy_cost": cost,
            "handback": handback, "elapsed_oft_steps": elapsed,
        })
    return records


def controller_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(records))
    persistent_successes = sum(row["persistent_success"] for row in records)
    persistent_cost = sum(row["persistent_cost"] for row in records)
    policy_cost = sum(row["policy_cost"] for row in records)
    return {
        "n_states": len(records),
        "persistent_success_rate": persistent_successes / n,
        "empirical_policy_success_rate": sum(row["empirical_policy_success"] for row in records) / n,
        "empirical_success_gap": (sum(row["empirical_policy_success"] for row in records) - persistent_successes) / n,
        "conservative_policy_success_rate": sum(row["conservative_policy_success"] for row in records) / n,
        "conservative_success_gap": (sum(row["conservative_policy_success"] for row in records) - persistent_successes) / n,
        "conditional_expected_false_handback": sum(row["expected_harm"] for row in records) / max(1.0, persistent_successes),
        "conditional_conservative_false_handback": sum(row["conservative_harm"] for row in records) / max(1.0, persistent_successes),
        "handback_rate": sum(row["handback"] for row in records) / n,
        "oft_savings": (persistent_cost - policy_cost) / max(1.0, persistent_cost),
        "persistent_total_oft_steps": persistent_cost,
        "policy_total_oft_steps": policy_cost,
    }


def task_cluster_bootstrap(
    records: list[dict[str, Any]], *, seed: int, replicates: int = 5000,
) -> dict[str, dict[str, float]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    keys = (
        "empirical_success_gap",
        "conditional_expected_false_handback",
        "oft_savings",
    )
    samples = {key: [] for key in keys}
    for _ in range(replicates):
        draw = rng.choice(tasks, len(tasks), replace=True)
        metrics = controller_metrics([
            row for task in draw for row in by_task[str(task)]
        ])
        for key in keys:
            samples[key].append(float(metrics[key]))
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    }


def choose_threshold(rows: list[dict[str, Any]], lcbs: np.ndarray, dwell: int) -> tuple[float, dict[str, float]]:
    candidates = [float(np.nextafter(np.max(lcbs), np.inf))]
    candidates.extend(sorted({float(value) for value in lcbs}, reverse=True))
    feasible = []
    for threshold in candidates:
        metrics = controller_metrics(controller_records(rows, lcbs, threshold, dwell))
        if metrics["empirical_success_gap"] >= -0.05 and metrics["conditional_expected_false_handback"] <= 0.05:
            feasible.append((metrics["oft_savings"], -metrics["handback_rate"], threshold, metrics))
    if not feasible:
        threshold = candidates[0]
        return threshold, controller_metrics(controller_records(rows, lcbs, threshold, dwell))
    best = max(feasible, key=lambda item: item[:2])
    return float(best[2]), best[3]


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = np.asarray(y, np.int64), np.asarray(score, np.float64)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and score[order[end]] == score[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def target_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    persistent = [int(bool(row["success_if_continue_oft"])) for row in rows]
    h0 = [row for row in rows if int(row["elapsed_oft_steps"]) == 0]
    return {
        "persistent_positive_rows": sum(persistent),
        "persistent_negative_rows": len(persistent) - sum(persistent),
        "persistent_target_degenerate": len(set(persistent)) < 2,
        "source_h0_states": len(h0),
        "source_success_trials": sum(int(row["handback_success_count"]) for row in h0),
        "source_failure_trials": sum(int(row["handback_repeats"]) - int(row["handback_success_count"]) for row in h0),
        "handback_success_trials": sum(int(row["handback_success_count"]) for row in rows),
        "handback_failure_trials": sum(int(row["handback_repeats"]) - int(row["handback_success_count"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--protocol-summary", type=Path,
        help="Optional manifest-aware A16 summary; training aborts unless its protocol gate is ready.",
    )
    parser.add_argument(
        "--require-opportunity-ready", action="store_true",
        help="Also abort unless the frozen probability-opportunity gate is ready.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--student-vla-adapter", default="smolvla")
    parser.add_argument("--teacher-vla-adapter", default="oft")
    parser.add_argument("--dwell", type=int, default=2)
    parser.add_argument("--lcb-z", type=float, default=1.6448536269514722)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--split-seed", type=int, default=20260820,
        help="Frozen task-fold assignment seed; keep fixed across training seeds.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.protocol_summary:
        protocol = json.loads(args.protocol_summary.read_text())
        validate_protocol_summary(
            protocol, dataset=args.dataset,
            require_opportunity_ready=args.require_opportunity_ready,
        )
    elif args.require_opportunity_ready:
        raise ValueError("--require-opportunity-ready requires --protocol-summary")
    rows = read_jsonl(args.dataset)
    if not rows:
        raise SystemExit("empty probabilistic boundary dataset")
    required = {"handback_success_count", "handback_repeats", "handback_success_probability"}
    if any(required - set(row) for row in rows):
        raise ValueError("dataset is not a K-repeat probabilistic boundary dataset")
    raw = feature_map(rows)
    # Resolve adapters before training so an unsupported VLA fails before any
    # fold is fit. Rows in a future shared dataset may override these defaults.
    create_vla_adapter(args.student_vla_adapter)
    create_vla_adapter(args.teacher_vla_adapter)
    row_index = {id(row): index for index, row in enumerate(rows)}
    oof = {key: np.full(len(rows), np.nan, np.float32) for key in (
        "mean", "lcb", "epistemic_std", "aleatoric_std", "persistent", "source_risk", "cost_median", "threshold"
    )}
    fold_reports = []
    model_parameter_count: int | None = None
    for fold_data in task_folds(rows, args.folds, args.split_seed):
        fold = int(fold_data["fold"])
        fit_rows, calibration_rows, calibration_tasks = inner_split(fold_data["train"], fold)
        normalized, normalization = normalize(rows, fit_rows, raw)
        cost_scale = max(float(row["remaining_teacher_steps"]) for row in fit_rows)
        models = []
        for member in range(args.ensemble_size):
            member_seed = args.seed + fold * 1000 + member
            torch.manual_seed(member_seed); np.random.seed(member_seed); random.seed(member_seed)
            model = make_model(len(next(iter(raw.values()))), args)
            if model_parameter_count is None:
                model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
            boot_rows = task_bootstrap(fit_rows, member_seed)
            train_member(
                model,
                build_batch(
                    boot_rows,
                    default_student_adapter=args.student_vla_adapter,
                    default_teacher_adapter=args.teacher_vla_adapter,
                    features=normalized, cost_scale=cost_scale, device=args.device,
                ),
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=member_seed,
            )
            models.append(model)
        calibration_prediction = predict(
            models,
            build_batch(
                calibration_rows,
                default_student_adapter=args.student_vla_adapter,
                default_teacher_adapter=args.teacher_vla_adapter,
                features=normalized, cost_scale=cost_scale, device=args.device,
            ),
            args.lcb_z,
        )
        threshold, calibration_metrics = choose_threshold(
            calibration_rows, calibration_prediction["handback_lcb"], args.dwell
        )
        validation_rows = fold_data["validation"]
        validation_prediction = predict(
            models,
            build_batch(
                validation_rows,
                default_student_adapter=args.student_vla_adapter,
                default_teacher_adapter=args.teacher_vla_adapter,
                features=normalized, cost_scale=cost_scale, device=args.device,
            ),
            args.lcb_z,
        )
        validation_metrics = controller_metrics(
            controller_records(validation_rows, validation_prediction["handback_lcb"], threshold, args.dwell)
        )
        indices = [row_index[id(row)] for row in validation_rows]
        for key, prediction_key in (
            ("mean", "handback_mean"), ("lcb", "handback_lcb"),
            ("epistemic_std", "epistemic_std"), ("aleatoric_std", "aleatoric_std"),
            ("persistent", "persistent"), ("source_risk", "source_risk"),
        ):
            oof[key][indices] = validation_prediction[prediction_key]
        oof["cost_median"][indices] = validation_prediction["cost_quantiles"][:, 1] * cost_scale
        oof["threshold"][indices] = threshold
        fold_report = {
            "fold": fold,
            "fit_tasks": sorted({str(row["task_id"]) for row in fit_rows}),
            "calibration_tasks": calibration_tasks,
            "validation_tasks": fold_data["validation_tasks"],
            "n_fit_rows": len(fit_rows), "n_calibration_rows": len(calibration_rows),
            "n_validation_rows": len(validation_rows), "threshold": threshold,
            "calibration_metrics": calibration_metrics,
            "validation_metrics": validation_metrics,
            "normalization": normalization,
        }
        fold_reports.append(fold_report)
        print(json.dumps(fold_report), flush=True)
        del models
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("incomplete OOF predictions")
    all_records: list[dict[str, Any]] = []
    by_state_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state_indices[str(row["state_key"])].append(index)
    for indices in by_state_indices.values():
        indices.sort(key=lambda index: int(rows[index]["elapsed_oft_steps"]))
        all_records.extend(controller_records(
            [rows[index] for index in indices], oof["lcb"][indices], float(oof["threshold"][indices[0]]), args.dwell
        ))
    metrics = controller_metrics(all_records)
    cluster_intervals = task_cluster_bootstrap(all_records, seed=args.seed + 900_000)
    repeat_labels, repeat_scores = [], []
    for row, score in zip(rows, oof["mean"]):
        repeat_labels.extend([1] * int(row["handback_success_count"]))
        repeat_labels.extend([0] * (int(row["handback_repeats"]) - int(row["handback_success_count"])))
        repeat_scores.extend([float(score)] * int(row["handback_repeats"]))
    soft_targets = np.asarray([float(row["handback_success_probability"]) for row in rows])
    gates = {
        "success_within_5pp": bool(metrics["empirical_success_gap"] >= -0.05),
        "false_handback_at_most_5pct": bool(metrics["conditional_expected_false_handback"] <= 0.05),
        "oft_savings_at_least_20pct": bool(metrics["oft_savings"] >= 0.20),
    }
    report = {
        "schema_version": "rase-pre-c0-r5-probabilistic-oof/v1",
        "deployment_artifacts_written": False,
        "source": str(args.dataset.resolve()),
        "source_sha256": sha256(args.dataset),
        "protocol_summary": str(args.protocol_summary.resolve()) if args.protocol_summary else None,
        "protocol_summary_sha256": sha256(args.protocol_summary) if args.protocol_summary else None,
        "trainer_source_sha256": sha256(Path(__file__)),
        "model_source_sha256": sha256(ROOT / "rase/risk/probabilistic_handback_student.py"),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_rows": len(rows), "n_states": len(by_state_indices),
        "n_tasks": len({str(row["task_id"]) for row in rows}),
        "folds": args.folds, "ensemble_size": args.ensemble_size,
        "single_member_parameter_count": model_parameter_count,
        "student_vla_adapters": sorted({
            str(row.get("student_vla_adapter", args.student_vla_adapter)) for row in rows
        }),
        "teacher_vla_adapters": sorted({
            str(row.get("teacher_vla_adapter", args.teacher_vla_adapter)) for row in rows
        }),
        "dwell": args.dwell, "lcb_z": args.lcb_z,
        "target_audit": target_audit(rows),
        "repeat_level_auc_diagnostic": auc(np.asarray(repeat_labels), np.asarray(repeat_scores)),
        "soft_brier": float(np.mean((oof["mean"] - soft_targets) ** 2)),
        "oof_state_metrics": metrics,
        "task_cluster_bootstrap_95": cluster_intervals,
        "fold_reports": fold_reports,
        "gates": gates,
        "all_controller_gates_passed": all(gates.values()),
        "oof_predictions": {
            f"{row['state_key']}:{row['elapsed_oft_steps']}": {
                key: float(values[index]) for key, values in oof.items()
            }
            for index, row in enumerate(rows)
        },
        "state_records": all_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

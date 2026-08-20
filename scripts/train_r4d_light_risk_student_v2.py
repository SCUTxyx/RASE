#!/usr/bin/env python3
"""Leakage-safe R4-D LightRiskStudent nested task-OOF trainer.

This v2 protocol fixes four failure modes in the original experiment:
  * every outer fold and ensemble member is initialized independently;
  * cluster bootstrap samples complete tasks, not one row per task;
  * thresholds are selected and evaluated as one stopping decision per state;
  * V-JEPA evidence is matched exactly by (state_key, elapsed_oft_steps).

Feature modes:
  baseline        RASE 128-D boundary latent only.
  baseline_delta  baseline plus deterministic 32-D projections of the
                  Student delta, OFT delta, and their branch difference.

The script writes OOF evidence only. It deliberately does not emit a deployment
checkpoint; a final model may be fitted only after this protocol passes.
"""

from __future__ import annotations

import argparse
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

from scripts.train_r4_safe_handback_wm_ridge import (  # noqa: E402
    grouped_task_folds,
    inner_task_split,
    read_jsonl,
)
from rase.risk.canonical_action import summary_from_chunk  # noqa: E402
from rase.risk.light_risk_student import LightRiskStudent  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402
from rase.risk.vla_action_adapters import create_vla_adapter  # noqa: E402


def evidence_feature_map(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    evidence_path: Path | None,
    projection_dim: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build deterministic, label-free input features for every boundary row."""
    baseline = {id(r): np.asarray(r["latent"], np.float32) for r in rows}
    if mode == "baseline":
        return baseline, {"matched": 0, "distinct_actions": 0, "feature_dim": 128}
    if evidence_path is None:
        raise ValueError("--teacher-evidence is required for baseline_delta")

    evidence = read_jsonl(evidence_path)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_keys = 0
    for ev in evidence:
        if ev.get("elapsed_oft_steps") is None:
            continue
        key = (str(ev["state_key"]), int(ev["elapsed_oft_steps"]))
        duplicate_keys += int(key in by_key)
        by_key[key] = ev
    if not by_key:
        raise ValueError(
            "teacher evidence has no exact elapsed_oft_steps keys; regenerate with "
            "cache_r4d_teacher_evidence.py --boundary-dataset"
        )

    first = next(iter(by_key.values()))
    teacher_dim = len(first.get("student_delta") or [])
    if teacher_dim == 0:
        raise ValueError("teacher evidence contains no student_delta")
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0, 1.0 / math.sqrt(teacher_dim), size=(teacher_dim, projection_dim)
    ).astype(np.float32)

    result: dict[int, np.ndarray] = {}
    missing: list[tuple[str, int]] = []
    distinct = 0
    branch_norms: list[float] = []
    for row in rows:
        key = (str(row["state_key"]), int(row.get("elapsed_oft_steps", 0)))
        ev = by_key.get(key)
        if ev is None:
            missing.append(key)
            continue
        student = np.asarray(ev.get("student_delta"), np.float32)
        oft = np.asarray(ev.get("oft_delta"), np.float32)
        if student.shape != (teacher_dim,) or oft.shape != (teacher_dim,):
            raise ValueError(f"bad teacher delta shape for {key}: {student.shape}, {oft.shape}")
        difference = student - oft
        difference_norm = float(np.linalg.norm(difference))
        branch_norms.append(difference_norm)
        distinct += int(difference_norm > 1e-8)
        scalar = np.asarray(
            [
                np.linalg.norm(student),
                np.linalg.norm(oft),
                difference_norm,
                float(np.dot(student, oft) / max(np.linalg.norm(student) * np.linalg.norm(oft), 1e-8)),
                float(ev.get("student_oft_first_action_l2", 0.0)),
            ],
            np.float32,
        )
        result[id(row)] = np.concatenate(
            [baseline[id(row)], student @ projection, oft @ projection,
             difference @ projection, scalar]
        ).astype(np.float32)
    if missing:
        raise ValueError(f"missing exact teacher evidence for {len(missing)}/{len(rows)} rows")
    if distinct == 0:
        raise ValueError("all Student/OFT teacher deltas are identical; cache is not action-conditioned")
    return result, {
        "matched": len(result),
        "duplicate_keys": duplicate_keys,
        "distinct_actions": distinct,
        "teacher_dim": teacher_dim,
        "projection_dim": projection_dim,
        "feature_dim": len(next(iter(result.values()))),
        "branch_difference_norm_quantiles": np.quantile(
            np.asarray(branch_norms), [0.0, 0.5, 0.9, 1.0]
        ).tolist(),
    }


def normalize_features(
    all_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    raw: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    fit = np.stack([raw[id(r)] for r in fit_rows]).astype(np.float32)
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = {
        id(r): np.nan_to_num((raw[id(r)] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        .astype(np.float32)
        for r in all_rows
    }
    return normalized, {
        "fit_rows": len(fit_rows),
        "constant_dimensions": int(np.sum(std == 1.0)),
    }


def build_batch(
    rows: list[dict[str, Any]],
    *,
    adapter: Any,
    device: str,
    features: dict[int, np.ndarray],
    cost_scale: float,
) -> dict[str, torch.Tensor]:
    n = len(rows)
    image = np.stack([features[id(r)] for r in rows])
    proprio = np.stack([np.asarray(r["proprio"], np.float32) for r in rows])
    student_sums = np.stack([
        summary_from_chunk(adapter.to_canonical(
            np.asarray(r["student_action_chunk"], np.float32)
        )).numpy()
        for r in rows
    ])
    oft_sums = np.stack([
        summary_from_chunk(adapter.to_canonical(
            np.asarray(r["oft_action"], np.float32).reshape(1, -1)
        )).numpy()
        for r in rows
    ])

    history = np.zeros((n, 4, 6), np.float32)
    for i, row in enumerate(rows):
        elapsed = float(row.get("elapsed_oft_steps", 0.0))
        for j in range(4):
            history[i, j, 0] = max(0.0, elapsed - (3 - j) * 8.0) / 128.0
            history[i, j, 1:] = proprio[i, :5]

    success = np.asarray([
        int(bool(r.get("success_if_handback_now", False))) for r in rows
    ], np.float32)
    cost = np.asarray([
        float(r.get("remaining_teacher_steps", 0.0)) / max(cost_scale, 1e-9)
        for r in rows
    ], np.float32)
    return {
        "image": torch.as_tensor(image, device=device),
        "proprio": torch.as_tensor(proprio, device=device),
        "student_action": torch.as_tensor(student_sums, device=device),
        "oft_action": torch.as_tensor(oft_sums, device=device),
        "history": torch.as_tensor(history, device=device),
        "success_label": torch.as_tensor(success, device=device),
        "cost_label": torch.as_tensor(cost, device=device),
    }


def make_model(feature_dim: int, args: argparse.Namespace) -> LightRiskStudent:
    encoder = TinyUniversalStateEncoder(
        image_size=128,
        proprio_dim=8,
        text_embed_dim=0,
        hidden_dim=args.hidden_dim,
        output_dim=128,
        input_mode="latent",
        latent_dim=feature_dim,
    )
    return LightRiskStudent(
        encoder,
        proprio_dim=8,
        action_dim=args.action_dim,
        history_dim=64,
        fused_dim=args.hidden_dim,
        head_hidden=128,
        n_members=1,
        n_cost_quantiles=3,
    ).to(args.device)


def _slice(batch: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    tensor_idx = torch.as_tensor(idx, device=next(iter(batch.values())).device)
    return {key: value.index_select(0, tensor_idx) for key, value in batch.items()}


def train_member(
    model: LightRiskStudent,
    batch: dict[str, torch.Tensor],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    n = len(batch["success_label"])
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), max(1, math.ceil(n / batch_size))):
            minibatch = _slice(batch, idx)
            model.train()
            optimizer.zero_grad()
            out = model(
                minibatch["image"], minibatch["proprio"],
                minibatch["student_action"], minibatch["oft_action"],
                minibatch["history"],
            )
            success = out["student_success"].squeeze(0)
            success_loss = F.binary_cross_entropy(success, minibatch["success_label"])
            cost = out["remaining_cost"].squeeze(0)
            target_cost = minibatch["cost_label"].unsqueeze(-1).expand_as(cost)
            cost_loss = F.smooth_l1_loss(cost, target_cost)
            ood = out["unsafe_ood"].squeeze(0)
            ood_loss = F.binary_cross_entropy(ood, 1.0 - minibatch["success_label"])
            loss = success_loss + 0.5 * cost_loss + 0.25 * ood_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def predict(models: list[LightRiskStudent], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    successes, costs, oods = [], [], []
    for model in models:
        model.eval()
        out = model(
            batch["image"], batch["proprio"], batch["student_action"],
            batch["oft_action"], batch["history"],
        )
        successes.append(out["student_success"].squeeze(0))
        costs.append(out["remaining_cost"].squeeze(0).mean(dim=-1))
        oods.append(out["unsafe_ood"].squeeze(0))
    success_stack = torch.stack(successes)
    return {
        "success": success_stack.mean(dim=0).cpu().numpy(),
        "success_std": success_stack.std(dim=0, unbiased=False).cpu().numpy(),
        "cost": torch.stack(costs).mean(dim=0).cpu().numpy(),
        "ood": torch.stack(oods).mean(dim=0).cpu().numpy(),
    }


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tie-aware rank AUC."""
    y_true = np.asarray(y_true, np.int64)
    y_score = np.asarray(y_score, np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def state_policy_records(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[str(row["state_key"])].append((row, float(score)))
    records = []
    for state_key, items in grouped.items():
        items.sort(key=lambda pair: int(pair[0].get("elapsed_oft_steps", 0)))
        reference = items[0][0]
        persistent_success = bool(reference.get("success_if_continue_oft", True))
        persistent_cost = float(reference.get(
            "persistent_executed_oft_steps",
            max((r.get("elapsed_oft_steps", 0) for r, _ in items), default=0),
        ))
        chosen = next(((r, s) for r, s in items if s >= threshold), None)
        if chosen is None:
            policy_success = persistent_success
            executed_cost = persistent_cost
            handback = False
            elapsed = None
        else:
            row, _ = chosen
            policy_success = bool(row.get("success_if_handback_now", False))
            elapsed = float(row.get("elapsed_oft_steps", 0.0))
            executed_cost = min(elapsed, persistent_cost)
            handback = True
        records.append({
            "state_key": state_key,
            "task_id": str(reference.get("task_id")),
            "persistent_success": persistent_success,
            "policy_success": policy_success,
            "persistent_cost": persistent_cost,
            "executed_cost": executed_cost,
            "handback": handback,
            "successful_handback": bool(handback and policy_success),
            "harm": bool(handback and persistent_success and not policy_success),
            "elapsed_oft_steps": elapsed,
        })
    return records


def summarize_state_records(records: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(records), 1)
    persistent_successes = sum(r["persistent_success"] for r in records)
    persistent_cost = sum(r["persistent_cost"] for r in records)
    policy_cost = sum(r["executed_cost"] for r in records)
    harms = sum(r["harm"] for r in records)
    return {
        "n_states": len(records),
        "persistent_success_rate": sum(r["persistent_success"] for r in records) / n,
        "policy_success_rate": sum(r["policy_success"] for r in records) / n,
        "success_gap": (
            sum(r["policy_success"] for r in records)
            - sum(r["persistent_success"] for r in records)
        ) / n,
        "handback_rate": sum(r["handback"] for r in records) / n,
        "successful_handback_rate": sum(r["successful_handback"] for r in records) / n,
        "harm_rate": harms / n,
        "conditional_false_handback_rate": harms / max(persistent_successes, 1),
        "oft_savings": (persistent_cost - policy_cost) / max(persistent_cost, 1e-9),
        "persistent_total_oft_steps": persistent_cost,
        "policy_total_oft_steps": policy_cost,
    }


def choose_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    max_success_drop: float,
    max_false_handback: float,
) -> tuple[float, dict[str, float]]:
    candidates = [float(np.nextafter(np.max(scores), np.inf))]
    candidates.extend(sorted({float(x) for x in scores}, reverse=True))
    feasible: list[tuple[float, float, float, float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = summarize_state_records(state_policy_records(rows, scores, threshold))
        if (metrics["success_gap"] >= -max_success_drop and
                metrics["conditional_false_handback_rate"] <= max_false_handback):
            feasible.append((metrics["oft_savings"], metrics["policy_success_rate"],
                             -metrics["handback_rate"], threshold, metrics))
    if not feasible:
        threshold = candidates[0]
        return threshold, summarize_state_records(state_policy_records(rows, scores, threshold))
    best = max(feasible, key=lambda x: x[:3])
    return float(best[3]), best[4]


def task_bootstrap(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    sampled = rng.choice(tasks, len(tasks), replace=True)
    output: list[dict[str, Any]] = []
    for task in sampled:
        output.extend(by_task[str(task)])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=("baseline", "baseline_delta"), default="baseline")
    parser.add_argument("--teacher-evidence", type=Path, default=None)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--max-success-drop", type=float, default=0.05)
    parser.add_argument("--max-false-handback", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rows = read_jsonl(args.dataset)
    if not rows:
        raise SystemExit("empty dataset")
    raw_features, evidence_report = evidence_feature_map(
        rows, mode=args.feature_mode, evidence_path=args.teacher_evidence,
        projection_dim=args.projection_dim, seed=args.seed,
    )
    feature_dim = len(next(iter(raw_features.values())))
    folds = grouped_task_folds(rows, args.folds)
    adapter = create_vla_adapter("smolvla")
    index = {id(row): i for i, row in enumerate(rows)}
    oof_scores = np.full(len(rows), np.nan, np.float32)
    oof_std = np.full(len(rows), np.nan, np.float32)
    oof_threshold = np.full(len(rows), np.nan, np.float32)
    fold_reports: list[dict[str, Any]] = []

    for fold_number, fold in enumerate(folds):
        train_rows, val_rows = fold["train"], fold["val"]
        fit_rows, calibration_rows, calibration_tasks = inner_task_split(train_rows, fold_number)
        normalized, norm_report = normalize_features(rows, fit_rows, raw_features)
        cost_scale = max(float(r.get("remaining_teacher_steps", 0.0)) for r in fit_rows)
        models: list[LightRiskStudent] = []
        for member in range(args.ensemble_size):
            member_seed = args.seed + fold_number * 1000 + member
            torch.manual_seed(member_seed)
            np.random.seed(member_seed)
            random.seed(member_seed)
            model = make_model(feature_dim, args)
            bootstrap_rows = task_bootstrap(fit_rows, seed=member_seed)
            train_batch = build_batch(
                bootstrap_rows, adapter=adapter, device=args.device,
                features=normalized, cost_scale=cost_scale,
            )
            train_member(
                model, train_batch, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, seed=member_seed,
            )
            models.append(model)

        calibration_batch = build_batch(
            calibration_rows, adapter=adapter, device=args.device,
            features=normalized, cost_scale=cost_scale,
        )
        calibration_scores = predict(models, calibration_batch)["success"]
        threshold, calibration_metrics = choose_threshold(
            calibration_rows, calibration_scores,
            max_success_drop=args.max_success_drop,
            max_false_handback=args.max_false_handback,
        )
        val_batch = build_batch(
            val_rows, adapter=adapter, device=args.device,
            features=normalized, cost_scale=cost_scale,
        )
        val_pred = predict(models, val_batch)
        val_indices = [index[id(row)] for row in val_rows]
        oof_scores[val_indices] = val_pred["success"]
        oof_std[val_indices] = val_pred["success_std"]
        oof_threshold[val_indices] = threshold
        labels = np.asarray([int(bool(r.get("success_if_handback_now", False))) for r in val_rows])
        val_state_metrics = summarize_state_records(
            state_policy_records(val_rows, val_pred["success"], threshold)
        )
        fold_report = {
            "fold": fold_number,
            "train_tasks": sorted({str(r["task_id"]) for r in fit_rows}),
            "calibration_tasks": sorted(calibration_tasks),
            "validation_tasks": sorted({str(r["task_id"]) for r in val_rows}),
            "n_train_rows": len(fit_rows),
            "n_calibration_rows": len(calibration_rows),
            "n_validation_rows": len(val_rows),
            "threshold": threshold,
            "calibration_metrics": calibration_metrics,
            "validation_auc": roc_auc_score_np(labels, val_pred["success"]),
            "validation_state_metrics": val_state_metrics,
            "normalization": norm_report,
        }
        fold_reports.append(fold_report)
        print(json.dumps(fold_report), flush=True)
        del models
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if np.isnan(oof_scores).any() or np.isnan(oof_threshold).any():
        raise RuntimeError("OOF predictions are incomplete")
    labels = np.asarray([int(bool(r.get("success_if_handback_now", False))) for r in rows])

    # Apply the threshold learned for each row's outer fold, then retain the
    # earliest accepted boundary per state.
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row["state_key"])].append(i)
    records = []
    for indices in grouped.values():
        indices.sort(key=lambda i: int(rows[i].get("elapsed_oft_steps", 0)))
        accepted = [i for i in indices if oof_scores[i] >= oof_threshold[i]]
        chosen_threshold = float(oof_threshold[indices[0]])
        state_rows = [rows[i] for i in indices]
        state_scores = np.asarray([oof_scores[i] for i in indices])
        records.extend(state_policy_records(state_rows, state_scores, chosen_threshold))
    state_metrics = summarize_state_records(records)

    report = {
        "schema_version": "rase-pre-c0-r4d-lightriskstudent/v2-leakage-safe",
        "feature_mode": args.feature_mode,
        "deployment_artifacts_written": False,
        "n_rows": len(rows),
        "n_states": len({str(r["state_key"]) for r in rows}),
        "n_tasks": len({str(r["task_id"]) for r in rows}),
        "feature_dim": feature_dim,
        "ensemble_size": args.ensemble_size,
        "oof_row_auc": roc_auc_score_np(labels, oof_scores),
        "oof_state_metrics": state_metrics,
        "fold_reports": fold_reports,
        "teacher_evidence_audit": evidence_report,
        "per_fold_thresholds": [r["threshold"] for r in fold_reports],
        "oof_predictions": {
            f"{r['state_key']}:{r['elapsed_oft_steps']}": float(score)
            for r, score in zip(rows, oof_scores)
        },
        "oof_prediction_std": {
            f"{r['state_key']}:{r['elapsed_oft_steps']}": float(score)
            for r, score in zip(rows, oof_std)
        },
        "oof_thresholds": {
            f"{r['state_key']}:{r['elapsed_oft_steps']}": float(threshold)
            for r, threshold in zip(rows, oof_threshold)
        },
        "gates": {
            "auc_at_least_0_70": bool(roc_auc_score_np(labels, oof_scores) >= 0.70),
            "success_within_5pp": bool(state_metrics["success_gap"] >= -0.05),
            "false_handback_at_most_5pct": bool(
                state_metrics["conditional_false_handback_rate"] <= 0.05
            ),
            "oft_savings_at_least_20pct": bool(state_metrics["oft_savings"] >= 0.20),
        },
        "source": str(args.dataset.resolve()),
        "teacher_evidence": str(args.teacher_evidence.resolve()) if args.teacher_evidence else None,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

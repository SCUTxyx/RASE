#!/usr/bin/env python3
"""Task-held-out low-capacity information gate for R10-B case/control data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool); pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg): return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def summarize_features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image = data["image_history"].astype(np.float32) / 255.0
    # Retain the last four frames while actions/proprio use the full eight-step
    # causal window.
    image = image[:, -4:]
    image_features = np.c_[
        image.mean(axis=(2, 3, 4, 5)), image.std(axis=(2, 3, 4, 5)),
        np.abs(np.diff(image, axis=1)).mean(axis=(2, 3, 4, 5)),
    ]
    proprio = data["proprio_history"].astype(np.float32)
    delta = data["proprio_delta_history"].astype(np.float32)
    accel = data["proprio_accel_history"].astype(np.float32)
    temporal = np.c_[proprio[:, -1], delta[:, 1:].mean(1), delta[:, 1:].std(1),
                     np.abs(delta[:, 1:]).max(1), accel[:, 2:].mean(1),
                     accel[:, 2:].std(1)]
    action = data["action_history"].astype(np.float32)
    action_delta = data["action_delta_history"].astype(np.float32)
    action_features = np.c_[action[:, -1], action.mean(1), action.std(1),
                            np.abs(action_delta[:, 1:]).mean(1),
                            np.abs(action_delta[:, 1:]).max(1)]
    semantic = data["language_hash"].astype(np.float32)
    return {
        "image_sequence": image_features,
        "temporal_state": temporal,
        "action_history": action_features,
        "semantic": semantic,
        "temporal_plus_action": np.c_[temporal, action_features],
        "all_causal": np.c_[image_features, temporal, action_features, semantic],
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = x.mean(0), x.std(0); std[std < 1e-6] = 1.0
    z = (x - mean) / std
    design = np.c_[z, np.ones(len(z))]
    rng = np.random.default_rng(seed); weights = rng.normal(0, 0.01, design.shape[1])
    for _ in range(800):
        logits = np.clip(design @ weights, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        gradient = design.T @ (probability - y) / len(y)
        gradient[:-1] += 3e-3 * weights[:-1]
        weights -= 0.05 * gradient
    return weights, mean, std


def task_bootstrap_auc(labels: np.ndarray, scores: np.ndarray, tasks: np.ndarray,
                       trials: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260813)
    unique = np.asarray(sorted(set(tasks.tolist())))
    values = []
    for _ in range(trials):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(tasks == task) for task in sampled])
        value = auc(labels[indices], scores[indices])
        if np.isfinite(value): values.append(value)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.5, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--repro-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.dataset_report.read_text())
    audit = json.loads(args.repro_audit.read_text())
    if report.get("status") != "complete" or audit.get("status") != "PASS":
        raise ValueError("R10-B upstream gate is not PASS")
    if report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset hash mismatch")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    labels = data["hazard_label"].astype(np.float64)
    tasks, policies, folds = data["task_id"], data["policy_id"], data["outer_fold"]
    features = summarize_features(data)
    results, oof_by_group = {}, {}
    for name, values in features.items():
        oof = np.full(len(labels), np.nan); fold_aurocs = []; errors = []
        for fold in range(5):
            validation = folds == fold; train = ~validation
            if len(np.unique(labels[train])) != 2 or len(np.unique(labels[validation])) != 2:
                errors.append(fold); continue
            weights, mean, std = fit_logistic(values[train], labels[train], 10100 + fold)
            z = (values[validation] - mean) / std
            logits = np.c_[z, np.ones(validation.sum())] @ weights
            oof[validation] = logits; fold_aurocs.append(auc(labels[validation], logits))
        valid = np.isfinite(oof)
        overall = auc(labels[valid], oof[valid])
        interval = task_bootstrap_auc(labels[valid], oof[valid], tasks[valid])
        policy_auc = {str(policy): auc(labels[valid & (policies == policy)],
                                       oof[valid & (policies == policy)])
                      for policy in sorted(set(policies.tolist()))}
        results[name] = {
            "fold_aurocs": fold_aurocs, "invalid_folds": errors,
            "oof_auroc": overall, "task_bootstrap_95": interval,
            "policy_aurocs": policy_auc,
        }
        oof_by_group[name] = oof
    # Cross-fitted policy-only prior; validation labels never estimate their
    # own policy prevalence.
    prior = np.full(len(labels), np.nan)
    for fold in range(5):
        validation = folds == fold; train = ~validation
        global_rate = float(labels[train].mean())
        for policy in sorted(set(policies.tolist())):
            source = train & (policies == policy)
            prior[validation & (policies == policy)] = float(labels[source].mean()) if source.any() else global_rate
    prior_auc = auc(labels, prior)
    all_causal = results["all_causal"]
    temporal = results["temporal_state"]
    temporal_action = results["temporal_plus_action"]
    gate = {
        "all_five_folds_valid": all(not row["invalid_folds"] for row in results.values()),
        "all_causal_oof_auroc_at_least_0p65": all_causal["oof_auroc"] >= 0.65,
        "all_causal_task_bootstrap_lower_at_least_0p58": all_causal["task_bootstrap_95"][0] >= 0.58,
        "temporal_state_oof_auroc_at_least_0p60": temporal["oof_auroc"] >= 0.60,
        "temporal_plus_action_oof_auroc_at_least_0p60": temporal_action["oof_auroc"] >= 0.60,
        "each_policy_all_causal_auroc_at_least_0p60": all(
            value >= 0.60 for value in all_causal["policy_aurocs"].values()),
        "all_causal_beats_crossfit_policy_prior_by_0p05": all_causal["oof_auroc"] >= prior_auc + 0.05,
    }
    status = "PASS" if all(gate.values()) else "FAIL"
    result = {
        "schema_version": "rase-r10c-case-control-information-gate/v1",
        "status": status,
        "decision": "UNLOCK_R10D_LIGHTWEIGHT_OOF" if status == "PASS"
                    else "STOP_BEFORE_R10D_MODEL",
        "scientific_scope": "case-control task-held-out representation information only",
        "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "repro_audit_sha256": sha256(args.repro_audit),
        "rows": len(labels), "tasks": len(set(tasks.tolist())),
        "labels": {"negative": int((labels == 0).sum()), "positive": int(labels.sum())},
        "crossfit_policy_prior_auroc": prior_auc, "feature_results": results,
        "gate": gate,
        "remains_locked": (["selector", "world_model", "validation", "test"]
                            if status == "PASS" else
                            ["risk_model", "selector", "world_model", "validation", "test"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "rows", "tasks", "labels", "crossfit_policy_prior_auroc",
        "feature_results", "gate")}, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

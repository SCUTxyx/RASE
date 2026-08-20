#!/usr/bin/env python3
"""Run the frozen low-capacity R9-B information-support gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean()
                  + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def folds(tasks: np.ndarray, suites: np.ndarray, count: int = 5) -> list[set[str]]:
    import hashlib
    result = [set() for _ in range(count)]
    unique = sorted(set(tasks.tolist()))
    for suite_index, suite in enumerate(sorted(set(suites.tolist()))):
        values = sorted(task for task in unique
                        if str(suites[np.flatnonzero(tasks == task)[0]]) == suite)
        values.sort(key=lambda value: hashlib.sha256(f"r9b-fold:{value}".encode()).hexdigest())
        for index, task in enumerate(values):
            result[(index + suite_index) % count].add(task)
    return result


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0)
    std = train.std(0)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std, mean


def fit_logistic(train: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, _, mean = standardize(train, train)
    std = train.std(0)
    # Recompute no-op normalized inputs are already standardized; retain an
    # explicit intercept and deterministic mini-batch-free gradient descent.
    x = np.concatenate([train, np.ones((len(train), 1), dtype=np.float32)], axis=1)
    y = labels.astype(np.float64)
    weights = rng.normal(0.0, 0.01, x.shape[1])
    for _ in range(600):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x.T @ (probability - y)) / len(y)
        gradient[:-1] += 1e-3 * weights[:-1]
        weights -= 0.08 * gradient
    return weights, mean


def predict(weights: np.ndarray, mean: np.ndarray, train_values: np.ndarray,
            values: np.ndarray) -> np.ndarray:
    std = train_values.std(0)
    std[std < 1e-6] = 1.0
    x = (values - mean) / std
    return 1.0 / (1.0 + np.exp(-np.clip(np.c_[x, np.ones(len(x))] @ weights, -30.0, 30.0)))


def probe_features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image = data["image_history"].astype(np.float32) / 255.0
    # Per-frame visual moments and causal frame deltas: no pixels after the
    # current boundary are used.
    frame_mean = image.mean(axis=(2, 3, 4, 5))
    frame_std = image.std(axis=(2, 3, 4, 5))
    frame_delta = np.abs(np.diff(image, axis=1)).mean(axis=(2, 3, 4, 5))
    image_features = np.c_[frame_mean, frame_std, frame_delta]
    proprio = data["proprio_history"].astype(np.float32)
    proprio_delta = data["proprio_delta_history"].astype(np.float32)
    temporal_features = np.c_[
        proprio[:, -1], proprio_delta[:, 1:].mean(1), proprio_delta[:, 1:].std(1),
        np.abs(proprio_delta[:, 1:]).max(1),
    ]
    action = data["action_history"].astype(np.float32)
    action_features = np.c_[
        action[:, -1], action[:, 1:].mean(1), action[:, 1:].std(1),
        np.abs(np.diff(action, axis=1)).mean(1),
    ]
    semantic = data["language_hash"].astype(np.float32)
    return {
        "image_sequence": image_features,
        "temporal_state": temporal_features,
        "action_history": action_features,
        "semantic": semantic,
        "temporal_plus_action": np.c_[temporal_features, action_features],
        "all_causal": np.c_[image_features, temporal_features, action_features, semantic],
    }


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
        raise ValueError("R9-B dataset/repro gate is not complete")
    if report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("R9-B dataset hash mismatch")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    labels = data["loss_hazard"].astype(np.float64)
    tasks, suites = data["task_id"], data["suite"]
    if len(set(tasks.tolist())) < 12:
        raise ValueError("R9-B information gate requires at least 12 tasks")
    support = {}
    for name, values in (("suite", suites), ("perturb_dim", data["perturb_dim"])):
        support[name] = {
            str(value): {
                "rows": int((values == value).sum()),
                "positives": int(labels[values == value].sum()),
                "negatives": int((values == value).sum() - labels[values == value].sum()),
            } for value in sorted(set(values.tolist()))
        }
    feature_groups = probe_features(data)
    fold_sets = folds(tasks, suites)
    results = {}
    for name, values in feature_groups.items():
        fold_auc = []
        skipped_folds = []
        for fold, validation_tasks in enumerate(fold_sets):
            train_mask = ~np.isin(tasks, list(validation_tasks))
            val_mask = ~train_mask
            if len(np.unique(labels[train_mask])) != 2 or len(np.unique(labels[val_mask])) != 2:
                skipped_folds.append(fold)
                continue
            train_features, val_features, mean = standardize(values[train_mask], values[val_mask])
            weights, _ = fit_logistic(train_features, labels[train_mask], seed=9001 + fold)
            # fit_logistic standardizes its input again; use its returned mean
            # and the same train std through a direct normalized prediction.
            train_std = train_features.std(0); train_std[train_std < 1e-6] = 1.0
            logits = np.c_[val_features, np.ones(len(val_features))] @ weights
            fold_auc.append(auc(labels[val_mask], logits))
        results[name] = {
            "fold_aurocs": fold_auc,
            "mean_auroc": float(np.mean(fold_auc)) if fold_auc else float("nan"),
            "minimum_auroc": float(np.min(fold_auc)) if fold_auc else float("nan"),
            "folds_with_auroc_at_least_0p65": int(sum(value >= 0.65 for value in fold_auc)),
            "expected_folds": len(fold_sets),
            "valid_folds": len(fold_auc),
            "skipped_folds": skipped_folds,
        }
    policy_baseline = np.asarray([
        (labels[(data["policy_id"] == policy) & (data["elapsed_source_steps"] == elapsed)].mean()
         if np.any((data["policy_id"] == policy) & (data["elapsed_source_steps"] == elapsed))
         else labels.mean())
        for policy, elapsed in zip(data["policy_id"], data["elapsed_source_steps"], strict=True)
    ])
    baseline_auc = auc(labels, policy_baseline)
    suite_support_pass = all(row["positives"] >= 20 and row["negatives"] >= 20
                             for row in support["suite"].values())
    perturb_support_pass = all(row["positives"] >= 20 and row["negatives"] >= 20
                               for row in support["perturb_dim"].values())
    all_causal = results["all_causal"]
    temporal = results["temporal_state"]
    action = results["action_history"]
    policy_results = {}
    for policy in sorted(set(data["policy_id"].tolist())):
        mask = data["policy_id"] == policy
        policy_results[policy] = {
            "rows": int(mask.sum()), "positives": int(labels[mask].sum()),
            "negatives": int(mask.sum() - labels[mask].sum()),
        }
    gate = {
        "tasks_at_least_12": len(set(tasks.tolist())) >= 12,
        "all_suites_support_20_each": suite_support_pass,
        "all_perturbations_support_20_each": perturb_support_pass,
        "at_least_four_causal_groups_mean_auroc_0p65": sum(
            result["mean_auroc"] >= 0.65 for result in results.values()
            if np.isfinite(result["mean_auroc"])
        ) >= 4,
        "all_five_task_held_out_folds_have_both_classes": all(
            result["valid_folds"] == result["expected_folds"]
            for result in results.values()
        ),
        "temporal_state_mean_auroc_at_least_0p60": temporal["mean_auroc"] >= 0.60,
        "all_causal_mean_auroc_at_least_0p65": all_causal["mean_auroc"] >= 0.65,
        "all_causal_beats_policy_horizon_prior_by_0p05": all_causal["mean_auroc"] >= baseline_auc + 0.05,
        "each_policy_support_20_each": all(row["positives"] >= 20 and row["negatives"] >= 20
                                            for row in policy_results.values()),
    }
    result = {
        "schema_version": "rase-r9b-information-gate/v1",
        "status": "PASS" if all(gate.values()) else "FAIL",
        "decision": "UNLOCK_R10_SHARED_RISK_OOF" if all(gate.values())
                    else "STOP_UNIVERSAL_RISK_FOR_CURRENT_OBSERVATIONS",
        "scientific_scope": "task-held-out low-capacity causal feature information gate",
        "dataset_sha256": sha256(args.dataset), "dataset_report_sha256": sha256(args.dataset_report),
        "repro_audit_sha256": sha256(args.repro_audit), "rows": int(len(labels)),
        "tasks": len(set(tasks.tolist())), "hazard_positives": int(labels.sum()),
        "hazard_prevalence": float(labels.mean()), "support": support,
        "policy_support": policy_results, "policy_horizon_prior_auroc": baseline_auc,
        "feature_results": results, "gate": gate,
        "remains_locked": (["world_model", "validation", "test", "closed_loop"]
                            if all(gate.values()) else
                            ["risk_model", "selector", "world_model", "validation", "test"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "rows", "tasks", "policy_horizon_prior_auroc",
        "feature_results", "gate",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

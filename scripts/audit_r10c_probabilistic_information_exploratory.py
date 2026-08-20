#!/usr/bin/env python3
"""Exploratory task-held-out information audit for R10-B K=3 counts.

This analysis is deliberately non-canonical: the probability target was
introduced only after the frozen deterministic R10-B reproducibility gate
failed.  It may justify a fresh, pre-registered confirmation cohort, but it
cannot unlock R10-D or a selector by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_auc(successes: np.ndarray, trials: np.ndarray, scores: np.ndarray) -> float:
    labels, expanded_scores = [], []
    for success, trial, score in zip(successes, trials, scores, strict=True):
        success_i, trial_i = int(success), int(trial)
        labels.extend([1] * success_i)
        labels.extend([0] * (trial_i - success_i))
        expanded_scores.extend([float(score)] * trial_i)
    labels_array = np.asarray(labels, dtype=bool)
    score_array = np.asarray(expanded_scores, dtype=np.float64)
    positive, negative = score_array[labels_array], score_array[~labels_array]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def binomial_log_loss(
    successes: np.ndarray, trials: np.ndarray, probabilities: np.ndarray
) -> float:
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    failures = trials - successes
    return float(
        -(successes * np.log(probabilities) + failures * np.log1p(-probabilities)).sum()
        / trials.sum()
    )


def summarize_features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image = data["image_history"].astype(np.float32) / 255.0
    image = image[:, -4:]
    image_features = np.c_[
        image.mean(axis=(2, 3, 4, 5)),
        image.std(axis=(2, 3, 4, 5)),
        np.abs(np.diff(image, axis=1)).mean(axis=(2, 3, 4, 5)),
    ]
    proprio = data["proprio_history"].astype(np.float32)
    delta = data["proprio_delta_history"].astype(np.float32)
    accel = data["proprio_accel_history"].astype(np.float32)
    temporal = np.c_[
        proprio[:, -1],
        delta[:, 1:].mean(1),
        delta[:, 1:].std(1),
        np.abs(delta[:, 1:]).max(1),
        accel[:, 2:].mean(1),
        accel[:, 2:].std(1),
    ]
    action = data["action_history"].astype(np.float32)
    action_delta = data["action_delta_history"].astype(np.float32)
    action_features = np.c_[
        action[:, -1],
        action.mean(1),
        action.std(1),
        np.abs(action_delta[:, 1:]).mean(1),
        np.abs(action_delta[:, 1:]).max(1),
    ]
    semantic = data["language_hash"].astype(np.float32)
    return {
        "image_sequence": image_features,
        "temporal_state": temporal,
        "action_history": action_features,
        "semantic": semantic,
        "temporal_plus_action": np.c_[temporal, action_features],
        "all_causal": np.c_[image_features, temporal, action_features, semantic],
    }


def fit_logistic_counts(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    design = np.c_[z, np.ones(len(z))]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, design.shape[1])
    total_trials = float(trials.sum())
    for _ in range(800):
        logits = np.clip(design @ weights, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (trials * probability - successes) / total_trials
        gradient[:-1] += 3e-3 * weights[:-1]
        weights -= 0.05 * gradient
    return weights, mean, std


def task_bootstrap_auc(
    successes: np.ndarray,
    trials: np.ndarray,
    scores: np.ndarray,
    tasks: np.ndarray,
    samples: int = 2000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260813)
    unique = np.asarray(sorted(set(tasks.tolist())))
    values = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(tasks == task) for task in selected])
        value = event_auc(successes[indices], trials[indices], scores[indices])
        if np.isfinite(value):
            values.append(value)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.5, 0.975]))


def crossfit_cell_prior(
    successes: np.ndarray,
    trials: np.ndarray,
    folds: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    prior = np.full(len(successes), np.nan, dtype=np.float64)
    for fold in range(5):
        validation, train = folds == fold, folds != fold
        global_rate = float(successes[train].sum() / trials[train].sum())
        for cell in sorted(set(cells.tolist())):
            source = train & (cells == cell)
            rate = (
                float(successes[source].sum() / trials[source].sum())
                if source.any()
                else global_rate
            )
            prior[validation & (cells == cell)] = rate
    return prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "complete_exploratory":
        raise ValueError("probability diagnostic dataset is not complete")
    if report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset hash mismatch")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    successes = data["hazard_successes_k3"].astype(np.float64)
    trials = data["hazard_trials_k3"].astype(np.float64)
    tasks = data["task_id"]
    policies = data["policy_id"]
    suites = data["suite"]
    folds = data["outer_fold"]
    selection_label = data["selection_label_k2"].astype(np.float64)
    if not np.all(trials == 3):
        raise ValueError("unexpected trial count")

    features = summarize_features(data)
    results = {}
    for name, values in features.items():
        oof = np.full(len(successes), np.nan)
        fold_aurocs, invalid_folds = [], []
        for fold in range(5):
            validation, train = folds == fold, folds != fold
            if successes[train].sum() in (0, trials[train].sum()) or successes[
                validation
            ].sum() in (0, trials[validation].sum()):
                invalid_folds.append(fold)
                continue
            weights, mean, std = fit_logistic_counts(
                values[train], successes[train], trials[train], 10100 + fold
            )
            z = (values[validation] - mean) / std
            logits = np.c_[z, np.ones(validation.sum())] @ weights
            oof[validation] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            fold_aurocs.append(
                event_auc(successes[validation], trials[validation], oof[validation])
            )
        valid = np.isfinite(oof)
        policy_aurocs = {
            str(policy): event_auc(
                successes[valid & (policies == policy)],
                trials[valid & (policies == policy)],
                oof[valid & (policies == policy)],
            )
            for policy in sorted(set(policies.tolist()))
        }
        results[name] = {
            "fold_aurocs": fold_aurocs,
            "invalid_folds": invalid_folds,
            "oof_event_auroc": event_auc(successes[valid], trials[valid], oof[valid]),
            "oof_binomial_log_loss": binomial_log_loss(
                successes[valid], trials[valid], oof[valid]
            ),
            "task_bootstrap_event_auroc_95": task_bootstrap_auc(
                successes[valid], trials[valid], oof[valid], tasks[valid]
            ),
            "policy_event_aurocs": policy_aurocs,
        }

    policy_prior = crossfit_cell_prior(successes, trials, folds, policies)
    policy_suite_cells = np.asarray(
        [f"{policy}|{suite}" for policy, suite in zip(policies, suites, strict=True)]
    )
    policy_suite_prior = crossfit_cell_prior(
        successes, trials, folds, policy_suite_cells
    )
    baselines = {
        "crossfit_policy_prior": {
            "event_auroc": event_auc(successes, trials, policy_prior),
            "binomial_log_loss": binomial_log_loss(successes, trials, policy_prior),
        },
        "crossfit_policy_suite_prior": {
            "event_auroc": event_auc(successes, trials, policy_suite_prior),
            "binomial_log_loss": binomial_log_loss(
                successes, trials, policy_suite_prior
            ),
        },
        "forbidden_k2_selection_label_sensitivity": {
            "event_auroc": event_auc(successes, trials, selection_label),
            "note": "audit-only; never a deployable input",
        },
    }
    strongest_prior_auc = max(
        baselines["crossfit_policy_prior"]["event_auroc"],
        baselines["crossfit_policy_suite_prior"]["event_auroc"],
    )
    all_causal = results["all_causal"]
    temporal = results["temporal_state"]
    temporal_action = results["temporal_plus_action"]
    exploratory_checks = {
        "all_five_folds_have_events_and_nonevents": all(
            not row["invalid_folds"] for row in results.values()
        ),
        "at_least_40_independent_k3_hazard_events": int(successes.sum()) >= 40,
        "at_least_10_hazard_positive_tasks": report["hazard_positive_tasks"] >= 10,
        "all_causal_event_auroc_at_least_0p65": all_causal["oof_event_auroc"] >= 0.65,
        "all_causal_task_bootstrap_lower_at_least_0p58": (
            all_causal["task_bootstrap_event_auroc_95"][0] >= 0.58
        ),
        "temporal_state_event_auroc_at_least_0p60": (
            temporal["oof_event_auroc"] >= 0.60
        ),
        "temporal_plus_action_event_auroc_at_least_0p60": (
            temporal_action["oof_event_auroc"] >= 0.60
        ),
        "each_policy_all_causal_event_auroc_at_least_0p60": all(
            value >= 0.60 for value in all_causal["policy_event_aurocs"].values()
        ),
        "all_causal_beats_strongest_crossfit_prior_by_0p05": (
            all_causal["oof_event_auroc"] >= strongest_prior_auc + 0.05
        ),
    }
    support = all(exploratory_checks.values())
    result = {
        "schema_version": "rase-r10c-probabilistic-information-exploratory/v1",
        "status": "EXPLORATORY_SUPPORT" if support else "EXPLORATORY_NO_SUPPORT",
        "decision": (
            "DESIGN_FRESH_CONFIRMATORY_PROBABILITY_COHORT"
            if support
            else "DO_NOT_ESCALATE_PROBABILISTIC_MODEL"
        ),
        "canonical_r10b_status_remains": "FAIL",
        "scientific_scope": "post-gate-failure diagnostic only",
        "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "groups": len(successes),
        "tasks": len(set(tasks.tolist())),
        "hazard_events": int(successes.sum()),
        "hazard_trials": int(trials.sum()),
        "hazard_positive_groups": int((successes > 0).sum()),
        "hazard_positive_tasks": int(
            len(set(tasks[successes > 0].tolist()))
        ),
        "feature_results": results,
        "baselines": baselines,
        "exploratory_checks": exploratory_checks,
        "remains_locked": [
            "risk_model",
            "selector",
            "world_model",
            "validation",
            "test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

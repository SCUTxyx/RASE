#!/usr/bin/env python3
"""Frozen-data observability diagnosis after the formal R8-B negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rase.risk.r7_source_protocol import task_folds  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(((positive[:, None] > negative[None, :]).mean()
                  + 0.5 * (positive[:, None] == negative[None, :]).mean()))


def entropy(labels: np.ndarray) -> float:
    if not len(labels):
        return float("nan")
    p = float(labels.mean())
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def categorical_information(labels: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    base = entropy(labels)
    conditional = 0.0
    for value in sorted(set(values.tolist())):
        mask = values == value
        conditional += float(mask.mean()) * entropy(labels[mask])
    gain = base - conditional
    return {
        "categories": len(set(values.tolist())), "label_entropy_bits": base,
        "conditional_entropy_bits": conditional, "information_gain_bits": gain,
        "normalized_information_gain": gain / base if base > 0 else 0.0,
    }


def smoothed_oof_categorical(labels: np.ndarray, tasks: np.ndarray, suites: np.ndarray,
                             categories: np.ndarray, *, alpha: float = 2.0) -> np.ndarray:
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    folds = task_folds(tasks, suites, count=5, seed=2026081207)
    all_tasks = set(tasks.tolist())
    for validation_tasks in folds:
        train = np.isin(tasks, list(all_tasks - validation_tasks))
        validation = np.isin(tasks, list(validation_tasks))
        global_rate = float(labels[train].mean())
        counts: dict[str, tuple[int, int]] = {}
        for value in sorted(set(categories[train].tolist())):
            mask = train & (categories == value)
            counts[str(value)] = (int(labels[mask].sum()), int(mask.sum()))
        for index in np.flatnonzero(validation):
            successes, trials = counts.get(str(categories[index]), (0, 0))
            prediction[index] = (successes + alpha * global_rate) / (trials + alpha)
    if not np.isfinite(prediction).all():
        raise AssertionError("categorical OOF prediction is incomplete")
    return prediction


def build_transitions(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(source["group_id"]):
        grouped[str(group)].append(index)
    records = []
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
                    or second_successes not in (0, second_trials)
                    or first_successes != first_trials
                    or str(source["cohort_role"][first]) != "natural"):
                continue
            records.append({
                "state_key": str(source["state_key"][first]),
                "group_id": str(source["group_id"][first]),
                "task_id": str(source["task_id"][first]),
                "suite": str(source["suite"][first]),
                "policy_id": str(source["policy_id"][first]),
                "horizon": start,
                "hazard": int(second_successes == 0),
            })
    return {key: np.asarray([row[key] for row in records]) for key in records[0]}


def prevalence_table(labels: np.ndarray, values: np.ndarray) -> dict[str, dict[str, float | int]]:
    return {str(value): {
        "rows": int((values == value).sum()),
        "positives": int(labels[values == value].sum()),
        "prevalence": float(labels[values == value].mean()),
    } for value in sorted(set(values.tolist()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--r8b-stability", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.dataset_report.read_text())
    stability = json.loads(args.r8b_stability.read_text())
    if report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset/report hash mismatch")
    if (stability.get("status") != "FAIL"
            or stability.get("decision") != "STOP_LOCAL_HAZARD_ESCALATION"
            or stability.get("pass_count") != 0):
        raise ValueError("R9 observability diagnosis requires formal R8-B 0/5 FAIL")
    initial = json.loads(args.initial_keys.read_text())
    metadata = {row["state_key"]: row for row in initial["records"]}
    with np.load(args.dataset, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    data = build_transitions(source)
    labels = data["hazard"].astype(np.float64)
    perturb_dim = np.asarray([metadata[key]["perturb_dim"] for key in data["state_key"]])
    perturb_level = np.asarray([metadata[key]["perturb_level"] for key in data["state_key"]])
    categories = {
        "policy": data["policy_id"],
        "horizon": data["horizon"].astype(str),
        "suite": data["suite"],
        "perturb_dim": perturb_dim,
        "perturb_level": perturb_level.astype(str),
        "policy_x_horizon": np.char.add(np.char.add(data["policy_id"], "|h"),
                                         data["horizon"].astype(str)),
        "suite_x_horizon": np.char.add(np.char.add(data["suite"], "|h"),
                                        data["horizon"].astype(str)),
        "policy_x_suite_x_horizon": np.asarray([
            f"{policy}|{suite}|h{horizon}" for policy, suite, horizon in
            zip(data["policy_id"], data["suite"], data["horizon"], strict=True)
        ]),
        "task": data["task_id"],
    }
    categorical = {}
    for name, value in categories.items():
        prediction = smoothed_oof_categorical(
            labels, data["task_id"], data["suite"], value
        ) if name != "task" else None
        categorical[name] = {
            "descriptive_information": categorical_information(labels, value),
            "task_held_out_smoothed_oof_auroc": None if prediction is None
                                               else auc(labels, prediction),
            # Perturbation identity/level are experimental metadata and are not
            # assumed observable at deployment.  Suite/task IDs are likewise
            # diagnostic only; policy and elapsed horizon are controller-known.
            "deployable": name in {"policy", "horizon", "policy_x_horizon"},
        }
    # Within-task empirical prevalence is explicitly leaky and reported only as
    # a clustering upper-bound diagnostic.
    task_score = np.asarray([
        labels[data["task_id"] == task].mean() for task in data["task_id"]
    ])
    seed_scores = []
    prediction_hashes = []
    for seed in (701, 702, 703, 704, 705):
        path = args.predictions_root / f"seed_{seed}.predictions.npz"
        if not path.is_file():
            raise ValueError(f"missing R8-B predictions: {path}")
        prediction_hashes.append({"seed": seed, "sha256": sha256(path)})
        with np.load(path, allow_pickle=False) as prediction:
            lookup = {
                (str(group), int(elapsed)): float(score)
                for group, elapsed, score, current in zip(
                    prediction["group_id"], prediction["elapsed_source_steps"],
                    prediction["hazard_probability"], prediction["current_recoverable"],
                    strict=True,
                ) if bool(current)
            }
        seed_scores.append(np.asarray([
            lookup[(str(group), int(horizon))]
            for group, horizon in zip(
                data["group_id"],
                data["horizon"], strict=True,
            )
        ]))
    mean_model_score = np.mean(seed_scores, axis=0)
    result = {
        "schema_version": "rase-r9-observability-audit/v1",
        "status": "complete",
        "decision": "NEW_TEMPORAL_DATA_CONTRACT_REQUIRED",
        "scientific_scope": "post-negative frozen-data diagnosis; not a method result",
        "dataset_sha256": sha256(args.dataset),
        "dataset_report_sha256": sha256(args.dataset_report),
        "initial_keys_sha256": sha256(args.initial_keys),
        "r8b_stability_sha256": sha256(args.r8b_stability),
        "rows": int(len(labels)), "positives": int(labels.sum()),
        "prevalence": float(labels.mean()), "tasks": len(set(data["task_id"].tolist())),
        "prevalence_by_policy": prevalence_table(labels, data["policy_id"]),
        "prevalence_by_horizon": prevalence_table(labels, data["horizon"]),
        "prevalence_by_suite": prevalence_table(labels, data["suite"]),
        "prevalence_by_perturb_dim": prevalence_table(labels, perturb_dim),
        "prevalence_by_perturb_level": prevalence_table(labels, perturb_level),
        "categorical_diagnostics": categorical,
        "leaky_within_task_prevalence_auroc": auc(labels, task_score),
        "five_seed_mean_model_score_auroc": auc(labels, mean_model_score),
        "prediction_hashes": prediction_hashes,
        "saved_causal_inputs": [
            "single boundary two-view RGB", "single boundary 8D proprio",
            "20D proposed source-action summary", "four previous 7D source actions",
            "elapsed boundary", "instruction hash", "source policy identity",
        ],
        "missing_candidate_mechanism_inputs": [
            "short observation frame sequence", "proprioception deltas",
            "object pose/velocity deltas", "contact/tactile history",
            "gripper transition history", "raw proposed action chunk",
            "fallback behavior descriptor", "source/fallback disagreement trajectory",
        ],
        "interpretation_rule": (
            "No new model is authorized. If task/suite diagnostics dominate deployable "
            "policy/horizon diagnostics and the five-seed mean remains weak, collect a "
            "small paired temporal pilot before reconsidering risk learning."
        ),
        "remains_locked": ["new risk model", "selector", "world-model", "new VLA",
                            "validation", "test", "closed-loop"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "decision", "rows", "positives", "prevalence",
        "five_seed_mean_model_score_auroc", "leaky_within_task_prevalence_auroc",
        "categorical_diagnostics",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

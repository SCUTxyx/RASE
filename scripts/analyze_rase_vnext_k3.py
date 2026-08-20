#!/usr/bin/env python3
"""K3 formal analysis: capture/capability audit, feature ladder OOF, action-swap,
risk-coverage, calibration, offline utility, and gate decisions.

Inputs: runs/rase_vnext/k3_collect_v1 (branches.jsonl + captures) and the frozen
k3_cohort_manifest_v1.json.  All gates are read from the frozen protocol;
nothing is re-selected after outcomes are observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.phase_c_pilot import (  # noqa: E402
    bootstrap_task_difference,
    grouped_metrics,
    raw_action_feature_vector,
    ridge_oof_predictions,
    task_folds,
    trace_feature_vector,
)
from rase.vnext.motion_trace import MotionSemanticMap  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_collection(output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branches_path = output_dir / "branches.jsonl"
    if not branches_path.exists():
        raise SystemExit(f"missing {branches_path}")
    rows = [json.loads(line) for line in branches_path.read_text().splitlines() if line.strip()]
    bound = json.loads((output_dir / "manifest.bound.json").read_text())
    return rows, bound


def capture_audit(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Three-layer audit: capture integrity, capability schema, provenance."""
    failures: list[str] = []
    meta_paths = sorted((output_dir / "captures").glob("*.json")) if (output_dir / "captures").exists() else []
    capture_failures: list[str] = []
    for meta_path in meta_paths:
        from rase.vnext.candidate_capture import audit_candidate_capture

        result = audit_candidate_capture(meta_path)
        if result["status"] != "PASS":
            capture_failures.append(f"{meta_path.name}: {result['failures']}")
    if capture_failures:
        failures.extend(capture_failures)
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operator[str(row["operator_id"])].append(row)
    capability_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for operator, operator_rows in sorted(by_operator.items()):
        for row in operator_rows:
            capability_counts[operator][str(row.get("capability_status"))] += 1
    provenance_missing = [
        f"{row['operator_id']}:{row['job_id']}"
        for row in rows
        if row.get("capability_status") == "executable"
        and row.get("inference_event_id") is None
        and row.get("chunk_origin") not in {"executed_trace"}
    ]
    if provenance_missing:
        failures.append(f"missing inference provenance: {provenance_missing[:5]} ...")
    queue_reconstruction = [
        f"{row['operator_id']}:{row['job_id']}"
        for row in rows
        if str(row.get("chunk_origin")) == "queue_reconstruction"
    ]
    if queue_reconstruction:
        failures.append(f"forbidden queue reconstruction: {queue_reconstruction[:5]} ...")
    executable_without_capture = [
        f"{row['operator_id']}:{row['job_id']}"
        for row in rows
        if row.get("capability_status") == "executable"
        and not row.get("candidate_capture_metadata_path")
    ]
    if executable_without_capture:
        failures.append(f"executable without capture: {executable_without_capture[:5]} ...")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "capture_files": len(meta_paths),
        "capability_counts": {k: dict(v) for k, v in capability_counts.items()},
        "rows_total": len(rows),
        "rows_executed": sum(
            row.get("execution_status") in {None, "executed"} and row.get("available") is True
            for row in rows
        ),
    }


def load_features_and_targets(
    rows: list[dict[str, Any]], output_dir: Path, bound: dict[str, Any],
) -> dict[str, Any]:
    """Build per-(root,operator,repeat) features from capture arrays."""
    captures = (output_dir / "captures").resolve()
    records: list[dict[str, Any]] = []
    state_features: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    trace_features: list[np.ndarray] = []
    semantic_features: list[np.ndarray] = []
    labels: list[dict[str, float]] = []
    errors: list[str] = []
    fold_by_task = {task: int(fold) for task, fold in bound["task_folds"].items()}
    semantics = (
        "delta_x", "delta_y", "delta_z",
        "delta_roll", "delta_pitch", "delta_yaw", "gripper",
    )
    semantic_map = MotionSemanticMap(
        translation=("delta_x", "delta_y", "delta_z"),
        rotation=("delta_roll", "delta_pitch", "delta_yaw"),
        gripper="gripper",
        translation_scale=0.05, rotation_scale=0.5,
        translation_frame="base", rotation_representation="euler_xyz",
        rotation_frame="eef",
    )
    for row in rows:
        if row.get("capability_status") != "executable":
            continue
        if row.get("execution_status") == "not_selected":
            continue
        meta_path = row.get("candidate_capture_metadata_path")
        if not meta_path:
            continue
        meta = json.loads(Path(meta_path).read_text())
        operator = str(row["operator_id"])
        operators = meta["operator_order"]
        if operator not in operators:
            continue
        arrays_path = Path(str(meta["arrays_path"]))
        if not arrays_path.exists():
            continue
        with np.load(arrays_path, allow_pickle=False) as arrays:
            index = operators.index(operator)
            actions = np.asarray(arrays["actions"][index], dtype=np.float32)
            mask = np.asarray(arrays["action_step_mask"][index], dtype=np.bool_)
            proprio = np.asarray(arrays["proprio"], dtype=np.float32)
            proprio_mask = np.asarray(arrays["proprio_mask"], dtype=np.bool_)
            # Continue candidates are captured as the *suffix* after the queue
            # cursor; for fair cross-operator comparison use the full chunk the
            # inference actually produced (same representation as requery /
            # fallback), never a 1-step leftover.
            full_key = f"full_env_chunk_{operator}"
            if full_key in arrays:
                full = np.asarray(arrays[full_key], dtype=np.float32)
                if len(full) >= 1:
                    actions = full
                    mask = np.ones(len(full), dtype=np.bool_)
        if not mask.any():
            continue
        valid_actions = actions[mask]
        record = {
            "row": row,
            "operator": operator,
            "root_id": str(row["root_id"]),
            "task_id": str(row["task_id"]),
            "suite": str(row["suite"]),
            "fold": fold_by_task.get(str(row["task_id"]), -1),
            "replica": int(row["seed_ledger"]["exact_repeat_replica"]),
            "actions": valid_actions,
            "proprio": proprio[proprio_mask] if proprio_mask.any() else proprio,
        }
        # P0 fix: feature computation and the record must be appended atomically;
        # a failed feature vector must drop the whole sample, never leave a
        # misaligned record behind (records/features/tasks must stay aligned).
        try:
            raw = raw_action_feature_vector(
                record["actions"], np.ones(len(record["actions"]), dtype=np.bool_),
            )
            trace = trace_feature_vector(
                record["actions"], np.ones(len(record["actions"]), dtype=np.bool_),
                semantics=semantics, policy_id="pi0fast.libero",
                semantic_map=semantic_map,
            )
            trace_flat = trace[:]
            state = np.asarray(record["proprio"], dtype=np.float64).reshape(-1)
            if state.size < 1:
                raise ValueError("empty proprio")
            state_padded = np.zeros(16, dtype=np.float64)
            state_padded[: min(state.size, 16)] = state[:16]
            row = record["row"]
            labels.append({
                "success": float(bool(row.get("success"))),
                "utility": float(
                    row.get("utility") if row.get("utility") is not None
                    else float(row.get("success"))
                ),
                "progress": float(row.get("post_decision_env_steps") or 0.0),
            })
        except Exception as exc:
            errors.append(f"{record['root_id']}/{record['operator']}: {type(exc).__name__}: {exc}")
            continue
        records.append(record)
        state_features.append(state_padded)
        raw_features.append(raw)
        trace_features.append(trace_flat)
        semantic_features.append(np.concatenate([raw, trace_flat]))
    if not records:
        raise SystemExit("no executable captured rows to analyze")
    # Alignment audit: every parallel list must be exactly as long as records.
    aligned = (
        len(records) == len(state_features) == len(raw_features)
        == len(trace_features) == len(semantic_features) == len(labels)
    )
    if not aligned:
        raise SystemExit(
            "feature extraction misalignment: records=%d state=%d raw=%d trace=%d "
            "semantic=%d labels=%d" % (
                len(records), len(state_features), len(raw_features),
                len(trace_features), len(semantic_features), len(labels),
            )
        )
    if errors:
        print("feature errors (dropped samples):", len(errors), errors[:3], file=sys.stderr)
    return {
        "records": records,
        "state": np.stack(state_features),
        "raw": np.stack(raw_features),
        "trace": np.stack(trace_features),
        "semantic": np.stack(semantic_features),
        "labels": labels,
        "tasks": [record["task_id"] for record in records],
        "groups": [record["root_id"] for record in records],
        "operators": [record["operator"] for record in records],
        "folds": [record["fold"] for record in records],
    }


def pairwise_ladder(dataset: dict[str, Any]) -> dict[str, Any]:
    """state-only vs raw/trace/semantic same-root pairwise ranking."""
    features = {
        "state-only": dataset["state"],
        "raw-action": dataset["raw"],
        "trace-only": dataset["trace"],
        "trace+semantic": dataset["semantic"],
    }
    # Primary same-root ranking target is *success* (protocol §6.1): utility
    # mixes operator-level costs (latency/query) into the label, which would
    # contaminate the action->outcome signal.  Utility is reported separately
    # in offline_utility.
    targets = np.asarray([label["success"] for label in dataset["labels"]], dtype=np.float64)
    folds_by_task = {
        task: fold for task, fold in zip(dataset["tasks"], dataset["folds"])
    }
    report: dict[str, Any] = {}
    for name, feature in features.items():
        predictions = ridge_oof_predictions(
            feature, targets, dataset["tasks"], folds_by_task, alpha=1.0,
        )
        metrics, _details = grouped_metrics(
            targets, predictions, dataset["groups"], tie_margin=0.0,
        )
        report[name] = {"oof_pairwise_accuracy": metrics["pairwise_accuracy"],
                        "pairwise_pairs": metrics["pairwise_pairs"],
                        "mean_oracle_regret": metrics["mean_oracle_regret"]}
    baseline = report["state-only"]["oof_pairwise_accuracy"]
    for name in ("raw-action", "trace-only", "trace+semantic"):
        report[name]["gain_vs_state_only"] = round(
            report[name]["oof_pairwise_accuracy"] - baseline, 5,
        )
    # Fold-level direction: per-fold pairwise accuracy (raw-action vs state)
    # computed on the main task-held-out OOF predictions (no re-fitting).
    best_action = "raw-action"
    all_preds: dict[str, np.ndarray] = {
        "state-only": ridge_oof_predictions(
            features["state-only"], targets, dataset["tasks"], folds_by_task, alpha=1.0,
        ),
        best_action: ridge_oof_predictions(
            features[best_action], targets, dataset["tasks"], folds_by_task, alpha=1.0,
        ),
    }
    fold_directions: dict[int, bool] = {}
    for fold in sorted(set(dataset["folds"])):
        test_indices = np.array([fold_value == fold for fold_value in dataset["folds"]])
        if int(test_indices.sum()) < 4:
            continue
        state_metric, _ = grouped_metrics(
            targets[test_indices], all_preds["state-only"][test_indices],
            [group for group, keep in zip(dataset["groups"], test_indices) if keep],
            tie_margin=0.0,
        )
        raw_metric, _ = grouped_metrics(
            targets[test_indices], all_preds[best_action][test_indices],
            [group for group, keep in zip(dataset["groups"], test_indices) if keep],
            tie_margin=0.0,
        )
        fold_directions[int(fold)] = bool(
            raw_metric["pairwise_accuracy"] >= state_metric["pairwise_accuracy"]
        )
    # Task-level bootstrap of per-task pairwise accuracy (raw-action vs state).
    task_accuracy: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for task in sorted(set(dataset["tasks"])):
        indices = np.array([value == task for value in dataset["tasks"]])
        for name in ("state-only", best_action):
            metric, _ = grouped_metrics(
                targets[indices], all_preds[name][indices],
                [group for group, keep in zip(dataset["groups"], indices) if keep],
                tie_margin=0.0,
            )
            task_accuracy[task][name] = metric["pairwise_accuracy"]
    gains = [task_accuracy[task][best_action] - task_accuracy[task]["state-only"] for task in task_accuracy]
    positive = sum(1 for gain in gains if gain > 0)
    bootstrap = bootstrap_task_difference(
        {task: [task_accuracy[task][best_action]] for task in task_accuracy},
        {task: [task_accuracy[task]["state-only"]] for task in task_accuracy},
        replicates=2000, seed=20270817,
    )
    report["fold_directions"] = {str(k): v for k, v in sorted(fold_directions.items())}
    report["folds_positive_direction"] = positive_folds = sum(1 for v in fold_directions.values() if v)
    report["folds_total"] = len(fold_directions)
    report["task_bootstrap_mean_gain"] = round(bootstrap[0], 5)
    report["task_bootstrap_95ci"] = [round(v, 5) for v in bootstrap[1]]
    report["tasks_positive_gain"] = positive
    report["tasks_total"] = len(gains)
    return report


def _fit_ridge_models(
    features: np.ndarray, targets: np.ndarray, tasks: Sequence[str],
    folds_by_task: Mapping[str, int], *, alpha: float = 1.0,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, float, np.ndarray]], dict[int, np.ndarray]]:
    """Per-fold standardized ridge models + their OOF predictions (original data)."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    models: dict[int, tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    for fold in sorted(set(folds_by_task.values())):
        test = np.array([folds_by_task[task] == fold for task in tasks])
        train = ~test
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (x[train] - mean) / scale
        x_test = (x[test] - mean) / scale
        y_mean = float(y[train].mean())
        design = np.column_stack((np.ones(len(x_train)), x_train))
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ (y[train] - y_mean))
        models[fold] = (mean, scale, y_mean, beta)
        predictions[test] = y_mean + np.column_stack((np.ones(len(x_test)), x_test)) @ beta
    return models, predictions


def _predict_with_models(
    features: np.ndarray, tasks: Sequence[str],
    folds_by_task: Mapping[str, int], models: Mapping[int, tuple[np.ndarray, np.ndarray, float, np.ndarray]],
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    predictions = np.full(len(x), np.nan, dtype=np.float64)
    for fold, (mean, scale, y_mean, beta) in models.items():
        test = np.array([folds_by_task[task] == fold for task in tasks])
        x_test = (x[test] - mean) / scale
        predictions[test] = y_mean + np.column_stack((np.ones(len(x_test)), x_test)) @ beta
    return predictions


def action_swap(dataset: dict[str, Any]) -> dict[str, Any]:
    """Swap action features across roots using the *original* OOF models.

    Protocol intent: predictions must change only through the action/trace
    features (state is untouched); swapping two candidates inside a root should
    degrade the same-root pairwise ranking.  Only capability-compatible
    executable candidates are swapped.
    """
    x = dataset["semantic"]
    targets = np.asarray([label["success"] for label in dataset["labels"]], dtype=np.float64)
    folds_by_task = {task: fold for task, fold in zip(dataset["tasks"], dataset["folds"])}
    models, original_preds = _fit_ridge_models(x, targets, dataset["tasks"], folds_by_task)
    original_metrics, _ = grouped_metrics(targets, original_preds, dataset["groups"], tie_margin=0.0)

    rng = np.random.default_rng(20270817)
    swapped_accuracies: list[float] = []
    swap_sensitivities: list[float] = []
    swapped_pairs = 0
    for _ in range(300):
        pair = rng.choice(len(dataset["records"]), size=2, replace=False)
        i, j = int(pair[0]), int(pair[1])
        if dataset["groups"][i] == dataset["groups"][j]:
            continue
        if dataset["operators"][i] == dataset["operators"][j]:
            continue
        x_swapped = x.copy()
        x_swapped[i] = x[j]
        x_swapped[j] = x[i]
        swapped_preds = _predict_with_models(x_swapped, dataset["tasks"], folds_by_task, models)
        metrics, _ = grouped_metrics(targets, swapped_preds, dataset["groups"], tie_margin=0.0)
        swapped_accuracies.append(metrics["pairwise_accuracy"])
        delta = np.abs(swapped_preds[[i, j]] - original_preds[[i, j]])
        swap_sensitivities.append(float((delta > 0.05).mean()))
        swapped_pairs += 1
    return {
        "original_pairwise_accuracy": original_metrics["pairwise_accuracy"],
        "original_pairs": original_metrics["pairwise_pairs"],
        "swapped_mean_pairwise_accuracy": (
            float(np.mean(swapped_accuracies)) if swapped_accuracies else None
        ),
        "swapped_pairs": swapped_pairs,
        "swap_sensitivity_gt_005": (
            float(np.mean(swap_sensitivities)) if swap_sensitivities else None
        ),
        "swap_effect": (
            round(original_metrics["pairwise_accuracy"] - float(np.mean(swapped_accuracies)), 5)
            if swapped_accuracies else None
        ),
        "swap_note": (
            "same-root/operator pairs swap only action-derived features with the "
            "original OOF models; positive swap_effect = action features carry "
            "same-root outcome information"
        ),
    }


def _brier_ece(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Brier score and ECE on [0,1]-clipped predictions (10 bins)."""
    clipped = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(targets, dtype=np.float64)
    brier = float(np.mean((clipped - y) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        selected = (clipped > left) & (clipped <= right)
        if not selected.any():
            continue
        ece += (int(selected.sum()) / len(clipped)) * abs(
            float(clipped[selected].mean()) - float(y[selected].mean())
        )
    return brier, ece


def risk_coverage_and_calibration(dataset: dict[str, Any]) -> dict[str, Any]:
    """Frozen-threshold selector coverage vs risk and Brier/ECE on success.

    The protocol gates the *relative* calibration change of action features vs
    the state-only baseline (delta Brier <= 0.02, delta ECE <= 0.05), not the
    absolute calibration of an uncalibrated ridge.
    """
    targets = np.asarray([label["success"] for label in dataset["labels"]], dtype=np.float64)
    folds_by_task = {task: fold for task, fold in zip(dataset["tasks"], dataset["folds"])}
    action_preds = ridge_oof_predictions(
        dataset["semantic"], targets, dataset["tasks"], folds_by_task, alpha=1.0,
    )
    state_preds = ridge_oof_predictions(
        dataset["state"], targets, dataset["tasks"], folds_by_task, alpha=1.0,
    )
    action_brier, action_ece = _brier_ece(action_preds, targets)
    state_brier, state_ece = _brier_ece(state_preds, targets)
    return {
        "action_brier": round(action_brier, 5),
        "action_ece": round(action_ece, 5),
        "state_only_brier": round(state_brier, 5),
        "state_only_ece": round(state_ece, 5),
        "delta_brier_vs_state_only": round(action_brier - state_brier, 5),
        "delta_ece_vs_state_only": round(action_ece - state_ece, 5),
        "prediction_mean": float(np.clip(action_preds, 0, 1).mean()),
        "target_mean": float(targets.mean()),
        "coverage_at_05": float(np.mean(action_preds >= 0.5)),
        "coverage_at_07": float(np.mean(action_preds >= 0.7)),
    }


def offline_utility(
    dataset: dict[str, Any],
    *,
    margin: float = 0.10,
    lambdas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0),
) -> dict[str, Any]:
    """Pure-prediction offline selector vs baselines (no realized-utility choice).

    Decision unit is (root, replica): every repeat is one independent boundary
    decision.  The selector ranks executable candidates by OOF success
    probability and abstains (falls back to continue) when the top-1/top-2
    margin is below the frozen threshold.  Realized outcomes are used only to
    *evaluate*; the selection itself never reads them.  Costs (query/fallback/
    latency) are applied only at the deployment layer as U_lambda = success -
    lambda * cost, and reported as a break-even curve against always-fallback.
    """
    success = np.asarray([label["success"] for label in dataset["labels"]], dtype=np.float64)
    folds_by_task = {task: fold for task, fold in zip(dataset["tasks"], dataset["folds"])}
    probabilities = ridge_oof_predictions(
        dataset["semantic"], success, dataset["tasks"], folds_by_task, alpha=1.0,
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    # decision units keyed by (root_id, replica)
    per_unit: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(dataset["records"]):
        per_unit[(str(record["root_id"]), int(record["replica"]))].append({
            "operator": str(record["operator"]),
            "p": float(probabilities[index]),
            "row": record["row"],
        })
    decisions: dict[str, list[str]] = {"selector": [], "continue": [], "requery": [], "fallback": []}
    utility_by_lambda: dict[float, dict[str, list[float]]] = {
        lam: {name: [] for name in decisions} | {"oracle": []} for lam in lambdas
    }
    for unit, candidates in per_unit.items():
        by_operator = {candidate["operator"]: candidate for candidate in candidates}
        cont = by_operator.get("continue.source")
        if cont is None:
            continue
        # selector: best predicted success probability, abstain on small margin
        ranked = sorted(candidates, key=lambda c: -c["p"])
        chosen = ranked[0]
        if len(ranked) >= 2 and (ranked[0]["p"] - ranked[1]["p"]) < margin:
            chosen = cont
        decisions["selector"].append(chosen["operator"])
        decisions["continue"].append("continue.source")
        decisions["requery"].append("requery.source" if "requery.source" in by_operator else "continue.source")
        decisions["fallback"].append("fallback.persistent" if "fallback.persistent" in by_operator else "continue.source")
        oracle_op = max(candidates, key=lambda c: float(c["row"].get("success") or 0.0))["operator"]
        for lam in lambdas:
            for name in decisions:
                op = decisions[name][-1]
                chosen_row = by_operator.get(op, cont)
                cost = (
                    float(chosen_row["row"].get("query_cost") or 0.0)
                    + float(chosen_row["row"].get("fallback_cost") or 0.0)
                    + float(chosen_row["row"].get("latency_cost") or 0.0)
                )
                value = float(chosen_row["row"].get("success") or 0.0) - lam * cost
                utility_by_lambda[lam][name].append(value)
            oracle_row = by_operator[oracle_op]
            oracle_cost = (
                float(oracle_row["row"].get("query_cost") or 0.0)
                + float(oracle_row["row"].get("fallback_cost") or 0.0)
                + float(oracle_row["row"].get("latency_cost") or 0.0)
            )
            utility_by_lambda[lam]["oracle"].append(
                float(oracle_row["row"].get("success") or 0.0) - lam * oracle_cost
            )
    unique_operators, unique_counts = np.unique(decisions["selector"], return_counts=True)
    report: dict[str, Any] = {
        "margin": margin,
        "decision_units": len(per_unit),
        "selector_operator_distribution": {
            str(operator): int(count)
            for operator, count in zip(unique_operators, unique_counts)
        } if decisions["selector"] else {},
        "break_even": {},
        "curves": {},
    }
    for lam in lambdas:
        curves = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in utility_by_lambda[lam].items()
        }
        report["curves"][str(lam)] = curves
        # break-even: smallest lambda where selector >= always-fallback
        selector_value = curves["selector"]
        fallback_value = curves["fallback"]
        report["break_even"][str(lam)] = {
            "selector_ge_continue": bool(selector_value >= curves["continue"]),
            "selector_ge_fallback": bool(selector_value >= fallback_value),
            "selector_minus_fallback": round(selector_value - fallback_value, 5),
        }
    # frozen-protocol headline numbers at the default lambda (0.1)
    headline = report["curves"]["0.1"]
    report.update({
        "headline_lambda": 0.1,
        "selector_mean_utility": headline["selector"],
        "continue_baseline_mean_utility": headline["continue"],
        "always_fallback_mean_utility": headline["fallback"],
        "oracle_mean_utility": headline["oracle"],
        "selector_gain_vs_continue": round(
            headline["selector"] - headline["continue"], 5,
        ),
        "oracle_regret_reduction": round(
            1.0 - (headline["oracle"] - headline["selector"])
            / max(headline["oracle"] - headline["continue"], 1e-9), 5,
        ),
    })
    return report


def operator_stripping_diagnostic(dataset: dict[str, Any]) -> dict[str, Any]:
    """Signal-source decomposition: operator-prior vs within-operator semantics.

    The full-cohort +0.55 raw-action gain could come mostly from the fallback
    action *distribution* (an operator prior: fallback chunks look different and
    succeed more often).  This diagnostic re-runs the same-root pairwise ladder
    on operator subsets:

      source-source  : continue.source vs requery.source only (same policy,
                       different sampling seed -> only action-level differences
                       separate the candidates; state is identical by design)
      cont-vs-fb     : continue.source vs fallback.persistent
      req-vs-fb      : requery.source vs fallback.persistent

    If source-source also shows a raw-action gain, the signal is not only an
    operator prior and the K3 story upgrades to within-policy action semantics.
    """
    operators = np.asarray(dataset["operators"])
    features: dict[str, np.ndarray] = {
        "state-only": dataset["state"],
        "raw-action": dataset["raw"],
        "trace-only": dataset["trace"],
    }
    targets = np.asarray([label["success"] for label in dataset["labels"]], dtype=np.float64)
    subsets = {
        "source-source": (
            (operators == "continue.source") | (operators == "requery.source")
        ),
        "cont-vs-fb": (
            (operators == "continue.source") | (operators == "fallback.persistent")
        ),
        "req-vs-fb": (
            (operators == "requery.source") | (operators == "fallback.persistent")
        ),
    }
    report: dict[str, Any] = {}
    for subset_name, mask in subsets.items():
        indices = np.flatnonzero(mask)
        if len(indices) < 12:
            report[subset_name] = {"error": "too few rows"}
            continue
        tasks = [dataset["tasks"][i] for i in indices]
        groups = [dataset["groups"][i] for i in indices]
        folds = {task: fold for task, fold in zip(tasks, [dataset["folds"][i] for i in indices])}
        subset_targets = targets[indices]
        entry: dict[str, Any] = {"rows": int(len(indices)), "roots": len(set(groups))}
        for name in ("state-only", "raw-action", "trace-only"):
            predictions = ridge_oof_predictions(
                features[name][indices], subset_targets, tasks, folds, alpha=1.0,
            )
            metrics, _ = grouped_metrics(
                subset_targets, predictions, groups, tie_margin=0.0,
            )
            entry[name] = {
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "pairwise_pairs": metrics["pairwise_pairs"],
                "mean_oracle_regret": metrics["mean_oracle_regret"],
            }
        entry["gain_raw_vs_state"] = round(
            entry["raw-action"]["pairwise_accuracy"]
            - entry["state-only"]["pairwise_accuracy"], 5,
        )
        report[subset_name] = entry
    # Per-operator-pair accuracy breakdown on the full cohort (same OOF model
    # per feature, pairs restricted to the operator pair in question).
    full_preds: dict[str, np.ndarray] = {}
    folds_by_task = {task: fold for task, fold in zip(dataset["tasks"], dataset["folds"])}
    for name in ("state-only", "raw-action"):
        full_preds[name] = ridge_oof_predictions(
            features[name], targets, dataset["tasks"], folds_by_task, alpha=1.0,
        )
    pair_breakdown: dict[str, Any] = {}
    for left, right in (
        ("continue.source", "requery.source"),
        ("continue.source", "fallback.persistent"),
        ("requery.source", "fallback.persistent"),
    ):
        mask = (operators == left) | (operators == right)
        indices = np.flatnonzero(mask)
        pair_entry: dict[str, Any] = {}
        for name in ("state-only", "raw-action"):
            metrics, _ = grouped_metrics(
                targets[indices], full_preds[name][indices],
                [dataset["groups"][i] for i in indices], tie_margin=0.0,
            )
            pair_entry[name] = metrics["pairwise_accuracy"]
        pair_entry["gain_raw_vs_state"] = round(
            pair_entry["raw-action"] - pair_entry["state-only"], 5,
        )
        pair_breakdown[f"{left}|{right}"] = pair_entry
    report["pair_breakdown"] = pair_breakdown
    report["interpretation"] = (
        "source-source gain > 0 => within-policy action semantics are learnable "
        "(not merely an operator prior); source-source gain ~ 0 with cont-vs-fb / "
        "req-vs-fb gains large => signal is mostly the fallback distribution prior."
    )
    return report


def gate_decisions(report: dict[str, Any]) -> dict[str, Any]:
    ladder = report["pairwise_ladder"]
    gains = {name: ladder[name].get("gain_vs_state_only") for name in ("raw-action", "trace-only", "trace+semantic")}
    signal_pass = any(
        gain is not None and gain >= 0.03 for gain in gains.values()
    )
    utility = report["offline_utility"]
    utility_pass = (
        utility["selector_gain_vs_continue"] >= 0.05
        or utility["oracle_regret_reduction"] >= 0.10
    )
    calibration_ok = (
        report["risk_coverage"]["delta_brier_vs_state_only"] <= 0.02
        and report["risk_coverage"]["delta_ece_vs_state_only"] <= 0.05
    )
    return {
        "K3_CAPTURE_PASS": report["capture_audit"]["status"] == "PASS",
        "K3_SIGNAL_PASS": bool(signal_pass),
        "K3_UTILITY_PASS": bool(utility_pass),
        "calibration_within_budget": bool(calibration_ok),
        "signal_gains": gains,
        "utility_gain": utility["selector_gain_vs_continue"],
        "oracle_regret_reduction": utility["oracle_regret_reduction"],
        "status": (
            "K3_SIGNAL_AND_UTILITY_PASS"
            if signal_pass and utility_pass else
            "K3_SIGNAL_PASS_UTILITY_FAIL" if signal_pass else
            "K3_UTILITY_PASS_SIGNAL_FAIL" if utility_pass else
            "K3_NO_PASS"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20270817)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen_confirmation":
        raise SystemExit("manifest is not frozen")
    rows, bound = load_collection(args.output_dir)
    report: dict[str, Any] = {
        "schema_version": "rase-vnext-k3-analysis/v1",
        "manifest_sha256": sha256(args.manifest),
        "manifest_status": manifest.get("status"),
    }
    report["capture_audit"] = capture_audit(args.output_dir, rows)
    dataset = load_features_and_targets(rows, args.output_dir, bound)
    report["rows"] = {
        "total": len(rows),
        "executable": len(dataset["records"]),
        "suites": {
            suite: int(sum(1 for rec in dataset["records"] if rec["suite"] == suite))
            for suite in ("Spatial", "Object", "Goal", "Long")
        },
        "tasks": len(set(dataset["tasks"])),
        "roots": len(set(dataset["groups"])),
    }
    report["pairwise_ladder"] = pairwise_ladder(dataset)
    report["action_swap"] = action_swap(dataset)
    report["operator_stripping"] = operator_stripping_diagnostic(dataset)
    report["risk_coverage"] = risk_coverage_and_calibration(dataset)
    report["offline_utility"] = offline_utility(dataset)
    report["gates"] = gate_decisions(report)
    atomic_json(args.output_dir / "k3_analysis.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

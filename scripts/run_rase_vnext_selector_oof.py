#!/usr/bin/env python3
"""K5 nested task-held-out OOF for the semantic selector (v2 plan §6-§8).

Outer folds: 4 x 12 tasks (frozen).  For each outer fold the model is trained
on the 27 non-calibration tasks of the other three folds; the abstain margin is
selected once on the 9 calibration tasks of those folds (inner choice); the 12
outer-test tasks are evaluated exactly once.  Two model families are compared:
candidate-level ridge and explicit pairwise ridge.  Costs enter only at the
deployment layer (U_lambda); no realized outcome is used for selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.phase_c_pilot import grouped_metrics  # noqa: E402
from rase.vnext.selector import (  # noqa: E402
    fit_pairwise,
    fit_ridge,
    risk_coverage_curve,
    select_candidates,
    utility_lambda,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambdas", type=str, default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--margins", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    manifest = json.loads(args.dataset.read_text())
    if manifest.get("status") != "frozen":
        raise SystemExit("dataset manifest is not frozen")
    arrays = np.load(Path(str(manifest["features_path"])), allow_pickle=False)
    rows = manifest["rows"]
    n = manifest["n_rows"]
    tasks = np.asarray([row["task_id"] for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    calib = np.asarray([bool(row["calibration_split"]) for row in rows])
    operators = np.asarray([row["operator"] for row in rows])
    roots = np.asarray([row["root_id"] for row in rows])
    replicas = np.asarray([int(row["replica"]) for row in rows])
    success = np.asarray([bool(row["success"]) for row in rows], dtype=np.float64)
    costs = np.asarray([
        float(row.get("query_cost") or 0.0)
        + float(row.get("fallback_cost") or 0.0)
        + float(row.get("latency_cost") or 0.0)
        for row in rows
    ], dtype=np.float64)
    features: dict[str, np.ndarray] = {
        "raw-action": arrays["raw"],
        "trace-only": arrays["trace"],
        "semantic": arrays["semantic"],
    }
    arrays.close()
    lambdas = [float(value) for value in args.lambdas.split(",")]
    margins = [float(value) for value in args.margins.split(",")]
    outer_folds = sorted(set(int(f) for f in folds))

    report: dict[str, Any] = {
        "schema_version": "rase-vnext-selector-oof/v1",
        "dataset_manifest_sha256": args.dataset.read_text() and __import__(
            "hashlib"
        ).sha256(args.dataset.read_bytes()).hexdigest(),
        "alpha": args.alpha,
        "lambdas": lambdas,
        "margins": margins,
        "outer_folds": outer_folds,
    }
    for feature_name, feature in features.items():
        feature_report: dict[str, Any] = {}
        for model_family in ("candidate", "pairwise"):
            unit_scores: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
            unit_success: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
            unit_cost: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
            chosen_by_margin: dict[float, dict[tuple[str, int], str]] = {
                margin: {} for margin in margins
            }
            chosen_margin: float | None = None
            # per-fold: train on 27 tasks, pick margin on 9 calib tasks, eval on 12
            for outer_fold in outer_folds:
                test_mask = folds == outer_fold
                calib_mask = calib & ~test_mask
                train_mask = ~calib_mask & ~test_mask
                model = None
                if model_family == "candidate":
                    model = fit_ridge(
                        feature[train_mask], success[train_mask], alpha=args.alpha,
                        model_type="candidate", feature_version="rase-vnext-selector-features/v1",
                        training_manifest_sha256=manifest.get("sha256", ""),
                    )
                else:
                    # Explicit pairwise: same-root/replica candidate pairs.
                    train_indices = [i for i in range(n) if bool(train_mask[i])]
                    units: dict[tuple[str, int], list[int]] = defaultdict(list)
                    for i in train_indices:
                        units[(str(rows[i]["root_id"]), int(rows[i]["replica"]))].append(i)
                    pairs_a: list[np.ndarray] = []
                    pairs_b: list[np.ndarray] = []
                    deltas: list[float] = []
                    for indices in units.values():
                        for a_index in range(len(indices)):
                            for b_index in range(a_index + 1, len(indices)):
                                ia, ib = indices[a_index], indices[b_index]
                                pairs_a.append(feature[ia])
                                pairs_b.append(feature[ib])
                                deltas.append(float(success[ia] - success[ib]))
                    if len(pairs_a) >= 10:
                        model = fit_pairwise(
                            np.stack(pairs_a), np.stack(pairs_b),
                            np.asarray(deltas, dtype=np.float64), alpha=args.alpha,
                            feature_version="rase-vnext-selector-features/v1",
                            training_manifest_sha256=manifest.get("sha256", ""),
                        )
                    else:
                        model = fit_ridge(
                            feature[train_mask], success[train_mask], alpha=args.alpha,
                            model_type="candidate", feature_version="rase-vnext-selector-features/v1",
                            training_manifest_sha256=manifest.get("sha256", ""),
                        )
                # inner margin selection on calibration tasks (utility at lambda 0.1)
                calib_rows = [i for i in range(n) if bool(calib_mask[i])]
                calib_units: dict[tuple[str, int], list[int]] = defaultdict(list)
                for i in calib_rows:
                    calib_units[(str(roots[i]), int(replicas[i]))].append(i)
                margin_utility: dict[float, list[float]] = {m: [] for m in margins}
                for unit, indices in calib_units.items():
                    unit_scores_unit = _unit_scores(
                        model, feature, indices, operators, model_family,
                    )
                    for margin in margins:
                        decision = select_candidates(unit_scores_unit, abstain_margin=margin)
                        chosen_index = next(
                            i for i in indices if str(operators[i]) == decision.chosen_operator
                        )
                        margin_utility[margin].append(
                            utility_lambda(success[chosen_index], costs[chosen_index], 0.1)
                        )
                best_margin = max(
                    margins,
                    key=lambda m: float(np.mean(margin_utility[m])) if margin_utility[m] else -1.0,
                )
                chosen_margin = best_margin
                # outer evaluation (once): collect unit scores
                test_rows = [i for i in range(n) if bool(test_mask[i])]
                test_units: dict[tuple[str, int], list[int]] = defaultdict(list)
                for i in test_rows:
                    test_units[(str(roots[i]), int(replicas[i]))].append(i)
                for unit, indices in test_units.items():
                    unit_scores[unit] = _unit_scores(
                        model, feature, indices, operators, model_family,
                    )
                    for i in indices:
                        unit_success[unit][str(operators[i])] = float(success[i])
                        unit_cost[unit][str(operators[i])] = float(costs[i])
                    for margin in margins:
                        decision = select_candidates(unit_scores[unit], abstain_margin=margin)
                        chosen_by_margin[margin][unit] = decision.chosen_operator
                # record frozen margin for this fold
                feature_report.setdefault("frozen_margin_per_fold", {})[str(outer_fold)] = best_margin
            # aggregate
            all_pairs_accuracy, pair_details = _pairwise_metrics(
                unit_scores, unit_success, operators,
            )
            entry: dict[str, Any] = {
                "pairwise_accuracy": all_pairs_accuracy,
                "pairs": sum(value[1] for value in pair_details.values()),
                "units": len(unit_scores),
                "task_bootstrap": _task_bootstrap(unit_scores, unit_success, rows, folds),
            }
            # layered operator-pair accuracy (all-pairs within unit)
            layer = _layered_accuracy(unit_scores, unit_success)
            entry["layered"] = layer
            # selector outcomes per margin
            selector_report: dict[str, Any] = {}
            for margin in margins:
                chosen_ops = [chosen_by_margin[margin][unit] for unit in unit_scores]
                cont_units = sum(1 for op in chosen_ops if op == "continue.source")
                abstain_rate = sum(
                    1 for unit in unit_scores
                    if select_candidates(unit_scores[unit], abstain_margin=margin).abstained
                ) / len(unit_scores) if unit_scores else 0.0
                curves: dict[str, dict[str, float]] = {}
                for lam in lambdas:
                    selector_values: list[float] = []
                    continue_values: list[float] = []
                    fallback_values: list[float] = []
                    requery_values: list[float] = []
                    oracle_values: list[float] = []
                    for unit in unit_scores:
                        ops = unit_scores[unit]
                        chosen = chosen_by_margin[margin][unit]
                        selector_values.append(
                            utility_lambda(unit_success[unit][chosen], unit_cost[unit][chosen], lam)
                        )
                        continue_values.append(utility_lambda(
                            unit_success[unit].get("continue.source", 0.0),
                            unit_cost[unit].get("continue.source", 0.0), lam,
                        ))
                        fallback_values.append(utility_lambda(
                            unit_success[unit].get("fallback.persistent", 0.0),
                            unit_cost[unit].get("fallback.persistent", 0.0), lam,
                        ))
                        requery_values.append(utility_lambda(
                            unit_success[unit].get("requery.source", 0.0),
                            unit_cost[unit].get("requery.source", 0.0), lam,
                        ))
                        oracle_value = max(
                            utility_lambda(unit_success[unit][op], unit_cost[unit][op], lam)
                            for op in ops
                        )
                        oracle_values.append(oracle_value)
                    curves[str(lam)] = {
                        "selector": float(np.mean(selector_values)),
                        "continue": float(np.mean(continue_values)),
                        "requery": float(np.mean(requery_values)),
                        "fallback": float(np.mean(fallback_values)),
                        "oracle": float(np.mean(oracle_values)),
                    }
                selector_report[str(margin)] = {
                    "abstain_rate": round(float(abstain_rate), 4),
                    "chosen_continue": cont_units,
                    "chosen_distribution": {
                        str(operator): int(count)
                        for operator, count in zip(
                            *np.unique(np.asarray(chosen_ops, dtype=object), return_counts=True)
                        )
                    } if chosen_ops else {},
                    "curves": curves,
                    "break_even_lambda_ge_fallback": [
                        lam for lam in lambdas
                        if curves[str(lam)]["selector"] >= curves[str(lam)]["fallback"]
                    ],
                }
            entry["selector"] = selector_report
            feature_report[model_family] = entry
        # risk-coverage on candidate scores (raw-action headline)
        report[feature_name] = feature_report
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _unit_scores(
    model: Any,
    feature: np.ndarray,
    indices: list[int],
    operators: np.ndarray,
    family: str,
) -> dict[str, float]:
    """Per-candidate scores inside one decision unit.

    Candidate family: absolute P(success).  Pairwise family: score is
    P(success_op > success_continue) relative to the unit's continue candidate
    (continue itself scores 0.5).
    """
    scores: dict[str, float] = {}
    if family == "candidate":
        for i in indices:
            scores[str(operators[i])] = float(model.predict(feature[i:i + 1])[0])
        return scores
    cont_index = next(
        (i for i in indices if str(operators[i]) == "continue.source"), None,
    )
    for i in indices:
        operator = str(operators[i])
        if operator == "continue.source":
            scores[operator] = 0.5
        elif cont_index is not None:
            from rase.vnext.selector import predict_pairwise

            scores[operator] = float(predict_pairwise(model, feature[i:i + 1], feature[cont_index:cont_index + 1])[0])
        else:
            scores[operator] = float(model.predict(feature[i:i + 1])[0])
    return scores


def _pairwise_metrics(
    unit_scores: dict[tuple[str, int], dict[str, float]],
    unit_success: dict[tuple[str, int], dict[str, float]],
    operators: np.ndarray,
) -> tuple[float, dict[tuple[str, int], Any]]:
    correct = total = 0
    details: dict[tuple[str, int], Any] = {}
    for unit, scores in unit_scores.items():
        truth = unit_success[unit]
        ops = list(scores)
        unit_correct = unit_total = 0
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                left, right = ops[i], ops[j]
                if truth[left] == truth[right]:
                    continue
                predicted = scores[left] - scores[right]
                actual = truth[left] - truth[right]
                unit_correct += int(predicted * actual > 0)
                unit_total += 1
        correct += unit_correct
        total += unit_total
        details[unit] = (unit_correct, unit_total)
    return (correct / total if total else 0.0), details


def _task_bootstrap(
    unit_scores: dict[tuple[str, int], dict[str, float]],
    unit_success: dict[tuple[str, int], dict[str, float]],
    rows: list[dict[str, Any]],
    folds: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 20270818,
) -> dict[str, Any]:
    """Per-task pairwise accuracy + paired bootstrap over tasks."""
    task_of_unit: dict[tuple[str, int], str] = {}
    for index, row in enumerate(rows):
        task_of_unit[(str(row["root_id"]), int(row["replica"]))] = str(row["task_id"])
    per_task: dict[str, list[float]] = {}
    for unit, scores in unit_scores.items():
        task = task_of_unit.get(unit)
        if task is None:
            continue
        truth = unit_success[unit]
        ops = list(scores)
        correct = total = 0
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                left, right = ops[i], ops[j]
                if truth[left] == truth[right]:
                    continue
                predicted = scores[left] - scores[right]
                actual = truth[left] - truth[right]
                correct += int(predicted * actual > 0)
                total += 1
        if total:
            per_task.setdefault(task, []).append(correct / total)
    tasks = sorted(per_task)
    values = np.asarray([float(np.mean(per_task[task])) for task in tasks])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": round(float(values.mean()), 5),
        "ci95": [round(float(np.quantile(samples, 0.025)), 5),
                 round(float(np.quantile(samples, 0.975)), 5)],
        "tasks": len(tasks),
        "positive_tasks": int((values > 0.5).sum()),
    }


def _layered_accuracy(
    unit_scores: dict[tuple[str, int], dict[str, float]],
    unit_success: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    layers: dict[str, list[bool]] = defaultdict(list)
    for unit, scores in unit_scores.items():
        truth = unit_success[unit]
        for pair in (
            ("continue.source", "requery.source"),
            ("continue.source", "fallback.persistent"),
            ("requery.source", "fallback.persistent"),
        ):
            if pair[0] not in scores or pair[1] not in scores:
                continue
            if truth[pair[0]] == truth[pair[1]]:
                continue
            predicted = scores[pair[0]] - scores[pair[1]]
            actual = truth[pair[0]] - truth[pair[1]]
            layers["|".join(pair)].append(bool(predicted * actual > 0))
    return {
        layer: round(float(np.mean(values)), 4) if values else None
        for layer, values in layers.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())

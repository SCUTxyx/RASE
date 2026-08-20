#!/usr/bin/env python3
"""B3: pi0.5 challenge analysis — two layers, never pooled.

Layer 1 (in-policy): train ridge models on pi0.5 rows only, task-held-out
(4 folds x 2 tasks, train 6 tasks per fold), report same-root pairwise
accuracy, source-source layer, task bootstrap CI.

Layer 2 (cross-policy transfer): train the pairwise model on the full frozen
K5 (pi0-fast) dataset and evaluate on pi0.5 rows without any pi0.5 training;
report the same metrics plus the gap vs the in-policy model.

No universal/pooled selector is trained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.selector import fit_pairwise, fit_ridge, predict_pairwise  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _unit_scores_pairwise(
    model: Any, feature: np.ndarray, indices: list[int], operators: np.ndarray,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    cont_index = next(
        (i for i in indices if str(operators[i]) == "continue.source"), None,
    )
    for i in indices:
        operator = str(operators[i])
        if operator == "continue.source":
            scores[operator] = 0.5
        elif cont_index is not None:
            scores[operator] = float(predict_pairwise(
                model, feature[i:i + 1], feature[cont_index:cont_index + 1],
            )[0])
        else:
            scores[operator] = float(model.predict(feature[i:i + 1])[0])
    return scores


def _evaluate(
    unit_scores: dict[tuple[str, int], dict[str, float]],
    unit_success: dict[tuple[str, int], dict[str, float]],
    task_of_unit: dict[tuple[str, int], str],
    replicates: int = 2000, seed: int = 20270818,
) -> dict[str, Any]:
    correct = total = 0
    layered: dict[str, list[bool]] = defaultdict(list)
    per_task: dict[str, list[float]] = defaultdict(list)
    for unit, scores in unit_scores.items():
        truth = unit_success[unit]
        ops = list(scores)
        for a_index in range(len(ops)):
            for b_index in range(a_index + 1, len(ops)):
                left, right = ops[a_index], ops[b_index]
                if truth[left] == truth[right]:
                    continue
                predicted = scores[left] - scores[right]
                actual = truth[left] - truth[right]
                hit = bool(predicted * actual > 0)
                correct += int(hit)
                total += 1
                for pair in (
                    ("continue.source", "requery.source"),
                    ("continue.source", "fallback.persistent"),
                    ("requery.source", "fallback.persistent"),
                ):
                    if {left, right} == set(pair):
                        layered["|".join(pair)].append(hit)
        task = task_of_unit.get(unit)
        if task is not None:
            per_task[task].append(correct_unit(unit, scores, truth))
    task_values = np.asarray([float(np.mean(per_task[t])) for t in per_task])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(task_values), size=(replicates, len(task_values)))
    samples = task_values[idx].mean(axis=1)
    return {
        "pairwise_accuracy": round(correct / total, 5) if total else None,
        "pairs": total,
        "task_bootstrap": {
            "mean": round(float(task_values.mean()), 5) if len(task_values) else None,
            "ci95": [round(float(np.quantile(samples, 0.025)), 5),
                     round(float(np.quantile(samples, 0.975)), 5)],
            "tasks": len(per_task),
            "positive_tasks": int((task_values > 0.5).sum()) if len(task_values) else 0,
        },
        "layered": {k: round(float(np.mean(v)), 4) if v else None for k, v in layered.items()},
        "units": len(unit_scores),
    }


def correct_unit(unit, scores, truth) -> float:
    ops = list(scores)
    unit_correct = unit_total = 0
    for a_index in range(len(ops)):
        for b_index in range(a_index + 1, len(ops)):
            left, right = ops[a_index], ops[b_index]
            if truth[left] == truth[right]:
                continue
            predicted = scores[left] - scores[right]
            actual = truth[left] - truth[right]
            unit_correct += int(predicted * actual > 0)
            unit_total += 1
    return unit_correct / unit_total if unit_total else 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-dataset", type=Path, required=True)
    parser.add_argument("--k5-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    def _load(path: Path) -> dict[str, Any]:
        manifest = json.loads(path.read_text())
        arrays = np.load(Path(str(manifest["features_path"])), allow_pickle=False)
        rows = manifest["rows"]
        return {
            "manifest": manifest,
            "feature": arrays["raw"],
            "rows": rows,
            "success": np.asarray([bool(row["success"]) for row in rows], dtype=np.float64),
        }

    pi05 = _load(args.pi05_dataset)
    k5 = _load(args.k5_dataset)
    n = len(pi05["rows"])
    feature = pi05["feature"]
    success = pi05["success"]
    rows = pi05["rows"]
    operators = np.asarray([row["operator"] for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    units: dict[tuple[str, int], list[int]] = defaultdict(list)
    task_of_unit: dict[tuple[str, int], str] = {}
    for i, row in enumerate(rows):
        key = (str(row["root_id"]), int(row["replica"]))
        units[key].append(i)
        task_of_unit[key] = str(row["task_id"])

    report: dict[str, Any] = {
        "schema_version": "rase-vnext-pi05-challenge-analysis/v1",
        "pi05_dataset_sha256": hashlib.sha256(args.pi05_dataset.read_bytes()).hexdigest(),
        "k5_dataset_sha256": hashlib.sha256(args.k5_dataset.read_bytes()).hexdigest(),
        "alpha": args.alpha,
        "note": "cross-policy challenge on shared K3 roots; NOT an independent external confirmation",
    }

    # ---- Layer 1: in-policy, task-held-out --------------------------------
    unit_scores: dict[tuple[str, int], dict[str, float]] = {}
    unit_success: dict[tuple[str, int], dict[str, float]] = {}
    for fold in sorted(set(int(f) for f in folds)):
        train_mask = folds != fold
        model = fit_ridge(
            feature[train_mask], success[train_mask], alpha=args.alpha,
            model_type="candidate",
            feature_version=pi05["manifest"].get("feature_version", "v1"),
            training_manifest_sha256=pi05["manifest"].get("sha256", ""),
        )
        test_units = {
            key: indices for key, indices in units.items()
            if int(folds[indices[0]]) == fold
        }
        for key, indices in test_units.items():
            unit_scores[key] = _unit_scores_pairwise(model, feature, indices, operators)
            unit_success[key] = {str(operators[i]): float(success[i]) for i in indices}
    report["in_policy"] = _evaluate(unit_scores, unit_success, task_of_unit)

    # ---- Layer 2: cross-policy transfer (pi0-fast model on pi0.5 rows) ----
    k5_units: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(k5["rows"]):
        k5_units[(str(row["root_id"]), int(row["replica"]))].append(i)
    pairs_a: list[np.ndarray] = []
    pairs_b: list[np.ndarray] = []
    deltas: list[float] = []
    for indices in k5_units.values():
        for a_index in range(len(indices)):
            for b_index in range(a_index + 1, len(indices)):
                ia, ib = indices[a_index], indices[b_index]
                pairs_a.append(k5["feature"][ia])
                pairs_b.append(k5["feature"][ib])
                deltas.append(float(k5["success"][ia] - k5["success"][ib]))
    transfer_model = fit_pairwise(
        np.stack(pairs_a), np.stack(pairs_b), np.asarray(deltas, dtype=np.float64),
        alpha=args.alpha, feature_version=k5["manifest"].get("feature_version", "v1"),
        training_manifest_sha256=k5["manifest"].get("sha256", ""),
    )
    transfer_scores: dict[tuple[str, int], dict[str, float]] = {}
    for key, indices in units.items():
        transfer_scores[key] = _unit_scores_pairwise(transfer_model, feature, indices, operators)
    report["cross_policy_transfer"] = _evaluate(transfer_scores, unit_success, task_of_unit)

    # ---- gap summary -------------------------------------------------------
    in_policy_acc = report["in_policy"]["pairwise_accuracy"]
    transfer_acc = report["cross_policy_transfer"]["pairwise_accuracy"]
    report["transfer_gap_vs_in_policy"] = (
        round(transfer_acc - in_policy_acc, 5) if in_policy_acc is not None else None
    )
    report["policy_signal_verdict"] = (
        "PASS" if in_policy_acc is not None and in_policy_acc > 0.5 else "FAIL"
    )
    report["transfer_verdict"] = (
        "PASS" if transfer_acc is not None and transfer_acc > 0.5 else "FAIL"
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""K3 one-shot external confirmation of the frozen K5 selector (v2 plan §8).

The model is trained once on the full frozen K5 dataset (raw-action features,
pairwise family, alpha and abstain margin frozen from the K5 nested OOF).  K3
metrics are read exactly once and written to a confirmation report; nothing is
re-selected after seeing K3 numbers.
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

from rase.vnext.selector import (  # noqa: E402
    fit_pairwise,
    fit_ridge,
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
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--confirm-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="raw-action")
    parser.add_argument("--model-family", default="pairwise")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()

    def _load(dataset_path: Path) -> dict[str, Any]:
        manifest = json.loads(dataset_path.read_text())
        if manifest.get("status") != "frozen":
            raise SystemExit(f"{dataset_path} is not frozen")
        arrays = np.load(Path(str(manifest["features_path"])), allow_pickle=False)
        key_map = {"raw-action": "raw", "trace-only": "trace", "semantic": "semantic"}
        feature_key = key_map.get(args.feature, args.feature)
        rows = manifest["rows"]
        return {
            "manifest": manifest,
            "feature": arrays[feature_key],
            "rows": rows,
            "success": np.asarray([bool(row["success"]) for row in rows], dtype=np.float64),
            "costs": np.asarray([
                float(row.get("query_cost") or 0.0)
                + float(row.get("fallback_cost") or 0.0)
                + float(row.get("latency_cost") or 0.0)
                for row in rows
            ], dtype=np.float64),
        }

    train = _load(args.train_dataset)
    confirm = _load(args.confirm_dataset)
    feature_version = train["manifest"]["feature_version"]

    # Train once on the full K5 dataset.
    n_train = len(train["rows"])
    if args.model_family == "pairwise":
        units: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, row in enumerate(train["rows"]):
            units[(str(row["root_id"]), int(row["replica"]))].append(i)
        pairs_a: list[np.ndarray] = []
        pairs_b: list[np.ndarray] = []
        deltas: list[float] = []
        for indices in units.values():
            for a_index in range(len(indices)):
                for b_index in range(a_index + 1, len(indices)):
                    ia, ib = indices[a_index], indices[b_index]
                    pairs_a.append(train["feature"][ia])
                    pairs_b.append(train["feature"][ib])
                    deltas.append(float(train["success"][ia] - train["success"][ib]))
        model = fit_pairwise(
            np.stack(pairs_a), np.stack(pairs_b), np.asarray(deltas, dtype=np.float64),
            alpha=args.alpha, feature_version=feature_version,
            training_manifest_sha256=hashlib.sha256(args.train_dataset.read_bytes()).hexdigest(),
        )
    else:
        model = fit_ridge(
            train["feature"], train["success"], alpha=args.alpha,
            model_type="candidate", feature_version=feature_version,
            training_manifest_sha256=hashlib.sha256(args.train_dataset.read_bytes()).hexdigest(),
        )

    # One-shot evaluation on K3.
    rows = confirm["rows"]
    feature = confirm["feature"]
    success = confirm["success"]
    costs = confirm["costs"]
    n = len(rows)
    units: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        units[(str(row["root_id"]), int(row["replica"]))].append(i)

    correct = total = 0
    per_task: dict[str, list[float]] = defaultdict(list)
    selector_values: dict[float, list[float]] = defaultdict(list)
    continue_values: dict[float, list[float]] = defaultdict(list)
    fallback_values: dict[float, list[float]] = defaultdict(list)
    oracle_values: dict[float, list[float]] = defaultdict(list)
    chosen_counter: dict[str, int] = defaultdict(int)
    abstained = 0
    layered: dict[str, list[bool]] = defaultdict(list)
    lambdas = (0.05, 0.1, 0.2, 0.5, 1.0)
    for unit, indices in units.items():
        scores: dict[str, float] = {}
        truth: dict[str, float] = {}
        unit_costs: dict[str, float] = {}
        cont_index = next(
            (i for i in indices if str(rows[i]["operator"]) == "continue.source"), None,
        )
        for i in indices:
            operator = str(rows[i]["operator"])
            truth[operator] = float(success[i])
            unit_costs[operator] = float(costs[i])
            if args.model_family == "pairwise":
                if operator == "continue.source":
                    scores[operator] = 0.5
                elif cont_index is not None:
                    from rase.vnext.selector import predict_pairwise

                    scores[operator] = float(predict_pairwise(
                        model, feature[i:i + 1], feature[cont_index:cont_index + 1],
                    )[0])
                else:
                    scores[operator] = float(model.predict(feature[i:i + 1])[0])
            else:
                scores[operator] = float(model.predict(feature[i:i + 1])[0])
        decision = select_candidates(scores, abstain_margin=args.margin)
        chosen_counter[decision.chosen_operator] += 1
        abstained += int(decision.abstained)
        task = str(rows[indices[0]]["task_id"])
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
                for pair in (
                    ("continue.source", "requery.source"),
                    ("continue.source", "fallback.persistent"),
                    ("requery.source", "fallback.persistent"),
                ):
                    if {left, right} == set(pair) and left in truth and right in truth:
                        layered["|".join(pair)].append(bool(predicted * actual > 0))
        if unit_total:
            correct += unit_correct
            total += unit_total
            per_task[task].append(unit_correct / unit_total)
        for lam in lambdas:
            chosen_op = decision.chosen_operator
            selector_values[lam].append(utility_lambda(truth[chosen_op], unit_costs[chosen_op], lam))
            continue_values[lam].append(utility_lambda(
                truth.get("continue.source", 0.0), unit_costs.get("continue.source", 0.0), lam,
            ))
            fallback_values[lam].append(utility_lambda(
                truth.get("fallback.persistent", 0.0),
                unit_costs.get("fallback.persistent", 0.0), lam,
            ))
            oracle_values[lam].append(max(
                utility_lambda(truth[op], unit_costs[op], lam) for op in ops
            ))
    task_values = np.asarray([float(np.mean(per_task[task])) for task in per_task])
    rng = np.random.default_rng(20270818)
    indices_boot = rng.integers(0, len(task_values), size=(2000, len(task_values)))
    samples = task_values[indices_boot].mean(axis=1)
    report = {
        "schema_version": "rase-vnext-selector-k3-confirm/v1",
        "one_shot": True,
        "train_dataset_sha256": hashlib.sha256(args.train_dataset.read_bytes()).hexdigest(),
        "confirm_dataset_sha256": hashlib.sha256(args.confirm_dataset.read_bytes()).hexdigest(),
        "feature": args.feature,
        "model_family": args.model_family,
        "alpha": args.alpha,
        "abstain_margin": args.margin,
        "units": len(units),
        "pairwise_accuracy": round(correct / total, 5) if total else None,
        "pairs": total,
        "task_bootstrap": {
            "mean": round(float(task_values.mean()), 5),
            "ci95": [round(float(np.quantile(samples, 0.025)), 5),
                     round(float(np.quantile(samples, 0.975)), 5)],
            "tasks": len(per_task),
            "positive_tasks": int((task_values > 0.5).sum()),
        },
        "layered": {k: round(float(np.mean(v)), 4) for k, v in layered.items()},
        "selector_operator_distribution": dict(chosen_counter),
        "abstain_rate": round(abstained / len(units), 4),
        "curves": {
            str(lam): {
                "selector": round(float(np.mean(selector_values[lam])), 5),
                "continue": round(float(np.mean(continue_values[lam])), 5),
                "fallback": round(float(np.mean(fallback_values[lam])), 5),
                "oracle": round(float(np.mean(oracle_values[lam])), 5),
            }
            for lam in lambdas
        },
        "break_even_lambda_ge_fallback": [
            lam for lam in lambdas
            if float(np.mean(selector_values[lam])) >= float(np.mean(fallback_values[lam]))
        ],
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

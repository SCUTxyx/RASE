#!/usr/bin/env python3
"""Frozen-prediction diagnostics for a completed R7-A five-seed OOF.

This audit is descriptive only.  It may localize a failure but cannot change
the formal stability gate or unlock a new model stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (2026081207, 2026081208, 2026081209, 2026081210, 2026081211)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive, negative = scores[labels], scores[~labels]
    return float(((positive[:, None] > negative[None, :]).mean()
                  + 0.5 * (positive[:, None] == negative[None, :]).mean()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def ece(labels: np.ndarray, probability: np.ndarray) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        mask = (probability >= lower) & ((probability < upper) if upper < 1.0
                                        else (probability <= upper))
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean())
                                               - float(probability[mask].mean()))
    return result


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    prevalence = float(labels.mean())
    ap = average_precision(labels, probability)
    return {"rows": int(len(labels)), "prevalence": prevalence,
            "auroc": auc(labels, probability), "average_precision": ap,
            "ap_above_prevalence": ap - prevalence,
            "ece_10_equal_width": ece(labels, probability)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stability_path = args.input_root / "stability.json"
    stability = json.loads(stability_path.read_text())
    reports, predictions = [], []
    for seed in SEEDS:
        report_path = args.input_root / f"seed_{seed}" / "report.json"
        prediction_path = report_path.with_suffix(".predictions.npz")
        report = json.loads(report_path.read_text())
        if int(report.get("seed", -1)) != seed:
            raise ValueError(f"seed mismatch for {report_path}")
        with np.load(prediction_path, allow_pickle=False) as loaded:
            predictions.append({key: loaded[key].copy() for key in loaded.files})
        reports.append(report)
    dataset_hashes = {report["dataset_sha256"] for report in reports}
    if len(dataset_hashes) != 1:
        raise ValueError("OOF reports do not bind one dataset")
    reference = predictions[0]
    for row in predictions[1:]:
        for key in ("state_key", "task_id", "suite", "source_failure"):
            if not np.array_equal(reference[key], row[key]):
                raise ValueError(f"prediction alignment mismatch: {key}")
    labels = reference["source_failure"].astype(np.float64)
    suite = reference["suite"]
    matrix = np.stack([row["calibrated_oof_probability"] for row in predictions])
    ensemble = matrix.mean(axis=0)
    weighted_fold_metrics = []
    for seed, report in zip(SEEDS, reports):
        folds = report["fold_reports"]
        weights = np.asarray([row["validation_rows"] for row in folds], dtype=np.float64)
        weighted_fold_metrics.append({
            "seed": seed,
            **{name: float(np.average(
                [row["validation_metrics"][name] for row in folds], weights=weights
            )) for name in ("auroc", "average_precision", "ap_above_prevalence",
                            "ece_10_equal_width", "brier")},
        })
    result = {
        "schema_version": "rase-r7a-oof-diagnostics/v1",
        "status": "DESCRIPTIVE_ONLY",
        "formal_stability_status": stability["status"],
        "formal_decision": stability["decision"],
        "formal_stability": str(stability_path.resolve()),
        "formal_stability_sha256": sha256(stability_path),
        "dataset_sha256": next(iter(dataset_hashes)),
        "five_seed_probability_ensemble": metrics(labels, ensemble),
        "ensemble_by_suite": {
            str(name): metrics(labels[suite == name], ensemble[suite == name])
            for name in sorted(set(suite.tolist()))
        },
        "ensemble_without_long": metrics(labels[suite != "Long"],
                                           ensemble[suite != "Long"]),
        "weighted_fold_metrics_by_seed": weighted_fold_metrics,
        "seed_probability_correlation": np.corrcoef(matrix).tolist(),
        "mean_per_state_seed_std": float(matrix.std(axis=0).mean()),
        "p95_per_state_seed_std": float(np.quantile(matrix.std(axis=0), 0.95)),
        "interpretation": [
            "some task-held-out ranking signal exists, strongest in Object",
            "Long remains below random under the five-seed ensemble",
            "fold-specific calibration scale and ECE are unstable",
            "these diagnostics do not modify the 0/5 formal gate",
        ],
        "unlocks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

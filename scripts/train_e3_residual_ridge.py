#!/usr/bin/env python3
"""Train a tiny group-CV residual chunk regressor for the E3 mechanism test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def image_grid_features(images: np.ndarray, cells: int = 6) -> np.ndarray:
    array = np.asarray(images, dtype=np.float32) / 255.0
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [N,H,W,3] images, got {array.shape}")
    n, height, width, channels = array.shape
    if height % cells or width % cells:
        raise ValueError("image dimensions must be divisible by spatial cells")
    pooled = array.reshape(
        n, cells, height // cells, cells, width // cells, channels
    ).mean(axis=(2, 4))
    return pooled.reshape(n, -1)


def build_features(data: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(data["proprio"], dtype=np.float32),
            np.asarray(data["source_action"], dtype=np.float32).reshape(len(data["proprio"]), -1),
            np.asarray(data["language_hash"], dtype=np.float32),
        ],
        axis=1,
    )
    if variant == "state":
        return state
    if variant == "state_vision":
        return np.concatenate(
            [state, image_grid_features(data["agentview"]), image_grid_features(data["wrist"])],
            axis=1,
        )
    raise ValueError(f"unknown feature variant: {variant}")


def group_folds(groups: Sequence[str], n_splits: int = 5) -> list[np.ndarray]:
    unique_counts = Counter(str(value) for value in groups)
    if len(unique_counts) < n_splits:
        raise ValueError("fewer semantic groups than CV folds")
    ordered = sorted(
        unique_counts,
        key=lambda group: (
            -unique_counts[group],
            hashlib.sha256(group.encode()).hexdigest(),
        ),
    )
    assignments: list[list[str]] = [[] for _ in range(n_splits)]
    loads = [0] * n_splits
    for group in ordered:
        fold = min(range(n_splits), key=lambda index: (loads[index], index))
        assignments[fold].append(group)
        loads[fold] += unique_counts[group]
    values = np.asarray([str(value) for value in groups])
    return [np.isin(values, fold_groups) for fold_groups in assignments]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    xs = (x - mean) / scale
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    # Dual solve is stable and cheap for N << D.
    kernel = xs @ xs.T
    dual = np.linalg.solve(kernel + float(alpha) * np.eye(len(xs)), yc)
    weight = xs.T @ dual
    return {
        "x_mean": mean.astype(np.float32),
        "x_scale": scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "weight": weight.astype(np.float32),
    }


def predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float32) - model["x_mean"]) / model["x_scale"]
    return xs @ model["weight"] + model["y_mean"]


def metrics(y: np.ndarray, prediction: np.ndarray, correction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(y)
    result = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "predicted_delta_abs_mean": float(np.mean(np.abs(prediction))),
    }
    for label, mask in (("correction", correction), ("identity", ~correction)):
        if mask.any():
            result[f"{label}_mse"] = float(np.mean(error[mask] ** 2))
            result[f"{label}_predicted_delta_abs_mean"] = float(
                np.mean(np.abs(prediction[mask]))
            )
    return result


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    with np.load(args.data.resolve(), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    y = np.asarray(data["delta_target"], dtype=np.float32).reshape(len(data["state_key"]), -1)
    correction = np.asarray(data["supervision"]) == "successful_exact_root_oft_prefix"
    folds = group_folds(data["group_id"].tolist(), args.folds)
    candidates: list[dict[str, Any]] = []
    for variant in ("state", "state_vision"):
        x = build_features(data, variant)
        for alpha in (0.1, 1.0, 10.0, 100.0):
            oof = np.zeros_like(y)
            fold_rows = []
            for fold_index, validation in enumerate(folds):
                training = ~validation
                model = fit_ridge(x[training], y[training], alpha)
                oof[validation] = predict(model, x[validation])
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "train_examples": int(training.sum()),
                        "validation_examples": int(validation.sum()),
                        "validation_groups": sorted(set(data["group_id"][validation].tolist())),
                    }
                )
            row_metrics = metrics(y, oof, correction)
            candidates.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "input_dim": int(x.shape[1]),
                    "metrics": row_metrics,
                    "folds": fold_rows,
                    "oof_prediction": oof,
                }
            )

    selected = min(candidates, key=lambda row: (row["metrics"]["mse"], row["input_dim"], row["alpha"]))
    selected_x = build_features(data, selected["variant"])
    final_model = fit_ridge(selected_x, y, selected["alpha"])
    final_prediction = predict(final_model, selected_x)
    zero_prediction = np.zeros_like(y)
    zero_metrics = metrics(y, zero_prediction, correction)
    selected_metrics = dict(selected["metrics"])
    checks = {
        "group_cv_correction_mse_improves_zero_by_10pct": (
            selected_metrics["correction_mse"] <= 0.90 * zero_metrics["correction_mse"]
        ),
        "group_cv_identity_mean_abs_delta_at_most_0_15": (
            selected_metrics["identity_predicted_delta_abs_mean"] <= 0.15
        ),
        "final_fit_finite": bool(np.isfinite(final_prediction).all()),
    }
    dataset_sha256 = hashlib.sha256(args.data.resolve().read_bytes()).hexdigest()
    atomic_save_npz(
        args.model_output.resolve(),
        **final_model,
        feature_variant=np.asarray(selected["variant"]),
        alpha=np.asarray(selected["alpha"], dtype=np.float32),
        horizon=np.asarray(data["source_action"].shape[1], dtype=np.int64),
        action_dim=np.asarray(data["source_action"].shape[2], dtype=np.int64),
        dataset_sha256=np.asarray(dataset_sha256),
    )
    report = {
        "schema_version": "rase-e3-residual-ridge-training/v1",
        "status": "complete",
        "scientific_scope": "development_group_cv_model_selection_not_system_eligibility",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "dataset": str(args.data.resolve()),
        "dataset_sha256": dataset_sha256,
        "n_examples": len(y),
        "n_groups": len(set(data["group_id"].tolist())),
        "n_correction": int(correction.sum()),
        "n_identity": int((~correction).sum()),
        "selected": {
            "variant": selected["variant"],
            "alpha": selected["alpha"],
            "input_dim": selected["input_dim"],
            "parameter_count": int(selected["input_dim"] * y.shape[1] + y.shape[1]),
            "group_cv_metrics": selected_metrics,
            "final_fit_metrics": metrics(y, final_prediction, correction),
        },
        "zero_residual_baseline": zero_metrics,
        "checks": checks,
        "candidates": [
            {
                "variant": row["variant"],
                "alpha": row["alpha"],
                "input_dim": row["input_dim"],
                "metrics": row["metrics"],
                "folds": row["folds"],
            }
            for row in candidates
        ],
        "claim_boundary": (
            "PASS only permits exact-root rollout of frozen predictions on an independent cohort; "
            "it is not task-success evidence."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], **report["selected"]}, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

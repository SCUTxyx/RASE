#!/usr/bin/env python3
"""Task-held-out diagnostic baselines for one-step action-conditioned dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_r4_safe_handback_world_model import build_arrays, grouped_task_folds, read_jsonl


class Ridge:
    """Small dependency-free multi-output ridge regressor."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        xc, yc = x - self.x_mean, y - self.y_mean
        if xc.shape[1] <= xc.shape[0]:
            gram = xc.T @ xc
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            self.coef = np.linalg.solve(gram, xc.T @ yc)
        else:
            gram = xc @ xc.T
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            self.coef = xc.T @ np.linalg.solve(gram, yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.x_mean) @ self.coef + self.y_mean


def expand(data: dict[str, np.ndarray], interactions: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, arms, action_dim = data["transition"].shape
    state = np.repeat(data["state"], arms, axis=0)
    action = data["transition"].reshape(n * arms, action_dim)
    features = [state, action]
    if interactions:
        latent_dim = data["delta"].shape[-1]
        latent = state[:, :latent_dim]
        features.append((latent[:, :, None] * action[:, None, :]).reshape(n * arms, -1))
    target = data["delta"].reshape(n * arms, -1)
    keep = (1.0 - data["terminal"].reshape(-1)).astype(bool)
    return np.concatenate(features, axis=1), target, keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset)
    folds = grouped_task_folds(rows, args.folds)
    results = []
    for interactions in (False, True):
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            squared_error = baseline_error = elements = 0.0
            for fold in folds:
                train, stats = build_arrays(fold["train"])
                val, _ = build_arrays(fold["val"], stats)
                x_train, y_train, keep_train = expand(train, interactions)
                x_val, y_val, keep_val = expand(val, interactions)
                model = Ridge(alpha=alpha).fit(x_train[keep_train], y_train[keep_train])
                prediction = model.predict(x_val[keep_val])
                squared_error += float(((prediction - y_val[keep_val]) ** 2).sum())
                baseline_error += float((y_val[keep_val] ** 2).sum())
                elements += float(y_val[keep_val].size)
            results.append({
                "model": "ridge_with_latent_action_interactions" if interactions else "ridge",
                "alpha": alpha,
                "mse": squared_error / elements,
                "persistence_mse": baseline_error / elements,
                "improvement": 1.0 - squared_error / baseline_error,
            })
    report = {
        "schema_version": "rase-pre-c0-r4-dynamics-probe/v1",
        "evaluation": "logical-task-held-out",
        "n_folds": len(folds),
        "results": sorted(results, key=lambda row: row["mse"]),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

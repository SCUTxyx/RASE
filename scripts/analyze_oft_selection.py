#!/usr/bin/env python3
"""Episode-level selection demo on the OFT model-pair matrix.

Given the per-task success matrix (analyze_oft_matrix.py output), estimate how
well an instruction-based selector can capture the comparative advantage:

  - oracle: pick the better model per task (upper bound)
  - single-best: always use the model with higher mean success
  - majority: always pick the model that wins more tasks
  - learned LOO classifier: leave-one-task-out linear classifier over
    instruction character-bigram features, labels = better model per task;
    then closed-loop success = mean over tasks of the chosen model's rate.
  - abstain variant: when the classifier is unsure (margin < thr), use
    single-best.

Also reports the E2 opportunity quantities (headroom, hetero rate).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np


def bigram_features(text: str, vocab: dict[str, int]) -> np.ndarray:
    text = text.lower()
    grams = [text[i:i + 2] for i in range(len(text) - 1)]
    x = np.zeros(len(vocab), dtype=np.float64)
    for g in grams:
        idx = vocab.get(g)
        if idx is not None:
            x[idx] += 1.0
    return x


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in texts:
        text = text.lower()
        for i in range(len(text) - 1):
            g = text[i:i + 2]
            vocab.setdefault(g, len(vocab))
    return vocab


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = (X - mean) / scale
    design = np.column_stack((np.ones(len(Xs)), Xs))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    y_mean = float(y.mean())
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ (y - y_mean))
    return np.concatenate([[y_mean + beta[0]], beta[1:]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-a", default="oft_spatial")
    parser.add_argument("--model-b", default="oft_object")
    parser.add_argument("--abstain-threshold", type=float, default=0.1,
                        help="margin below which the LOO classifier falls back to single-best")
    args = parser.parse_args()

    data = json.loads(args.matrix.read_text())
    tasks = [t for t in data["per_task"] if t["A"] is not None and t["B"] is not None]
    if len(tasks) < 4:
        print("need at least 4 matched tasks; got", len(tasks))
        return 1
    n = len(tasks)
    A = np.array([t["A"] for t in tasks])
    B = np.array([t["B"] for t in tasks])
    oracle = np.maximum(A, B)
    single_best = max(A.mean(), B.mean())
    oracle_mean = oracle.mean()

    # LOO classifier over instruction bigrams, label = argmax per task
    texts = [t["task"] for t in tasks]
    vocab = build_vocab(texts)
    X = np.stack([bigram_features(t, vocab) for t in texts])
    y = (B > A).astype(int)  # 1 -> choose B
    if y.std() == 0:
        print("no task-level disagreement; LOO classifier not meaningful")
        return 1

    preds = np.zeros(n)
    margins = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = ridge_fit(X[mask], y[mask])
        scores = 1.0 / (1.0 + np.exp(-(beta[0] + X[i] @ beta[1:])))
        preds[i] = scores
        margins[i] = abs(scores - 0.5)

    choose_b = preds > 0.5
    chosen_rate = np.where(choose_b, B, A)
    # abstain: below margin -> single-best model
    abstain_mask = margins < args.abstain_threshold
    base_model_b = y.sum() > n - y.sum()  # majority better model
    rate_abstain = np.where(
        abstain_mask,
        B if base_model_b else A,
        chosen_rate,
    )

    report = {
        "n_tasks": n,
        "mean_A": float(A.mean()),
        "mean_B": float(B.mean()),
        "single_best_mean": float(single_best),
        "oracle_mean": float(oracle_mean),
        "oracle_headroom_pp": float((oracle_mean - single_best) * 100),
        "hetero_rate_ge_0.2": float(np.mean(np.abs(A - B) >= 0.2)),
        "loo_classifier_accuracy": float(np.mean((preds > 0.5) == y)),
        "loo_selection_mean": float(chosen_rate.mean()),
        "loo_abstain_mean": float(rate_abstain.mean()),
        "abstain_fraction": float(abstain_mask.mean()),
        "gain_vs_single_best_pp": float((rate_abstain.mean() - single_best) * 100),
        "per_task": [
            {"task": t["task"], "suite": t["suite"], "A": t["A"], "B": t["B"],
             "oracle": float(max(t["A"], t["B"])),
             "chosen": float(chosen_rate[i]),
             "loo_choose_b": bool(choose_b[i]), "margin": float(margins[i])}
            for i, t in enumerate(tasks)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase W2: learned future predictor (high-dim state version).

WM:  (s_t proprio, canonical chunk, bigram) -> predicted future summary
     (proprio trajectory stats 16-d + object trajectory stats 6-d)

Pipeline per LOVO fold (hold out one VLA):
  1. train WM (SmallMLP) on the other VLAs to predict future summary;
  2. risk ridge: B1 (direct) vs B2_learned (B1 + predicted future),
     evaluate on the held-out VLA;
  3. also report WM prediction error and B2_oracle reference
     (from W1: real-future ceiling) for the same fold.

Gate W2:
  - learned future L1 discrimination AUROC >= 0.8 in >= 2/3 folds, AND
  - B2_learned > B1 in >= 2/3 folds (or at least not worse than B1
    while oracle shows headroom) -> proceed to W3 (drift detection).

Usage (server, oft env):
  python train_future_predictor.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --output runs/oft_opportunity/w2_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase_common import (
    canonical_chunk_features, fit_ridge, ridge_predict, auroc,
    build_bigram_vocab, bigram_features,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def future_summary_proprio(fut: list[list[float]]) -> np.ndarray:
    a = np.asarray(fut, dtype=np.float64)
    if len(a) == 0:
        return np.zeros(16)
    end = a[-1]
    start = a[0]
    disp = end[:3] - start[:3]
    speed = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1) if len(a) > 1 else [0]
    return np.concatenate([
        end, disp, np.array([float(end[7] - start[7])]),
        [float(np.mean(speed)), float(np.max(speed))],
        [float(a[:, :3].std(axis=0).mean())], [float(len(a))],
    ])


def object_traj_features(rows_objects: list) -> np.ndarray:
    out = np.zeros(6)
    if not rows_objects:
        return out
    try:
        s_t = {n: np.asarray(p, dtype=np.float64) for n, p in rows_objects[0]}
        shifts = []
        for snap in rows_objects:
            d = {n: np.asarray(p, dtype=np.float64) for n, p in snap}
            common = set(s_t) & set(d)
            if common:
                shifts.append(np.array([np.linalg.norm(d[n] - s_t[n])
                                        for n in common]))
        if shifts:
            S = np.stack(shifts)
            ms, mx = S.mean(axis=1), S.max(axis=1)
            out = np.array([ms.mean(), ms.max(), ms[-1],
                            mx.mean(), mx.max(), mx[-1]])
    except Exception:
        pass
    return out


class SmallMLP:
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 128,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, 0.1, (d_hidden, d_hidden))
        self.b2 = np.zeros(d_hidden)
        self.W3 = rng.normal(0, 0.1, (d_hidden, d_out))
        self.b3 = np.zeros(d_out)
        self.mean = np.zeros(d_in)
        self.scale = np.ones(d_in)
        self.ym = np.zeros(d_out)
        self.ys = np.ones(d_out)

    def _std(self, X):
        self.mean = X.mean(0)
        self.scale = X.std(0) + 1e-8
        return (X - self.mean) / self.scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def fit(self, X: np.ndarray, Y: np.ndarray, steps: int = 2000,
            lr: float = 1e-2) -> float:
        Xs = self._std(X)
        self.ym = Y.mean(0)
        self.ys = Y.std(0) + 1e-8
        Ys = (Y - self.ym) / self.ys
        for _ in range(steps):
            h1 = np.tanh(Xs @ self.W1 + self.b1)
            h2 = np.tanh(h1 @ self.W2 + self.b2)
            out = h2 @ self.W3 + self.b3
            dout = (out - Ys) / len(Xs)
            dW3 = h2.T @ dout
            db3 = dout.sum(0)
            dh2 = dout @ self.W3.T * (1 - h2 ** 2)
            dW2 = h1.T @ dh2
            db2 = dh2.sum(0)
            dh1 = dh2 @ self.W2.T * (1 - h1 ** 2)
            dW1 = Xs.T @ dh1
            db1 = dh1.sum(0)
            for p, g in [(self.W3, dW3), (self.b3, db3), (self.W2, dW2),
                         (self.b2, db2), (self.W1, dW1), (self.b1, db1)]:
                p -= lr * g
        return float(np.sqrt(np.mean((self.forward(Xs) - Ys) ** 2)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.scale
        return self.forward(Xs) * self.ys + self.ym


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    rows = load_rows(args.data)
    if not rows:
        print("no rows")
        return 1
    vlas = sorted({r["model"] for r in rows})
    vocab = build_bigram_vocab([r["task"] for r in rows])
    labels = np.array([r.get("consequence_label", r.get("displacement", 0.0))
                       for r in rows])
    yb = (labels > np.median(labels)).astype(float) \
        if set(np.unique(labels)) - {0.0, 1.0} else labels

    X_state = np.stack([np.asarray(r["s_t_proprio"], dtype=np.float64)
                        for r in rows])
    X_chunk = np.stack([canonical_chunk_features(
        np.asarray(r["chunk_raw"], dtype=np.float64).reshape(-1, 7))
        for r in rows])
    X_bigram = np.stack([bigram_features(r["task"], vocab) for r in rows])
    X_B1 = np.concatenate([X_state, X_chunk, X_bigram], axis=1)
    def sample_traj(fut: list[list[float]], k: int = 8) -> np.ndarray:
        a = np.asarray(fut, dtype=np.float64)
        if len(a) == 0:
            return np.zeros(k * 8)
        idx = np.linspace(0, len(a) - 1, k).astype(int)
        return a[idx].reshape(-1)

    Y_future = np.stack([
        np.concatenate([
            sample_traj(r["future_proprio"], k=8),
            object_traj_features(r.get("future_objects", [])),
        ]) for r in rows])

    folds: dict[str, dict] = {}
    for held in vlas:
        tr = [i for i, r in enumerate(rows) if r["model"] != held]
        te = [i for i, r in enumerate(rows) if r["model"] == held]
        if len(tr) < 20 or len(te) < 5:
            continue
        # --- train WM on train VLAs, predict future on all ---
        wm = SmallMLP(X_B1.shape[1], Y_future.shape[1], d_hidden=128, seed=0)
        rmse = wm.fit(X_B1[tr], Y_future[tr], steps=args.steps)
        F_hat = wm.predict(X_B1)
        # oracle future (real) for reference
        F_real = Y_future
        # --- risk models ---
        X_B2l = np.concatenate([X_B1, F_hat], axis=1)
        X_B2r = np.concatenate([X_B1, F_real], axis=1)
        p1 = ridge_predict(X_B1[te], *fit_ridge(X_B1[tr], yb[tr],
                                                alpha=args.alpha))
        p2l = ridge_predict(X_B2l[te], *fit_ridge(X_B2l[tr], yb[tr],
                                                  alpha=args.alpha))
        p2r = ridge_predict(X_B2r[te], *fit_ridge(X_B2r[tr], yb[tr],
                                                  alpha=args.alpha))
        u1 = auroc(p1, yb[te])
        u2l = auroc(p2l, yb[te])
        u2r = auroc(p2r, yb[te])
        # prediction error of the future summary (normalized RMSE)
        folds[held] = {
            "B1_lovo_auroc": round(u1, 4),
            "B2_learned_lovo_auroc": round(u2l, 4),
            "B2_oracle_lovo_auroc": round(u2r, 4),
            "delta_learned_vs_B1": round(u2l - u1, 4),
            "delta_oracle_vs_B1": round(u2r - u1, 4),
            "wm_normalized_rmse": round(rmse, 4),
        }
        print(f"LOVO {held}: B1={u1:.3f} B2learned={u2l:.3f} "
              f"(delta{u2l - u1:+.3f}) B2oracle={u2r:.3f} "
              f"(delta{u2r - u1:+.3f}) wm_rmse={rmse:.3f}")

    n_learned_ge_08 = sum(1 for f in folds.values()
                          if f["B2_learned_lovo_auroc"] >= 0.8)
    n_learned_gt_b1 = sum(1 for f in folds.values()
                          if f["delta_learned_vs_B1"] > 0.02)
    n_oracle_gt_b1 = sum(1 for f in folds.values()
                         if f["delta_oracle_vs_B1"] > 0.02)
    verdict = "PASS" if (n_learned_ge_08 >= 2 and n_learned_gt_b1 >= 2) else (
        "PARTIAL" if n_oracle_gt_b1 >= 2 else "FAIL")
    report = {
        "schema": "rase-w2-future-predictor/v1",
        "n_rows": len(rows),
        "vlas": vlas,
        "folds": folds,
        "gate_w2": {
            "learned_auroc_ge_08_folds": n_learned_ge_08,
            "learned_gt_B1_folds": n_learned_gt_b1,
            "oracle_gt_B1_folds": n_oracle_gt_b1,
            "verdict": verdict,
            "note": "PASS -> W3 drift detection; PARTIAL -> improve WM "
                    "(capacity/features) once; FAIL -> lightweight route",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

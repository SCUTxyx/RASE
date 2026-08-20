#!/usr/bin/env python3
"""Phase 4: World Model MVP — does modeling future consequence improve
candidate ranking?  (roadmap §11, Gate D)

Three baselines on the same-root dataset:
  B0 Action Statistics : raw chunk statistics -> risk (the v3-style model)
  B1 Direct Risk       : (state, action) -> risk
  B2 Future-Bottleneck : (state, action) -> predicted future -> risk

Evaluation:
  - in-domain AUROC / pairwise ranking
  - LOVO: train on seen VLAs, report unseen-VLA AUROC + TransferRetention
  - identity probe per variant (predictive representation should be harder
    to use for VLA-ID than direct risk, while retaining risk signal)

B2 uses a small MLP (numpy, no torch dependency in the risk core; torch path
available on the server).  For the MVP the "world model" predicts the H-step
proprio future given (s_t, canonical chunk); the risk critic is a ridge over
the predicted future + (s_t, a) context.

Usage:
  python train_wm_mvp.py \
    --data runs/oft_opportunity/same_root_v1.jsonl \
    --output runs/oft_opportunity/wm_mvp_report.json \
    --lovo-test oft_goal
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rase_common import (
    canonical_chunk_features, raw_chunk_stats_feats, fit_ridge, ridge_predict,
    auroc, brier, ece, probe_identity, build_bigram_vocab, bigram_features,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class SmallMLP:
    """2-layer MLP (tanh) trained with a few Adam steps; pure numpy."""

    def __init__(self, d_in: int, d_hidden: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, 0.1, (d_hidden, d_hidden))
        self.b2 = np.zeros(d_hidden)
        self.W3 = rng.normal(0, 0.1, (d_hidden, 19))  # future_summary dim (16+3 obj)
        self.b3 = np.zeros(19)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def fit(self, X: np.ndarray, Y: np.ndarray, steps: int = 2000,
            lr: float = 1e-2) -> float:
        mu = X.mean(0); sd = X.std(0) + 1e-8
        Xs = (X - mu) / sd
        ym = Y.mean(0); ys = Y.std(0) + 1e-8
        Ys = (Y - ym) / ys
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
        # return normalized RMSE
        return float(np.sqrt(np.mean((self.forward(Xs) - Ys) ** 2)))


def future_summary(fut: list[list[float]], row: dict | None = None) -> np.ndarray:
    a = np.asarray(fut, dtype=np.float64)
    if len(a) == 0:
        base = np.zeros(8 * 4)
    else:
        end = a[-1]
        start = a[0]
        disp = end[:3] - start[:3]
        speed = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1) if len(a) > 1 else [0]
        base = np.concatenate([
            end,  # 8
            disp, np.array([float(end[7] - start[7])]),  # 4
            [float(np.mean(speed)), float(np.max(speed))],  # 2
            [float(a[:, :3].std(axis=0).mean())],  # 1
            [float(len(a))],  # 1
        ])  # 16
    # object-level future summary (task-sensitive, privileged)
    obj = np.zeros(3)
    if row is not None and row.get("s_t_objects") and row.get("s_th_objects"):
        da = {n: np.asarray(p, dtype=np.float64) for n, p in row["s_t_objects"]}
        db = {n: np.asarray(p, dtype=np.float64) for n, p in row["s_th_objects"]}
        common = set(da) & set(db)
        if common:
            shifts = np.array([np.linalg.norm(db[n] - da[n]) for n in common])
            obj = np.array([float(shifts.mean()), float(shifts.max()),
                            float(np.sum(shifts))])
    return np.concatenate([base, obj])  # 19


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lovo-test", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_rows(args.data)
    if not rows:
        print("no rows")
        return 1
    vlas = sorted({r["model"] for r in rows})
    vocab = build_bigram_vocab([r["task"] for r in rows])
    # NOTE: consequence_label (proprio displacement) is used until a
    # re-collection with correctly-timed s_t_objects is available (current
    # s_t_objects were snapshotted after candidate execution).
    labels = np.array([
        float(r.get("recovery_success", r.get("consequence_label",
                                              r.get("displacement", 0.0))))
        for r in rows])
    if set(np.unique(labels)) - {0.0, 1.0}:
        thr = np.median(labels)
        yb = (labels > thr).astype(float)
    else:
        yb = (labels > 0).astype(float)

    X_state = np.stack([np.asarray(r["s_t_proprio"], dtype=np.float64)
                        for r in rows])
    X_chunk_can = np.stack([canonical_chunk_features(
        np.asarray(r["chunk_raw"], dtype=np.float64).reshape(-1, 7))
        for r in rows])
    X_chunk_raw = np.stack([raw_chunk_stats_feats(
        np.asarray(r["chunk_raw"], dtype=np.float64).reshape(-1, 7))
        for r in rows])
    X_bigram = np.stack([bigram_features(r["task"], vocab) for r in rows])
    Y_future = np.stack([future_summary(r["future_proprio"], r) for r in rows])

    report: dict = {"rows": len(rows), "vlas": vlas, "baselines": {}}

    def evaluate(name: str, X: np.ndarray, probe_X: np.ndarray | None = None):
        entry = {"feat_dim": int(X.shape[1])}
        m_, s_, b_, ym_ = fit_ridge(X, yb, alpha=args.alpha)
        p = ridge_predict(X, m_, s_, b_, ym_)
        entry["in_domain_auroc"] = round(auroc(p, yb), 4)
        entry["in_domain_brier"] = round(brier(p, yb), 4)
        entry["in_domain_ece"] = round(ece(p, yb), 4)
        if args.lovo_test and args.lovo_test in vlas:
            tr = [i for i, r in enumerate(rows)
                  if r["model"] != args.lovo_test]
            te = [i for i, r in enumerate(rows)
                  if r["model"] == args.lovo_test]
            if len(tr) >= 20 and len(te) >= 5:
                m2, s2, b2, ym2 = fit_ridge(X[tr], yb[tr], alpha=args.alpha)
                p2 = ridge_predict(X[te], m2, s2, b2, ym2)
                a_te = auroc(p2, yb[te])
                a_tr = auroc(ridge_predict(X[tr], m2, s2, b2, ym2), yb[tr])
                entry[f"lovo_{args.lovo_test}_auroc"] = round(a_te, 4)
                entry[f"lovo_{args.lovo_test}_seen_auroc"] = round(a_tr, 4)
                entry[f"transfer_retention"] = round(
                    (a_te - 0.5) / max(1e-9, (a_tr - 0.5)), 4) \
                    if a_tr > 0.5 else None
        if probe_X is not None:
            feats_by_vla = {v: probe_X[[i for i, r in enumerate(rows)
                                        if r["model"] == v]] for v in vlas}
            pr = probe_identity(feats_by_vla, alpha=args.alpha)
            entry["identity_probe"] = {
                "in_domain_multiclass_accuracy":
                    pr["in_domain_multiclass_accuracy"],
                "random_baseline": pr["random_baseline"],
            }
        report["baselines"][name] = entry
        print(f"{name}: dim={entry['feat_dim']} "
              f"iid_auroc={entry['in_domain_auroc']} "
              f"lovo={entry.get(f'lovo_{args.lovo_test}_auroc')} "
              f"retention={entry.get('transfer_retention')}", flush=True)

    # B0: action statistics only (v3-style)
    evaluate("B0_action_stats", X_chunk_raw, probe_X=X_chunk_raw)

    # B1: direct state+action risk
    X_b1 = np.concatenate([X_state, X_chunk_can, X_bigram], axis=1)
    evaluate("B1_direct", X_b1, probe_X=X_b1)

    # B2: future-bottleneck risk — train WM (MLP) on a train split, predict
    # future for all rows, then ridge risk over (s_t, a, predicted future)
    wm = SmallMLP(X_b1.shape[1], d_hidden=64, seed=0)
    if args.lovo_test and args.lovo_test in vlas:
        tr = [i for i, r in enumerate(rows) if r["model"] != args.lovo_test]
        wm.fit(X_b1[tr], Y_future[tr], steps=300)
        pred_future = wm.forward(X_b1)
    else:
        wm.fit(X_b1, Y_future, steps=300)
        pred_future = wm.forward(X_b1)
    X_b2 = np.concatenate([X_state, X_chunk_can, pred_future], axis=1)
    evaluate("B2_future_bottleneck", X_b2, probe_X=X_b2)

    # Gate D verdict
    b1 = report["baselines"].get("B1_direct", {})
    b2 = report["baselines"].get("B2_future_bottleneck", {})
    key = f"lovo_{args.lovo_test}_auroc" if args.lovo_test else "in_domain_auroc"
    v1 = b1.get(key, 0.0)
    v2 = b2.get(key, 0.0)
    report["gate_d"] = {
        "criterion": key,
        "B1_value": v1, "B2_value": v2,
        "B2_beats_B1": bool(v2 > v1 + 0.02),
        "note": "requirement: B2 > B1 on held-out VLA, not only IID",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

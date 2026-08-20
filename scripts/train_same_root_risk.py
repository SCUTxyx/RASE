#!/usr/bin/env python3
"""Train the B1 (direct, identity-free) risk model on same-root data.

Feature layout (B1): [s_t proprio(8), canonical chunk(24), bigram(V)] -> risk.
Labels: consequence_label (proprio displacement), median-binarized.

Outputs:
  - full-data model:      <output>.npz           (seen-candidate arbitration)
  - per-VLA LOVO models:  <output>_lovo_<vla>.npz (unseen-candidate zero-shot)

Usage (server):
  python train_same_root_risk.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --output runs/oft_opportunity/same_root_risk.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase_common import (
    canonical_chunk_features, fit_ridge, build_bigram_vocab, bigram_features,
    auroc, ridge_predict,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
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
    X = np.concatenate([X_state, X_chunk, X_bigram], axis=1)
    print(f"rows={len(rows)} vlas={vlas} feat_dim={X.shape[1]} "
          f"pos_rate={yb.mean():.3f}")

    # full-data model
    mean, scale, beta, y_mean = fit_ridge(X, yb, alpha=args.alpha)
    p = ridge_predict(X, mean, scale, beta, y_mean)
    print(f"full-data AUROC: {auroc(p, yb):.4f}")
    np.savez_compressed(
        args.output,
        mean=mean, scale=scale, weights=beta[1:],
        intercept=y_mean + beta[0], alpha=args.alpha,
        model_type="candidate", feature_version="same-root-B1/v1",
    )
    with args.output.with_suffix(".vocab.json").open("w") as fh:
        json.dump(vocab, fh)

    # LOVO models + evaluation
    for held in vlas:
        tr = [i for i, r in enumerate(rows) if r["model"] != held]
        te = [i for i, r in enumerate(rows) if r["model"] == held]
        if len(tr) < 20 or len(te) < 5:
            continue
        m_, s_, b_, ym_ = fit_ridge(X[tr], yb[tr], alpha=args.alpha)
        p_te = ridge_predict(X[te], m_, s_, b_, ym_)
        p_tr = ridge_predict(X[tr], m_, s_, b_, ym_)
        u_te = auroc(p_te, yb[te])
        u_tr = auroc(p_tr, yb[tr])
        print(f"LOVO {held}: seen={u_tr:.4f} unseen={u_te:.4f} "
              f"retention={(u_te - 0.5) / max(1e-9, u_tr - 0.5):.3f}")
        out = args.output.with_name(
            f"{args.output.stem}_lovo_{held}.npz")
        np.savez_compressed(
            out,
            mean=m_, scale=s_, weights=b_[1:],
            intercept=ym_ + b_[0], alpha=args.alpha,
            model_type="candidate",
            feature_version="same-root-B1/v1",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

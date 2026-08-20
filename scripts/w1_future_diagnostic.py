#!/usr/bin/env python3
"""Phase W1b: real-future information-value diagnostic (option C).

B1      : [s_t proprio, canonical chunk, bigram] -> risk          (direct)
B2_real : B1 + [REAL future proprio trajectory stats + object pose
                trajectory stats] -> risk                          (future info)

The future is the GROUND-TRUTH trajectory from same-root collection (no WM
trained).  This measures the information-value ceiling of future knowledge:
if B2_real does not beat B1 on held-out VLAs, then even a perfect future
predictor (video/latent) cannot help transfer on this domain.

Verdict (Gate W1):
  PASS if B2_real > B1 + 0.02 in >= 2/3 LOVO folds -> proceed to W2 (learned WM)
  FAIL otherwise -> lightweight route (Phase 8 closed loop)

Usage (server, oft env):
  python w1_future_diagnostic.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --output runs/oft_opportunity/w1_diagnostic.json
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
        end,  # 8
        disp, np.array([float(end[7] - start[7])]),  # 4
        [float(np.mean(speed)), float(np.max(speed))],  # 2
        [float(a[:, :3].std(axis=0).mean())],  # 1
        [float(len(a))],  # 1
    ])  # 16


def object_traj_features(rows_objects: list) -> np.ndarray:
    """Summarize the object pose trajectory: per-timestep mean shift from
    s_t objects, plus final shift.  Returns 6 dims (mean/max/final of
    mean-shift and max-shift)."""
    out = np.zeros(6)
    if not rows_objects:
        return out
    try:
        s_t = {n: np.asarray(p, dtype=np.float64)
               for n, p in rows_objects[0]}
        shifts = []
        for snap in rows_objects:
            d = {n: np.asarray(p, dtype=np.float64) for n, p in snap}
            common = set(s_t) & set(d)
            if common:
                shifts.append(np.array([np.linalg.norm(d[n] - s_t[n])
                                        for n in common]))
        if shifts:
            S = np.stack(shifts)  # (T, n_obj)
            mean_shift = S.mean(axis=1)
            max_shift = S.max(axis=1)
            out = np.array([
                mean_shift.mean(), mean_shift.max(), mean_shift[-1],
                max_shift.mean(), max_shift.max(), max_shift[-1],
            ])
    except Exception:
        pass
    return out


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
    X_future = np.stack([future_summary_proprio(r["future_proprio"])
                         for r in rows])
    X_obj = np.stack([object_traj_features(r.get("future_objects", []))
                      for r in rows])

    X_B1 = np.concatenate([X_state, X_chunk, X_bigram], axis=1)
    X_B2 = np.concatenate([X_state, X_chunk, X_bigram, X_future, X_obj],
                          axis=1)
    print(f"rows={len(rows)} vlas={vlas} dim B1={X_B1.shape[1]} "
          f"B2={X_B2.shape[1]}")

    folds: dict[str, dict] = {}
    deltas = []
    for held in vlas:
        tr = [i for i, r in enumerate(rows) if r["model"] != held]
        te = [i for i, r in enumerate(rows) if r["model"] == held]
        if len(tr) < 20 or len(te) < 5:
            continue
        p1 = ridge_predict(X_B1[te], *fit_ridge(X_B1[tr], yb[tr],
                                                alpha=args.alpha))
        p2 = ridge_predict(X_B2[te], *fit_ridge(X_B2[tr], yb[tr],
                                                alpha=args.alpha))
        s1 = ridge_predict(X_B1[tr], *fit_ridge(X_B1[tr], yb[tr],
                                                alpha=args.alpha))
        s2 = ridge_predict(X_B2[tr], *fit_ridge(X_B2[tr], yb[tr],
                                                alpha=args.alpha))
        u1, u2 = auroc(p1, yb[te]), auroc(p2, yb[te])
        su1, su2 = auroc(s1, yb[tr]), auroc(s2, yb[tr])
        folds[held] = {
            "B1_lovo_auroc": round(u1, 4), "B2real_lovo_auroc": round(u2, 4),
            "B1_seen_auroc": round(su1, 4), "B2real_seen_auroc": round(su2, 4),
            "delta": round(u2 - u1, 4),
        }
        deltas.append(u2 - u1)
        print(f"LOVO {held}: B1={u1:.3f}(seen{su1:.3f}) "
              f"B2real={u2:.3f}(seen{su2:.3f}) delta={u2 - u1:+.3f}")

    n_pass = sum(1 for d in deltas if d > 0.02)
    verdict = "PASS" if n_pass >= 2 else "FAIL"
    report = {
        "schema": "rase-w1-future-info-diagnostic/v1",
        "n_rows": len(rows),
        "vlas": vlas,
        "feature_dims": {"B1": int(X_B1.shape[1]), "B2real": int(X_B2.shape[1])},
        "folds": folds,
        "gate_w1": {
            "require_delta_gt": 0.02,
            "folds_above_threshold": n_pass,
            "folds_total": len(deltas),
            "verdict": verdict,
            "note": "PASS -> learned future WM (W2+); FAIL -> lightweight route",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

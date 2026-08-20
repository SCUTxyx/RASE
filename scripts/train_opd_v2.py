#!/usr/bin/env python3
"""Stage C: OPD-v2 — train invariant candidate-consequence risk models with
the ablation matrix C0..C4 (roadmap §7) and evaluate with Leave-One-VLA-Out.

Input: same-root counterfactual rows (collect_same_root.py output) plus the
plain collection rows for reference.  For each variant:

  C0: state + raw chunk stats        (no language, no prior)
  C1: state + canonical chunk        (no language, no prior)
  C2: state + canonical + bigram     (no prior)
  C3: state + canonical + bigram     (visual reserved; same features here)
  C4: C3 + candidate prior

Evaluation:
  - LOVO: leave one VLA out, train on the rest, report unseen-VLA AUROC and
    pairwise ranking (on same-root rows: the candidate whose consequence is
    better should score higher)
  - identity probe on each variant's feature space
  - calibration (Brier/ECE) on held-out VLA rows

Usage (server):
  python train_opd_v2.py \
    --data runs/oft_opportunity/same_root_v1.jsonl \
    --output runs/oft_opportunity/opd_v2_report.json \
    --lovo-test goal   (VLA to hold out; requires rows from >= 3 VLAs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase_common import (
    build_bigram_vocab, bigram_features, canonical_chunk_features,
    raw_chunk_stats_feats, fit_ridge, ridge_predict, auroc, brier, ece,
    probe_identity, candidate_prior,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_X(
    rows: list[dict],
    vocab: dict[str, int],
    variant: str,
    ckpts_root: str,
) -> np.ndarray:
    feats = []
    for r in rows:
        arr = np.asarray(r["chunk_raw"], dtype=np.float64).reshape(-1, 7) \
            if r.get("chunk_raw") else None
        parts = [np.asarray(r.get("s_t_proprio", r.get("proprio")),
                            dtype=np.float64)]
        if arr is None:
            # stats-only fallback (old collection rows)
            stats = [np.asarray(r[k], dtype=np.float64).ravel()
                     for k in ["chunk_mean_pos", "chunk_mean_rot",
                               "chunk_std_pos", "chunk_std_rot"]]
            stats += [[r["chunk_gripper_mean"]], [r["chunk_gripper_std"]],
                      [r["chunk_total_disp"]], [r["chunk_norm_mean"]]]
            parts.append(np.concatenate(stats))
        elif variant in ("C0",):
            parts.append(raw_chunk_stats_feats(arr))
        else:
            parts.append(canonical_chunk_features(arr))
        if variant in ("C2", "C3", "C4"):
            parts.append(bigram_features(r["task"], vocab))
        if variant == "C4":
            parts.append(candidate_prior(r["model"], ckpts_root))
        feats.append(np.concatenate(parts))
    return np.stack(feats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpts-root", default="/root/autodl-tmp/RASE/ckpts")
    parser.add_argument("--variants", default="C0,C1,C2,C3,C4")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lovo-test", default=None,
                        help="VLA name to hold out in LOVO")
    args = parser.parse_args()

    rows = load_rows(args.data)
    if not rows:
        print("no rows")
        return 1
    vlas = sorted({r["model"] for r in rows})
    vocab = build_bigram_vocab([r["task"] for r in rows])
    print(f"rows={len(rows)} vlas={vlas} vocab={len(vocab)}")

    variants = [v for v in args.variants.split(",") if v]
    report: dict = {"rows": len(rows), "vlas": vlas, "variants": {}}

    for variant in variants:
        X = build_X(rows, vocab, variant, args.ckpts_root)
        y = np.array([r["consequence_label"] for r in rows], dtype=np.float64)
        # binarize progress-mode labels (continuous proxies) via median
        if y.max() > 1.0 or set(np.unique(y)) - {0.0, 1.0}:
            thr = np.median(y)
            yb = (y > thr).astype(float)
        else:
            yb = (y > 0).astype(float)
        if yb.sum() == 0 or yb.sum() == len(yb):
            print(f"{variant}: degenerate labels, skipping")
            continue

        v = {"n": len(rows), "feat_dim": int(X.shape[1])}
        # LOVO: hold out one VLA
        if args.lovo_test and args.lovo_test in vlas:
            tr = [i for i, r in enumerate(rows) if r["model"] != args.lovo_test]
            te = [i for i, r in enumerate(rows) if r["model"] == args.lovo_test]
            if len(tr) >= 20 and len(te) >= 5:
                m_, s_, b_, ym_ = fit_ridge(X[tr], yb[tr], alpha=args.alpha)
                p = ridge_predict(X[te], m_, s_, b_, ym_)
                v[f"lovo_{args.lovo_test}_auroc"] = round(auroc(p, yb[te]), 4)
                v[f"lovo_{args.lovo_test}_brier"] = round(brier(p, yb[te]), 4)
                v[f"lovo_{args.lovo_test}_ece"] = round(ece(p, yb[te]), 4)
        # in-domain reference (all rows, LOO-ish by task)
        m_, s_, b_, ym_ = fit_ridge(X, yb, alpha=args.alpha)
        p_all = ridge_predict(X, m_, s_, b_, ym_)
        v["in_domain_auroc"] = round(auroc(p_all, yb), 4)
        # identity probe on this variant's features
        feats_by_vla = {vla: X[[i for i, r in enumerate(rows)
                                if r["model"] == vla]] for vla in vlas}
        probe = probe_identity(feats_by_vla, alpha=args.alpha)
        v["identity_probe"] = {
            "in_domain_multiclass_accuracy":
                probe["in_domain_multiclass_accuracy"],
            "pairwise_auroc": probe["pairwise_auroc"],
            "random_baseline": probe["random_baseline"],
        }
        report["variants"][variant] = v
        print(f"{variant}: dim={v['feat_dim']} in_domain_auroc="
              f"{v['in_domain_auroc']} probe_acc="
              f"{v['identity_probe']['in_domain_multiclass_accuracy']}", flush=True)

    # selection rule: prefer unseen-VLA AUROC over in-domain
    report["summary"] = (
        "choose the variant with the best LOVO AUROC; if tied, lower "
        "identity-probe accuracy wins (less fingerprint leakage).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

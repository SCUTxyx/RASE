#!/usr/bin/env python3
"""Stage A: zero-shot cross-VLA falsification of the frozen risk model.

Scores collection rows of unseen VLAs with the frozen v3 risk model (feature
layout: proprio + raw chunk stats + bigram + candidate prior — identical to
training) and reports:

  - per-VLA discrimination: AUROC / AUPRC / success-vs-failure separation
  - calibration: Brier / ECE / bins
  - distribution shift: train-VLA vs unseen-VLA score distributions,
    Wasserstein distance
  - pairwise ranking on the training-domain paired rows (reference)
  - VLA identity probe on the shared feature space (shortcut diagnostic)
  - verdict: PASS / PARTIAL / FAIL per Stage A gate

Usage:
  python analyze_zs_vla.py \
    --risk-model runs/oft_opportunity/oft_risk_model_v3.npz \
    --vocab runs/oft_opportunity/oft_risk_vocab.json \
    --train-a runs/oft_opportunity/dp_collect_spatial.jsonl \
    --train-b runs/oft_opportunity/dp_collect_object.jsonl \
    --test-vlas goal=runs/oft_opportunity/dp_collect_goal.jsonl \
                   pi0fast=runs/oft_opportunity/dp_collect_pi0fast.jsonl \
    --output runs/oft_opportunity/zero_shot_vla_analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rase_common import (
    RiskModel, build_bigram_vocab, build_row_features, auroc, auprc, brier,
    ece, pairwise_ranking_accuracy, wasserstein_1d, probe_identity,
    CHUNK_KEYS_RAW,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-model", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--train-a", type=Path, default=None)
    parser.add_argument("--train-b", type=Path, default=None)
    parser.add_argument("--test-vlas", nargs="*", default=[],
                        help="name=path pairs, e.g. goal=...jsonl pi0fast=...jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpts-root", default="/root/autodl-tmp/RASE/ckpts")
    parser.add_argument("--auroc-threshold", type=float, default=0.60,
                        help="unseen AUROC >= this counts as transfer signal")
    parser.add_argument("--ece-max", type=float, default=0.30,
                        help="calibration acceptable if ECE <= this")
    args = parser.parse_args()

    model = RiskModel(args.risk_model)
    vocab = json.loads(args.vocab.read_text())
    kw = dict(action="raw", use_language=True, use_prior=True,
              ckpts_root=args.ckpts_root)

    def score_row(row: dict) -> float:
        x = build_row_features(row, vocab, **kw)
        return float(model.predict(x[None, :])[0])

    def row_feats(row: dict) -> np.ndarray:
        return build_row_features(row, vocab, **kw)

    # ---- training-domain reference (pairwise ranking on pairs) ----
    ref = {}
    if args.train_a and args.train_b:
        rows_a = load_rows(args.train_a)
        rows_b = load_rows(args.train_b)
        sa = np.array([score_row(r) for r in rows_a])
        sb = np.array([score_row(r) for r in rows_b])
        ref["train_A_mean_score"] = float(sa.mean())
        ref["train_B_mean_score"] = float(sb.mean())
        by_a = {(r["task"], r["episode_idx"], r["decision_idx"]): i
                for i, r in enumerate(rows_a)}
        pairs = []
        for j, rb in enumerate(rows_b):
            i = by_a.get((rb["task"], rb["episode_idx"], rb["decision_idx"]))
            if i is not None:
                pairs.append((sa[i], sb[j], rows_a[i], rb))
        better_a = np.array([ra["success"] > rb["success"] for _, _, ra, rb in pairs])
        tie = np.array([ra["success"] == rb["success"] for _, _, ra, rb in pairs])
        inf = ~tie
        ref["n_pairs"] = len(pairs)
        ref["n_informative_pairs"] = int(inf.sum())
        if inf.any():
            ref["pairwise_ranking_accuracy"] = round(
                pairwise_ranking_accuracy(
                    np.array([p[0] for p in pairs])[inf],
                    np.array([p[1] for p in pairs])[inf],
                    better_a[inf]), 4)
        ref["train_score_mean"] = float(np.concatenate([sa, sb]).mean())

    # ---- unseen VLAs ----
    test: dict[str, dict] = {}
    feats_by_vla: dict[str, np.ndarray] = {}
    score_by_vla: dict[str, np.ndarray] = {}
    y_by_vla: dict[str, np.ndarray] = {}
    for spec in args.test_vlas:
        name, path = spec.split("=", 1)
        rows = load_rows(Path(path))
        scores = np.array([score_row(r) for r in rows])
        y = np.array([r["success"] for r in rows])
        feats = np.stack([row_feats(r) for r in rows])
        score_by_vla[name] = scores
        y_by_vla[name] = y
        feats_by_vla[name] = feats
        pos = scores[y == 1]
        neg = scores[y == 0]
        entry = {
            "n_rows": len(rows),
            "n_success": int(y.sum()),
            "auroc": round(auroc(scores, y), 4),
            "auprc": round(auprc(scores, y), 4),
            "brier": round(brier(scores, y), 4),
            "ece": round(ece(scores, y), 4),
            "score_mean": round(float(scores.mean()), 4),
            "score_std": round(float(scores.std()), 4),
            "score_q05": round(float(np.percentile(scores, 5)), 4),
            "score_q95": round(float(np.percentile(scores, 95)), 4),
            "success_score_mean": round(float(pos.mean()), 4) if len(pos) else None,
            "failure_score_mean": round(float(neg.mean()), 4) if len(neg) else None,
            "bins": {},
        }
        # calibration bins
        for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            m = (scores >= lo) & (scores < hi)
            if m.sum() >= 5:
                entry["bins"][f"{lo:.1f}-{hi:.1f}"] = {
                    "n": int(m.sum()),
                    "predicted_mean": round(float(scores[m].mean()), 3),
                    "actual_rate": round(float(y[m].mean()), 3),
                }
        if "train_score_mean" in ref:
            entry["wasserstein_vs_train"] = None  # filled below with real train scores
        test[name] = entry

    # Wasserstein vs training domain properly: reuse train rows if available
    if args.train_a and args.train_b and "train_scores" not in ref:
        train_scores = np.concatenate([
            np.array([score_row(r) for r in load_rows(args.train_a)]),
            np.array([score_row(r) for r in load_rows(args.train_b)]),
        ])
        ref["train_scores_mean"] = float(train_scores.mean())
        ref["train_scores_std"] = float(train_scores.std())
        for name in test:
            test[name]["wasserstein_vs_train"] = round(
                wasserstein_1d(score_by_vla[name], train_scores), 4)

    # ---- identity probe on shared feature space ----
    probe = probe_identity(feats_by_vla, alpha=1.0)

    # ---- verdict ----
    verdicts = {}
    for name in test:
        e = test[name]
        auroc_ok = e["auroc"] >= args.auroc_threshold
        calib_ok = e["ece"] <= args.ece_max
        if auroc_ok and calib_ok:
            verdicts[name] = "PASS"
        elif auroc_ok:
            verdicts[name] = "PARTIAL(caltbration)"
        else:
            verdicts[name] = "FAIL"
    arch_groups = {
        "oft": [n for n in test if n.startswith("oft_")],
        "other": [n for n in test if not n.startswith("oft_")],
    }
    overall = "PASS" if all(verdicts[n] == "PASS" for n in test) else (
        "PARTIAL" if any(verdicts[n] in ("PASS", "PARTIAL(caltbration)")
                         for n in test) else "FAIL")
    if arch_groups["oft"] and arch_groups["other"]:
        if all(verdicts[n] == "PASS" for n in arch_groups["oft"]) and \
           any(verdicts[n] == "FAIL" for n in arch_groups["other"]):
            overall = "PARTIAL: checkpoint transfer OK, architecture transfer FAIL"

    report = {
        "schema": "rase-stage-a-zero-shot/v1",
        "risk_model": str(args.risk_model),
        "feature_layout": "proprio+rawchunk+bigram+prior (v3 frozen)",
        "training_reference": ref,
        "unseen_vlas": test,
        "identity_probe": probe,
        "verdicts": verdicts,
        "overall_verdict": overall,
        "gate": {
            "auroc_threshold": args.auroc_threshold,
            "ece_max": args.ece_max,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

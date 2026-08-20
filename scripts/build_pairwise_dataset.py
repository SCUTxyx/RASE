#!/usr/bin/env python3
"""Build the P1 pairwise dataset from same-root collection.

216 roots x C(3,2) pairs = 648 pairs.  q v1: z(progress) - 0.5*z(drift);
recoverability = 0 (needs P3 reference rollouts).

Usage (server, oft env):
  python build_pairwise_dataset.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --output runs/oft_opportunity/crr_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crr_common import load_rows, build_pairs, q_scores, object_drift, \
    group_roots, root_key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--w2", type=float, default=0.5)
    args = ap.parse_args()

    rows = load_rows(args.data)
    from rase_common import build_bigram_vocab
    vocab = build_bigram_vocab([r["task"] for r in rows])
    pairs = build_pairs(rows, vocab)

    # meta
    dqs = np.array([p["dq"] for p in pairs])
    ties = int((np.abs(dqs) <= 1e-12).sum())
    prog = np.array([float(r["consequence_label"]) for r in rows])
    drifts = np.array([object_drift(r) for r in rows])
    roots = group_roots(rows)
    per_root = sorted((len(v) for v in roots.values()), reverse=True)
    suites = {}
    for r in rows:
        suites[r["suite"]] = suites.get(r["suite"], 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    meta = {
        "schema": "rase-p1-pairs/v1",
        "n_rows": len(rows), "n_roots": len(roots),
        "n_pairs": len(pairs), "n_ties": ties,
        "candidates_per_root": {str(int(k)): int(v) for k, v in zip(
            *np.unique([len(v) for v in roots.values()], return_counts=True))},
        "suites_rows": suites,
        "q": {"w2_drift_weight": args.w2, "recoverability": 0,
              "progress_range": [float(prog.min()), float(prog.max())],
              "drift_range": [float(drifts.min()), float(drifts.max())],
              "abs_dq_median": float(np.median(np.abs(dqs))),
              "abs_dq_p25": float(np.percentile(np.abs(dqs), 25)),
              "abs_dq_p75": float(np.percentile(np.abs(dqs), 75))},
        "vocab_size": len(vocab),
    }
    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

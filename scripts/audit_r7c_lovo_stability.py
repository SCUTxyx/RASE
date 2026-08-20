#!/usr/bin/env python3
"""Five-seed stability audit for heldout-VLA adaptation curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase.risk.r7_source_protocol import TRAIN_SEEDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for seed in TRAIN_SEEDS:
        path = args.input_root / f"seed_{seed}" / f"heldout_{args.heldout_policy}.json"
        if not path.is_file():
            raise ValueError(f"missing LOVO report: {path}")
        row = json.loads(path.read_text())
        if row.get("heldout_policy") != args.heldout_policy or int(row.get("seed", -1)) != seed:
            raise ValueError(f"heldout/seed mismatch: {path}")
        reports.append(row)
    hashes = {row.get("dataset_sha256") for row in reports}
    if len(hashes) != 1:
        raise ValueError("LOVO seeds do not bind one dataset")
    per_seed = []
    for row in reports:
        zero = row["curves"]["unlabeled"]["0"]["metrics"]
        adapted = row["curves"]["unlabeled"]["32"]["metrics"]
        passed = adapted["auroc"] >= 0.65 and adapted["ap_above_prevalence"] >= 0.05
        per_seed.append({
            "seed": row["seed"], "passed_32_unlabeled": passed,
            "zero_shot": zero, "adapted_32_unlabeled": adapted,
            "delta_auroc_32_minus_zero": adapted["auroc"] - zero["auroc"],
            "nonmonotone_auroc_curve": any(
                row["curves"]["unlabeled"][str(right)]["metrics"]["auroc"]
                < row["curves"]["unlabeled"][str(left)]["metrics"]["auroc"]
                for left, right in ((0, 8), (8, 16), (16, 32))
            ),
        })
    passed = sum(row["passed_32_unlabeled"] for row in per_seed)
    result = {
        "schema_version": "rase-r7c-lovo-stability/v1",
        "status": "PASS" if passed >= 4 else "FAIL",
        "heldout_policy": args.heldout_policy,
        "dataset_sha256": next(iter(hashes)),
        "seeds_passed_32_unlabeled": passed, "seeds_required": 4,
        "mean_zero_shot_auroc": float(np.mean([
            row["zero_shot"]["auroc"] for row in per_seed
        ])),
        "mean_32_unlabeled_auroc": float(np.mean([
            row["adapted_32_unlabeled"]["auroc"] for row in per_seed
        ])),
        "mean_32_minus_zero_auroc": float(np.mean([
            row["delta_auroc_32_minus_zero"] for row in per_seed
        ])),
        "nonmonotone_curve_seeds": sum(row["nonmonotone_auroc_curve"] for row in per_seed),
        "per_seed": per_seed,
        "zero_shot_is_gate": False,
        "gate": "32 unlabeled trajectories: AUROC>=0.65 and AP-prevalence>=0.05 in >=4/5 seeds",
        "remains_locked": ["selector", "world-model", "validation", "test"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

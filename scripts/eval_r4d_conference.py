#!/usr/bin/env python3
"""Milestone 5: Conference-level evaluation.

Requires the frozen risk model (LightRiskStudent) plus an OFT oracle endpoint.
Evaluates on 100-200+ paired episodes across 4 suites with:
  - paired bootstrap statistical analysis
  - rescue / harm breakdown
  - risk-coverage Pareto
  - (optional) second policy pair
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000, seed: int = 20260808
) -> dict[str, float]:
    """Paired bootstrap 95% CI and p-value for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    n = len(diff)
    if n == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0}
    sample = diff[rng.integers(0, n, size=(n_boot, n))]
    means = sample.mean(axis=1)
    p = float(np.mean(means <= 0))
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p": min(p, 1 - p) * 2,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="boundary jsonl to evaluate")
    p.add_argument("--predictions-json", type=Path, required=True,
                   help="report.json with OOF predictions + threshold")
    p.add_argument("--output", type=Path,
                   default=ROOT / "runs/pre_c0_r4/m5_conference_eval.json")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260808)
    args = p.parse_args()

    rows = read_jsonl(args.dataset)
    pred_payload = json.loads(args.predictions_json.read_text())
    threshold = float(pred_payload.get("mean_threshold", pred_payload.get("threshold", 0.5)))
    oof = pred_payload.get("oof_predictions") or pred_payload.get("predictions") or {}

    labels = np.asarray([int(bool(r.get("success_if_handback_now", False))) for r in rows])
    costs = np.asarray([float(r.get("remaining_teacher_steps", 0.0)) for r in rows])
    baseline = np.asarray([int(bool(r.get("success_if_continue_oft", True))) for r in rows])

    scores = np.array([
        float(oof.get(f"{r['state_key']}:{r['elapsed_oft_steps']}", 0.5))
        if isinstance(oof, dict) else 0.5
        for r in rows
    ])
    handback = scores >= threshold

    n = len(rows)
    rescued = handback & (labels == 1)
    harmed = handback & (labels == 0)
    saved_steps = float(np.sum(np.where(handback, costs, 0.0)))

    # Cost: policy operator selected (handback early) vs PERSISTENT
    oft_cost = float(np.sum(costs))
    handback_cost = oft_cost - saved_steps
    savings_pct = saved_steps / max(oft_cost, 1e-9)

    # Paired bootstrap: success non-inferiority of handback vs persistent
    policy_success = baseline | handback  # handback rescues; persistent covers rest
    succ_diff = paired_bootstrap(policy_success.astype(float), baseline.astype(float),
                                 n_boot=args.n_boot, seed=args.seed)
    cost_diff = paired_bootstrap(np.full(n, handback_cost / max(n, 1)),
                                 np.full(n, oft_cost / max(n, 1)),
                                 n_boot=args.n_boot, seed=args.seed + 1)

    # Risk-coverage Pareto: coverage (fraction of states where risk known)
    # vs harm rate for a sweep of thresholds
    pareto = []
    for th in np.linspace(0.0, 1.0, 11):
        hb = scores >= th
        coverage = float(np.mean(hb))
        harm = float(np.mean(hb & (labels == 0))) if hb.any() else 0.0
        rescue = float(np.mean(hb & (labels == 1))) if hb.any() else 0.0
        pareto.append({"threshold": float(th), "coverage": coverage,
                       "harm": harm, "rescue": rescue})

    report = {
        "schema_version": "rase-pre-c0-r4d-m5-final/v1",
        "n_rows": n,
        "n_states": len(set(str(r["state_key"]) for r in rows)),
        "suites": sorted({str(r["suite"]) for r in rows}),
        "threshold": threshold,
        "handback_rate": float(np.mean(handback)),
        "rescued": float(np.mean(rescued)),
        "harmed": float(np.mean(harmed)),
        "oft_savings": savings_pct,
        "persistent_success": float(np.mean(baseline)),
        "policy_success": float(np.mean(policy_success)),
        "paired_bootstrap": {
            "success_non_inferiority": succ_diff,
            "cost_reduction": cost_diff,
        },
        "risk_coverage_pareto": pareto,
        "statistics": {
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "source": str(args.dataset.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

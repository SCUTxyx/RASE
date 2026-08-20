#!/usr/bin/env python3
"""Analyze seed sensitivity of leakage-safe R4-D nested OOF controllers."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decisions(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, tuple[int | None, bool]]:
    predictions = report["oof_predictions"]
    thresholds = report["oof_thresholds"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_key"])].append(row)
    output = {}
    for state, items in grouped.items():
        items.sort(key=lambda r: int(r.get("elapsed_oft_steps", 0)))
        chosen = None
        for row in items:
            key = f"{row['state_key']}:{row['elapsed_oft_steps']}"
            if float(predictions[key]) >= float(thresholds[key]):
                chosen = row
                break
        output[state] = (
            int(chosen["elapsed_oft_steps"]) if chosen is not None else None,
            bool(chosen.get("success_if_handback_now", False)) if chosen is not None else
            bool(items[0].get("success_if_continue_oft", True)),
        )
    return output


def range_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
    }


def zero_error_upper_bound(n: int, alpha: float = 0.05) -> float:
    return 1.0 - alpha ** (1.0 / max(n, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-error", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    reports = [json.loads(path.read_text()) for path in args.report]
    seed_decisions = [decisions(rows, report) for report in reports]
    state_keys = sorted(seed_decisions[0])
    if any(set(decision) != set(state_keys) for decision in seed_decisions):
        raise ValueError("state sets differ across seed reports")

    pair_agreement = []
    for left, right in itertools.combinations(seed_decisions, 2):
        pair_agreement.append(sum(left[key][0] == right[key][0] for key in state_keys) / len(state_keys))
    unstable_states = [
        key for key in state_keys
        if len({decision[key][0] for decision in seed_decisions}) > 1
    ]
    any_handback_frequency = {
        key: sum(decision[key][0] is not None for decision in seed_decisions) / len(seed_decisions)
        for key in state_keys
    }

    metrics = [report["oof_state_metrics"] for report in reports]
    all_thresholds = [
        float(value) for report in reports for value in report["per_fold_thresholds"]
    ]
    calibration_state_counts = [
        int(fold["calibration_metrics"]["n_states"])
        for report in reports for fold in report["fold_reports"]
    ]
    n_required_zero_error = math.ceil(math.log(0.05) / math.log(1.0 - args.target_error))
    report = {
        "schema_version": "rase-pre-c0-r5-seed-stability/v1",
        "n_seeds": len(reports),
        "seeds": [r.get("seed") for r in reports],
        "n_states": len(state_keys),
        "row_auc": range_summary([float(r["oof_row_auc"]) for r in reports]),
        "policy_success_rate": range_summary([float(m["policy_success_rate"]) for m in metrics]),
        "success_gap": range_summary([float(m["success_gap"]) for m in metrics]),
        "conditional_false_handback_rate": range_summary([
            float(m["conditional_false_handback_rate"]) for m in metrics
        ]),
        "oft_savings": range_summary([float(m["oft_savings"]) for m in metrics]),
        "all_four_gates_passed": sum(all(r["gates"].values()) for r in reports),
        "thresholds": {
            **range_summary(all_thresholds),
            "values": all_thresholds,
        },
        "decision_stability": {
            "mean_pairwise_exact_handback_time_agreement": float(np.mean(pair_agreement)),
            "pairwise_agreement_range": [float(min(pair_agreement)), float(max(pair_agreement))],
            "unstable_states": len(unstable_states),
            "unstable_state_fraction": len(unstable_states) / len(state_keys),
            "states_always_handback": sum(v == 1.0 for v in any_handback_frequency.values()),
            "states_never_handback": sum(v == 0.0 for v in any_handback_frequency.values()),
            "states_seed_dependent_handback": sum(0.0 < v < 1.0 for v in any_handback_frequency.values()),
        },
        "calibration_feasibility": {
            "calibration_states_per_fold": range_summary(calibration_state_counts),
            "worst_zero-error_one_sided_95_upper_bound": max(
                zero_error_upper_bound(n) for n in calibration_state_counts
            ),
            "zero-error_samples_required_for_95_upper_at_target": n_required_zero_error,
            "target_false_handback_rate": args.target_error,
        },
        "source_reports": [str(path.resolve()) for path in args.report],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

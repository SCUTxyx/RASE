#!/usr/bin/env python3
"""Diagnose a completed R6-C.1 OOF failure without changing its gate.

This audit never reselects a deployment threshold on OOF labels.  It measures
ranking observability (mean versus LCB), ensemble collapse/dispersion,
train-fold calibration variability, entry timing, and suite-localized harm.
These are descriptive post-hoc diagnostics only; ``stability.json`` remains
the sole formal stage decision.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks(scores)[labels].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def score_metrics(rows: list[dict], label_key: str, score_key: str) -> dict:
    labels = np.asarray([bool(row[label_key]) for row in rows])
    scores = np.asarray([float(row[score_key]) for row in rows])
    return {
        "rows": len(rows), "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "auc": auc(labels, scores), "ap": ap(labels, scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stability = json.loads(args.stability.read_text())
    reports = [json.loads(path.read_text()) for path in args.report]
    if not reports or any(report.get("status") != "complete" for report in reports):
        raise ValueError("all OOF reports must be complete")
    seeds = []
    for report in reports:
        t0 = [row for row in report["predictions"]
              if int(row["elapsed_source_steps"]) == 0]
        for row in t0:
            row["source_failure"] = not bool(row["source_success"])
            row["rescue_benefit"] = (
                row["source_failure"] and bool(row["persistent_success"])
            )
            row["source_risk_mean"] = -float(row["source_mean"])
            row["source_risk_lcb"] = -float(row["source_lcb"])
        thresholds = [fold["controller_params"] for fold in report["fold_reports"]]
        trajectories = report["trajectory_records"]
        prediction_by_group = defaultdict(list)
        for row in report["predictions"]:
            prediction_by_group[str(row["group_id"])].append(row)
        missed_blockers = Counter()
        for trajectory in trajectories:
            if not bool(trajectory["false_continue"]):
                continue
            candidates = sorted(
                prediction_by_group[str(trajectory["group_id"])],
                key=lambda row: int(row["elapsed_source_steps"]),
            )
            t0_row = next((row for row in candidates
                           if int(row["elapsed_source_steps"]) == 0), None)
            if t0_row is None:
                missed_blockers["missing_t0"] += 1
                continue
            risky = float(t0_row["source_lcb"]) < float(t0_row["risk_thr"])
            worth = float(t0_row["advantage_lcb"]) > float(t0_row["adv_thr"])
            if not risky and not worth:
                missed_blockers["t0_blocked_by_both"] += 1
            elif not risky:
                missed_blockers["t0_blocked_by_source_risk"] += 1
            elif not worth:
                missed_blockers["t0_blocked_by_advantage_lcb"] += 1
            else:
                missed_blockers["t0_would_enter_inconsistent"] += 1
        timing = Counter(
            "none" if row["enter_elapsed_source_steps"] is None
            else f"t{int(row['enter_elapsed_source_steps'])}"
            for row in trajectories
        )
        harm_by_suite = defaultdict(lambda: {"episodes": 0, "harm": 0})
        for row in trajectories:
            bucket = harm_by_suite[str(row["suite"])]
            bucket["episodes"] += 1
            bucket["harm"] += int(bool(row["paired_harm"]))
        seeds.append({
            "seed": int(report["seed"]),
            "formal_metrics": report["metrics"],
            "t0_ranking": {
                "source_risk_mean": score_metrics(t0, "source_failure", "source_risk_mean"),
                "source_risk_lcb": score_metrics(t0, "source_failure", "source_risk_lcb"),
                "rescue_advantage_mean": score_metrics(t0, "rescue_benefit", "advantage_mean"),
                "rescue_advantage_lcb": score_metrics(t0, "rescue_benefit", "advantage_lcb"),
                "persistent_success_mean": score_metrics(
                    t0, "persistent_success", "persistent_success_mean"
                ),
            },
            "uncertainty": {
                "source_std_mean": float(np.mean([row["source_std"] for row in t0])),
                "advantage_std_mean": float(np.mean([row["advantage_std"] for row in t0])),
                "source_lcb_zero_fraction": float(np.mean([
                    float(row["source_lcb"]) <= 1e-8 for row in t0
                ])),
            },
            "fold_thresholds": thresholds,
            "risk_threshold_range": [
                float(min(value["risk_thr"] for value in thresholds)),
                float(max(value["risk_thr"] for value in thresholds)),
            ],
            "advantage_threshold_range": [
                float(min(value["adv_thr"] for value in thresholds)),
                float(max(value["adv_thr"] for value in thresholds)),
            ],
            "entry_timing": dict(sorted(timing.items())),
            "missed_rescue_t0_blockers": dict(sorted(missed_blockers.items())),
            "harm_by_suite": {
                suite: {**value, "rate": value["harm"] / max(1, value["episodes"])}
                for suite, value in sorted(harm_by_suite.items())
            },
        })

    keys = ["source_risk_mean", "source_risk_lcb", "rescue_advantage_mean",
            "rescue_advantage_lcb", "persistent_success_mean"]
    aggregate = {
        key: {
            "auc_mean": float(np.nanmean([seed["t0_ranking"][key]["auc"] for seed in seeds])),
            "auc_min": float(np.nanmin([seed["t0_ranking"][key]["auc"] for seed in seeds])),
            "auc_max": float(np.nanmax([seed["t0_ranking"][key]["auc"] for seed in seeds])),
            "ap_mean": float(np.nanmean([seed["t0_ranking"][key]["ap"] for seed in seeds])),
        }
        for key in keys
    }
    result = {
        "schema_version": "rase-r6c1-selector-failure-mechanism/v1",
        "status": "complete",
        "formal_stage_gate_passed": bool(stability.get("stage_gate_passed", False)),
        "formal_decision_unchanged": True,
        "reports": [str(path) for path in args.report],
        "aggregate_t0_ranking": aggregate,
        "seed_diagnostics": seeds,
        "interpretation_rules": {
            "mean_good_lcb_bad": "ensemble dispersion or conservative LCB dominates ranking",
            "source_good_advantage_bad": "failure risk is observable but fallback recoverability is not",
            "both_bad": "current deployment representation lacks task-held-out signal",
            "threshold_ranges_wide": "small calibration folds do not identify a stable controller",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "formal_stage_gate_passed": result["formal_stage_gate_passed"],
        "aggregate_t0_ranking": aggregate,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

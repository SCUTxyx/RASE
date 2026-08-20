#!/usr/bin/env python3
"""Leakage-safe offline evaluation for nested-OOF R4-D predictions.

This is not named a conference result: boundary rows are collapsed to one
earliest handback decision per state and uncertainty is bootstrapped by task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_state_records(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = report["oof_predictions"]
    thresholds = report.get("oof_thresholds")
    default_threshold = float(report.get("mean_threshold", report.get("threshold", 0.5)))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['state_key']}:{row['elapsed_oft_steps']}"
        if key not in predictions:
            raise ValueError(f"missing OOF prediction: {key}")
        item = dict(row)
        item["_score"] = float(predictions[key])
        item["_threshold"] = (
            float(thresholds[key]) if thresholds is not None else default_threshold
        )
        grouped[str(row["state_key"])].append(item)

    records = []
    for state_key, items in grouped.items():
        items.sort(key=lambda row: int(row.get("elapsed_oft_steps", 0)))
        reference = items[0]
        accepted = [row for row in items if row["_score"] >= row["_threshold"]]
        chosen = accepted[0] if accepted else None
        persistent_success = bool(reference.get("success_if_continue_oft", True))
        persistent_cost = float(reference.get("persistent_executed_oft_steps", 0.0))
        if chosen is None:
            policy_success = persistent_success
            policy_cost = persistent_cost
        else:
            policy_success = bool(chosen.get("success_if_handback_now", False))
            policy_cost = min(float(chosen.get("elapsed_oft_steps", 0.0)), persistent_cost)
        records.append({
            "state_key": state_key,
            "task_id": str(reference.get("task_id")),
            "suite": str(reference.get("suite")),
            "handback": chosen is not None,
            "handback_at": int(chosen["elapsed_oft_steps"]) if chosen is not None else None,
            "persistent_success": persistent_success,
            "policy_success": policy_success,
            "persistent_cost": persistent_cost,
            "policy_cost": policy_cost,
            "harm": bool(chosen is not None and persistent_success and not policy_success),
            "successful_handback": bool(chosen is not None and policy_success),
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(records), 1)
    persistent_successes = sum(r["persistent_success"] for r in records)
    persistent_cost = sum(r["persistent_cost"] for r in records)
    policy_cost = sum(r["policy_cost"] for r in records)
    return {
        "n_states": len(records),
        "persistent_success_rate": sum(r["persistent_success"] for r in records) / n,
        "policy_success_rate": sum(r["policy_success"] for r in records) / n,
        "success_difference": (
            sum(r["policy_success"] for r in records)
            - sum(r["persistent_success"] for r in records)
        ) / n,
        "handback_rate": sum(r["handback"] for r in records) / n,
        "successful_handback_rate": sum(r["successful_handback"] for r in records) / n,
        "harm_rate": sum(r["harm"] for r in records) / n,
        "conditional_false_handback_rate": (
            sum(r["harm"] for r in records) / max(persistent_successes, 1)
        ),
        "oft_savings": (persistent_cost - policy_cost) / max(persistent_cost, 1e-9),
        "persistent_total_oft_steps": persistent_cost,
        "policy_total_oft_steps": policy_cost,
    }


def clustered_bootstrap(
    records: list[dict[str, Any]], *, n_boot: int, seed: int, noninferiority_margin: float
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    success_diffs, savings = [], []
    for _ in range(n_boot):
        sampled_tasks = rng.choice(tasks, len(tasks), replace=True)
        sampled = [record for task in sampled_tasks for record in by_task[str(task)]]
        metrics = summarize(sampled)
        success_diffs.append(metrics["success_difference"])
        savings.append(metrics["oft_savings"])
    success_diffs = np.asarray(success_diffs)
    savings = np.asarray(savings)
    # Add-one correction prevents impossible p=0 reports from a finite bootstrap.
    p_noninferior = (1 + int(np.sum(success_diffs < -noninferiority_margin))) / (n_boot + 1)
    p_no_savings = (1 + int(np.sum(savings <= 0.0))) / (n_boot + 1)
    return {
        "cluster_unit": "task",
        "n_tasks": len(tasks),
        "n_boot": n_boot,
        "success_difference_ci95": np.quantile(success_diffs, [0.025, 0.975]).tolist(),
        "oft_savings_ci95": np.quantile(savings, [0.025, 0.975]).tolist(),
        "p_violate_noninferiority_margin": float(p_noninferior),
        "p_no_cost_savings": float(p_no_savings),
        "minimum_reportable_p": 1.0 / (n_boot + 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    prediction_report = json.loads(args.predictions_json.read_text())
    records = build_state_records(rows, prediction_report)
    metrics = summarize(records)
    report = {
        "schema_version": "rase-pre-c0-r4d-offline-nested-oof-eval/v2",
        "claim_scope": "offline_nested_task_oof_not_closed_loop",
        "metrics": metrics,
        "clustered_bootstrap": clustered_bootstrap(
            records, n_boot=args.n_boot, seed=args.seed,
            noninferiority_margin=args.noninferiority_margin,
        ),
        "per_state": records,
        "source": str(args.dataset.resolve()),
        "predictions_source": str(args.predictions_json.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_state"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

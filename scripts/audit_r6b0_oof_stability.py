#!/usr/bin/env python3
"""Aggregate R6-B0 OOF seeds with task-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def controller_metrics(data: dict[str, np.ndarray], idx: np.ndarray,
                       cont: np.ndarray) -> dict[str, float]:
    source = data["source_seed_success"][idx].astype(bool)
    persistent = data["persistent_success"][idx].astype(bool)
    psteps = data["persistent_teacher_steps"][idx].astype(float)
    decisions = np.repeat(cont[:, None], 2, axis=1)
    success = np.where(decisions, source, persistent[:, None])
    baseline = np.repeat(persistent[:, None], 2, axis=1)
    false = decisions & (~source) & baseline
    teacher = np.where(decisions, 0.0, psteps[:, None])
    total = np.repeat(psteps[:, None], 2, axis=1)
    return {
        "success_gap": float((success.sum() - baseline.sum()) / success.size),
        "false_continue_rate": float(false.sum() / max(1, baseline.sum())),
        "savings": float(1.0 - teacher.sum() / max(1.0, total.sum())),
    }


def gate(metrics: dict[str, float]) -> bool:
    return (metrics["success_gap"] >= -0.05
            and metrics["false_continue_rate"] <= 0.05
            and metrics["savings"] >= 0.20)


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def discrimination(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool).reshape(-1)
    scores = scores.astype(float).reshape(-1)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return {"auroc": float("nan"), "average_precision": float(positive > 0),
                "brier": float(np.mean((scores - labels.astype(float)) ** 2))}
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auc = (ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)
    descending = np.argsort(-scores, kind="stable")
    sorted_labels = labels[descending]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    ap = float(precision[sorted_labels].mean())
    return {"auroc": float(auc), "average_precision": ap,
            "brier": float(np.mean((scores - labels.astype(float)) ** 2))}


def bootstrap(data: dict[str, np.ndarray], idx: np.ndarray, decision: np.ndarray,
              *, repeats: int, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    tasks = np.asarray(sorted(set(data["task_id"][idx].tolist())))
    values = {key: [] for key in ["success_gap", "false_continue_rate", "savings"]}
    task_for_row = data["task_id"][idx]
    for _ in range(repeats):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        local_parts: list[np.ndarray] = []
        decision_parts: list[np.ndarray] = []
        for task in sampled:
            local = np.where(task_for_row == task)[0]
            local_parts.append(idx[local])
            decision_parts.append(decision[local])
        sample_idx = np.concatenate(local_parts)
        sample_decision = np.concatenate(decision_parts)
        metrics = controller_metrics(data, sample_idx, sample_decision)
        for key in values:
            values[key].append(metrics[key])
    return {key: interval(item) for key, item in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--required-passing-seeds", type=int, default=4)
    args = parser.parse_args()

    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    paths = sorted(args.input_dir.glob("seed_*.json"))
    if not paths:
        raise ValueError(f"no seed_*.json files in {args.input_dir}")
    rng = np.random.default_rng(args.bootstrap_seed)
    seed_reports: list[dict[str, Any]] = []
    all_policies: set[str] = set()
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise ValueError(f"incomplete result: {path}")
        predictions = result["predictions"]
        idx = np.asarray([row["index"] for row in predictions], dtype=int)
        decision = np.asarray([row["continue_source"] for row in predictions], dtype=bool)
        score = np.asarray([row["source_lcb"] for row in predictions], dtype=float)
        policy_values = sorted(set(data["policy_id"][idx].tolist()))
        all_policies.update(policy_values)
        groups: dict[str, dict[str, Any]] = {}
        for name in ["overall", *policy_values]:
            mask = np.ones(len(idx), dtype=bool) if name == "overall" else data["policy_id"][idx] == name
            group_idx, group_decision = idx[mask], decision[mask]
            metrics = controller_metrics(data, group_idx, group_decision)
            episode_labels = data["source_seed_success"][group_idx].reshape(-1)
            episode_scores = np.repeat(score[mask], 2)
            groups[name] = {
                "metrics": metrics,
                "gate_passed": gate(metrics),
                "discrimination": discrimination(episode_labels, episode_scores),
                "task_cluster_bootstrap": bootstrap(
                    data, group_idx, group_decision,
                    repeats=args.bootstrap_repeats, rng=rng,
                ),
            }
        strict = groups["overall"]["gate_passed"] and all(
            groups[name]["gate_passed"] for name in policy_values
        )
        seed_reports.append({
            "seed": result["seed"], "path": str(path),
            "strict_gate_passed": strict, "groups": groups,
        })

    passing = sum(report["strict_gate_passed"] for report in seed_reports)
    metric_across_seeds: dict[str, Any] = {}
    for group in ["overall", *sorted(all_policies)]:
        available = [report["groups"][group] for report in seed_reports if group in report["groups"]]
        metric_across_seeds[group] = {
            key: interval([item["metrics"][key] for item in available])
            for key in ["success_gap", "false_continue_rate", "savings"]
        }
    result = {
        "schema_version": "rase-r6b0-oof-stability/v1",
        "status": "complete",
        "scientific_scope": "initial exact-state feasibility; task-clustered uncertainty",
        "input_dir": str(args.input_dir),
        "seed_count": len(seed_reports),
        "passing_seed_count": passing,
        "required_passing_seed_count": args.required_passing_seeds,
        "stage_gate_passed": len(seed_reports) >= 5 and passing >= args.required_passing_seeds,
        "metric_across_seeds": metric_across_seeds,
        "seed_reports": seed_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed_count": result["seed_count"],
        "passing_seed_count": passing,
        "stage_gate_passed": result["stage_gate_passed"],
        "metric_across_seeds": metric_across_seeds,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze live exact-boundary handback labels and teacher-step Pareto tradeoffs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize(decisions: list[dict]) -> dict:
    n = len(decisions)
    successes = sum(bool(row["success"]) for row in decisions)
    persistent_successes = sum(bool(row["persistent_success"]) for row in decisions)
    steps = sum(int(row["steps"]) for row in decisions)
    persistent_steps = sum(int(row["persistent_steps"]) for row in decisions)
    rescues = sum(row["success"] and not row["persistent_success"] for row in decisions)
    harms = sum(not row["success"] and row["persistent_success"] for row in decisions)
    return {
        "n_states": n,
        "successes": successes,
        "success_rate": successes / max(1, n),
        "persistent_successes": persistent_successes,
        "persistent_success_rate": persistent_successes / max(1, n),
        "success_minus_persistent": (successes - persistent_successes) / max(1, n),
        "executed_oft_steps": steps,
        "persistent_executed_oft_steps": persistent_steps,
        "oft_step_savings_fraction": 1.0 - steps / max(1, persistent_steps),
        "paired_rescues": rescues,
        "paired_harms": harms,
    }


def _cluster_bootstrap(decisions: list[dict], seed: int, draws: int) -> dict:
    by_task = defaultdict(list)
    for row in decisions:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    success_diffs, savings = [], []
    for _ in range(draws):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        values = [row for task in sampled for row in by_task[str(task)]]
        summary = _summarize(values)
        success_diffs.append(summary["success_minus_persistent"])
        savings.append(summary["oft_step_savings_fraction"])
    return {
        "unit": "logical_task",
        "draws": draws,
        "success_minus_persistent_ci95": np.quantile(success_diffs, [0.025, 0.975]).tolist(),
        "oft_step_savings_fraction_ci95": np.quantile(savings, [0.025, 0.975]).tolist(),
    }


def _pareto(methods: dict[str, dict]) -> list[str]:
    names = []
    for name, value in methods.items():
        dominated = any(
            other != name
            and candidate["success_rate"] >= value["success_rate"]
            and candidate["executed_oft_steps"] <= value["executed_oft_steps"]
            and (
                candidate["success_rate"] > value["success_rate"]
                or candidate["executed_oft_steps"] < value["executed_oft_steps"]
            )
            for other, candidate in methods.items()
        )
        if not dominated:
            names.append(name)
    return sorted(names, key=lambda name: methods[name]["executed_oft_steps"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    collection = json.loads(args.collection_report.read_text())
    if collection.get("schema_version") != "rase-pre-c0-r4-boundary-merged/v3":
        raise SystemExit("expected merged v3 collection report")
    if Path(str(collection.get("output", ""))).resolve() != args.dataset.resolve():
        raise SystemExit("dataset does not match collection report")
    rows = _read_jsonl(args.dataset)
    by_state = defaultdict(list)
    for row in rows:
        by_state[str(row["state_key"])].append(row)
    for values in by_state.values():
        values.sort(key=lambda row: int(row["elapsed_oft_steps"]))

    boundaries = sorted({int(row["elapsed_oft_steps"]) for row in rows})
    decisions_by_method: dict[str, list[dict]] = {}
    for boundary in boundaries:
        decisions = []
        for state, values in sorted(by_state.items()):
            reference = values[0]
            candidate = next(
                (row for row in values if int(row["elapsed_oft_steps"]) == boundary),
                None,
            )
            persistent_success = bool(reference["success_if_continue_oft"])
            persistent_steps = int(reference["persistent_executed_oft_steps"])
            decisions.append({
                "state_key": state,
                "task_id": str(reference["task_id"]),
                "success": (
                    bool(candidate["success_if_handback_now"])
                    if candidate is not None else persistent_success
                ),
                "steps": boundary if candidate is not None else persistent_steps,
                "persistent_success": persistent_success,
                "persistent_steps": persistent_steps,
            })
        decisions_by_method[f"fixed_h{boundary}"] = decisions

    persistent_decisions, oracle_decisions = [], []
    for state, values in sorted(by_state.items()):
        reference = values[0]
        persistent_success = bool(reference["success_if_continue_oft"])
        persistent_steps = int(reference["persistent_executed_oft_steps"])
        persistent_decisions.append({
            "state_key": state,
            "task_id": str(reference["task_id"]),
            "success": persistent_success,
            "steps": persistent_steps,
            "persistent_success": persistent_success,
            "persistent_steps": persistent_steps,
        })
        successful = [row for row in values if row["success_if_handback_now"]]
        if successful:
            chosen = min(successful, key=lambda row: int(row["elapsed_oft_steps"]))
            success, steps = True, int(chosen["elapsed_oft_steps"])
        else:
            success, steps = persistent_success, persistent_steps
        oracle_decisions.append({
            "state_key": state,
            "task_id": str(reference["task_id"]),
            "success": success,
            "steps": steps,
            "persistent_success": persistent_success,
            "persistent_steps": persistent_steps,
        })
    decisions_by_method["persistent_oft"] = persistent_decisions
    decisions_by_method["privileged_earliest_safe_oracle"] = oracle_decisions

    methods = {}
    for offset, (name, decisions) in enumerate(sorted(decisions_by_method.items())):
        methods[name] = {
            **_summarize(decisions),
            "task_cluster_bootstrap": _cluster_bootstrap(
                decisions, args.seed + offset, args.bootstrap_draws
            ),
        }
    history_match = sum(
        bool(row["success_if_handback_now"])
        == bool(row.get("historical_success_if_handback_now"))
        for row in rows
    )
    report = {
        "schema_version": "rase-pre-c0-r4-live-opportunity/v1",
        "dataset": str(args.dataset.resolve()),
        "collection_report": str(args.collection_report.resolve()),
        "n_states": len(by_state),
        "n_tasks": len({row["task_id"] for row in rows}),
        "n_boundaries": len(rows),
        "boundaries": boundaries,
        "historical_handback_label_matches": history_match,
        "historical_handback_labels_compared": len(rows),
        "methods": methods,
        "pareto_frontier": _pareto(methods),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

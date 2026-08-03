#!/usr/bin/env python3
"""Sweep preregistration candidates for fixed per-intervention utility costs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _complete_rows(
    snapshots: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    operator_ids: list[str],
) -> list[dict[str, Any]]:
    snapshot_by_id = {row["snapshot_id"]: row for row in snapshots}
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.get("observed") and not outcome.get("proxy", False):
            key = (str(outcome["snapshot_id"]), str(outcome["operator_id"]))
            values[key].append(float(bool(outcome.get("success"))))
    rows = []
    for snapshot_id, snapshot in snapshot_by_id.items():
        success = {
            operator_id: float(np.mean(values[(snapshot_id, operator_id)]))
            for operator_id in operator_ids
            if values[(snapshot_id, operator_id)]
        }
        if len(success) == len(operator_ids):
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "task_id": str(snapshot["task_id"]),
                    "success": success,
                }
            )
    if not rows:
        raise ValueError("matrix contains no complete states")
    return rows


def evaluate_cost_point(
    rows: list[dict[str, Any]],
    operator_ids: list[str],
    penalties: dict[str, float],
    *,
    success_reward: float = 1.0,
) -> dict[str, Any]:
    unknown = set(penalties) - set(operator_ids)
    if unknown:
        raise ValueError(f"penalties reference unknown operators: {sorted(unknown)}")
    effective = {name: float(penalties.get(name, 0.0)) for name in operator_ids}
    if success_reward <= 0 or any(value < 0 for value in effective.values()):
        raise ValueError("success reward must be positive and penalties non-negative")

    utilities = [
        {
            name: success_reward * row["success"][name] - effective[name]
            for name in operator_ids
        }
        for row in rows
    ]
    fixed = {
        name: float(np.mean([value[name] for value in utilities]))
        for name in operator_ids
    }
    best_fixed = max(operator_ids, key=lambda name: fixed[name])
    oracle = float(np.mean([max(value.values()) for value in utilities]))
    state_winners: Counter[str] = Counter()
    task_winners: dict[str, set[str]] = defaultdict(set)
    supported_state_winners: Counter[str] = Counter()
    supported_task_winners: dict[str, set[str]] = defaultdict(set)
    failure_only_state_winners: Counter[str] = Counter()
    failure_only_task_winners: dict[str, set[str]] = defaultdict(set)
    n_no_success_states = 0
    for row, value in zip(rows, utilities):
        best = max(value.values())
        winners = [name for name in operator_ids if np.isclose(value[name], best)]
        any_success = any(row["success"][name] > 0.5 for name in operator_ids)
        n_no_success_states += int(not any_success)
        if len(winners) == 1:
            winner = winners[0]
            state_winners[winner] += 1
            task_winners[winner].add(row["task_id"])
            if row["success"][winner] > 0.5:
                supported_state_winners[winner] += 1
                supported_task_winners[winner].add(row["task_id"])
            elif not any_success:
                failure_only_state_winners[winner] += 1
                failure_only_task_winners[winner].add(row["task_id"])
    return {
        "penalty_by_operator": effective,
        "per_operator_mean_utility": fixed,
        "best_fixed_operator": best_fixed,
        "best_fixed_utility": fixed[best_fixed],
        "same_state_oracle_utility": oracle,
        "oracle_minus_best_fixed": oracle - fixed[best_fixed],
        "unique_winner_state_counts": {
            name: int(state_winners.get(name, 0)) for name in operator_ids
        },
        "unique_winner_task_counts": {
            name: len(task_winners.get(name, set())) for name in operator_ids
        },
        "n_no_success_states": n_no_success_states,
        "success_supported_unique_winner_state_counts": {
            name: int(supported_state_winners.get(name, 0)) for name in operator_ids
        },
        "success_supported_unique_winner_task_counts": {
            name: len(supported_task_winners.get(name, set())) for name in operator_ids
        },
        "failure_only_cost_winner_state_counts": {
            name: int(failure_only_state_winners.get(name, 0)) for name in operator_ids
        },
        "failure_only_cost_winner_task_counts": {
            name: len(failure_only_task_winners.get(name, set())) for name in operator_ids
        },
        "winner_count_semantics": (
            "success_supported requires the unique utility winner itself to succeed; "
            "failure_only reports cheaper choices on states where every arm fails"
        ),
    }


def _parse_penalty(value: str) -> tuple[str, float]:
    try:
        name, raw = value.split("=", 1)
        penalty = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected OPERATOR=NONNEGATIVE_FLOAT") from error
    if not name or penalty < 0:
        raise argparse.ArgumentTypeError("expected OPERATOR=NONNEGATIVE_FLOAT")
    return name, penalty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument(
        "--base-penalty",
        action="append",
        default=[],
        type=_parse_penalty,
        metavar="OPERATOR=VALUE",
    )
    parser.add_argument("--sweep-operator", required=True)
    parser.add_argument(
        "--sweep-values",
        default="0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5",
    )
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    registry = _read_json(matrix_dir / "operators.json")
    operator_ids = [str(row["operator_id"]) for row in registry["operators"]]
    if args.sweep_operator not in operator_ids:
        raise SystemExit(f"unknown --sweep-operator: {args.sweep_operator}")
    base = dict(args.base_penalty)
    try:
        sweep_values = [float(value) for value in args.sweep_values.split(",")]
    except ValueError as error:
        raise SystemExit("--sweep-values must be comma-separated floats") from error
    if not sweep_values or any(value < 0 for value in sweep_values):
        raise SystemExit("--sweep-values must be non-empty and non-negative")

    rows = _complete_rows(
        _read_jsonl(matrix_dir / "snapshots.jsonl"),
        _read_jsonl(matrix_dir / "outcomes.jsonl"),
        operator_ids,
    )
    points = []
    for sweep_value in sweep_values:
        penalties = dict(base)
        penalties[args.sweep_operator] = sweep_value
        points.append(
            evaluate_cost_point(
                rows,
                operator_ids,
                penalties,
                success_reward=args.success_reward,
            )
        )
    result = {
        "schema_version": "rase-intervention-cost-sensitivity/v1",
        "interpretation": "diagnostic sensitivity analysis; freeze costs before confirmation",
        "n_complete_states": len(rows),
        "n_tasks": len({row["task_id"] for row in rows}),
        "success_reward": args.success_reward,
        "sweep_operator": args.sweep_operator,
        "points": points,
    }
    output = (args.output or matrix_dir / "cost_sensitivity.json").resolve()
    _write_json(output, result)
    print(json.dumps({"output": str(output), **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

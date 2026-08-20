#!/usr/bin/env python3
"""Evaluate conservative handback policies from immutable task-OOF predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(decisions: list[dict]) -> dict:
    n = len(decisions)
    persistent_successes = sum(row["persistent_success"] for row in decisions)
    successes = sum(row["success"] for row in decisions)
    persistent_steps = sum(row["persistent_steps"] for row in decisions)
    steps = sum(row["steps"] for row in decisions)
    harms = sum(row["persistent_success"] and not row["success"] for row in decisions)
    handbacks = sum(row["handback"] for row in decisions)
    return {
        "n_states": n,
        "success_rate": successes / max(1, n),
        "persistent_success_rate": persistent_successes / max(1, n),
        "success_minus_persistent": (successes - persistent_successes) / max(1, n),
        "executed_oft_steps": steps,
        "persistent_executed_oft_steps": persistent_steps,
        "oft_step_savings_fraction": 1.0 - steps / max(1, persistent_steps),
        "handbacks": handbacks,
        "false_handbacks": harms,
        "false_handback_rate_persistent_rescuable": harms / max(1, persistent_successes),
        "false_handback_rate_conditional": harms / max(1, handbacks),
    }


def evaluate(rows: list[dict], *, threshold: float, z: float, confirmations: int,
             use_risk_agreement: bool, cost_credit: float) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_key"])].append(row)
    decisions = []
    for state, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["elapsed_oft_steps"]))
        run = 0
        selected = None
        for row in values:
            hand_lcb = row["handback_mean"] - z * row["handback_std"]
            risk_safe_lcb = 1.0 - (row["risk_mean"] + z * row["risk_std"])
            persistent_lcb = row["persistent_mean"] - z * row["persistent_std"]
            remaining = row["remaining_teacher_steps"] / max(
                1.0, row["persistent_executed_oft_steps"]
            )
            safe_score = min(hand_lcb, risk_safe_lcb) if use_risk_agreement else hand_lcb
            eligible = (
                safe_score >= threshold
                and safe_score >= persistent_lcb - cost_credit * remaining
            )
            run = run + 1 if eligible else 0
            if run >= confirmations:
                selected = row
                break
        reference = values[0]
        persistent_success = bool(reference["success_if_continue_oft"])
        persistent_steps = int(reference["persistent_executed_oft_steps"])
        decisions.append({
            "state_key": state,
            "persistent_success": persistent_success,
            "persistent_steps": persistent_steps,
            "handback": selected is not None,
            "success": (
                bool(selected["success_if_handback_now"])
                if selected is not None else persistent_success
            ),
            "steps": (
                int(selected["elapsed_oft_steps"])
                if selected is not None else persistent_steps
            ),
        })
    return summarize(decisions)


def pareto(rows: list[dict]) -> list[int]:
    result = []
    for index, row in enumerate(rows):
        if not any(
            other["success_rate"] >= row["success_rate"]
            and other["executed_oft_steps"] <= row["executed_oft_steps"]
            and (
                other["success_rate"] > row["success_rate"]
                or other["executed_oft_steps"] < row["executed_oft_steps"]
            )
            for j, other in enumerate(rows) if j != index
        ):
            result.append(index)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z", type=float, default=1.64)
    parser.add_argument("--cost-credit", type=float, default=0.20)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    results = []
    for use_risk in (False, True):
        for confirmations in (1, 2):
            for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99):
                results.append({
                    "use_risk_agreement": use_risk,
                    "confirmations": confirmations,
                    "threshold": threshold,
                    **evaluate(
                        rows,
                        threshold=threshold,
                        z=args.z,
                        confirmations=confirmations,
                        use_risk_agreement=use_risk,
                        cost_credit=args.cost_credit,
                    ),
                })
    frontier_indices = pareto(results)
    report = {
        "schema_version": "rase-pre-c0-r4-policy-development-sweep/v1",
        "warning": "Development-set structural selection only; requires frozen independent validation.",
        "n_rows": len(rows),
        "n_states": len({row["state_key"] for row in rows}),
        "results": results,
        "pareto_frontier": [results[index] for index in frontier_indices],
        "passing_development_gates": [
            row for row in results
            if row["success_minus_persistent"] >= -0.05
            and row["false_handback_rate_persistent_rescuable"] <= 0.05
            and row["oft_step_savings_fraction"] >= 0.20
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_results": len(results),
        "n_pareto": len(frontier_indices),
        "n_passing_development_gates": len(report["passing_development_gates"]),
        "passing_development_gates": report["passing_development_gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

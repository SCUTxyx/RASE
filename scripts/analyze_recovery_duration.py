#!/usr/bin/env python3
"""Analyze the minimum closed-loop OFT duration needed before Smol handback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze(duration: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    lengths = [int(value) for value in duration["prefix_lengths"]]
    if not lengths or lengths[0] != 0:
        raise ValueError("duration sweep must start at zero")
    direct = {str(row["state_key"]): bool(row["oft_only_success"]) for row in fallback["per_task"]}
    rows = []
    for row in duration["per_state"]:
        state_key = str(row["state_key"])
        outcomes = [bool(arm["success"]) for arm in row["arms"]]
        if len(outcomes) != len(lengths) or state_key not in direct:
            raise ValueError(f"incomplete state join: {state_key}")
        base = outcomes[0]
        successful_durations = [
            length
            for length, success in zip(lengths[1:], outcomes[1:], strict=True)
            if success
        ]
        minimum = min(successful_durations) if successful_durations else None
        rescue = (not base) and minimum is not None
        harmed_durations = [
            length
            for length, success in zip(lengths[1:], outcomes[1:], strict=True)
            if base and not success
        ]
        nonmonotonic = any(
            earlier and not later
            for earlier, later in zip(outcomes[1:-1], outcomes[2:], strict=True)
        )
        rows.append(
            {
                "state_key": state_key,
                "task_id": str(row["task_id"]),
                "suite": str(row["suite"]),
                "cell": (
                    f"{row['perturbation_dimension']}:L{row['perturbation_level']}"
                ),
                "outcomes": dict(zip((str(value) for value in lengths), outcomes, strict=True)),
                "base_success": base,
                "minimum_successful_duration": minimum,
                "fixed_duration_rescue": rescue,
                "harmed_durations": harmed_durations,
                "nonmonotonic_finite_duration": nonmonotonic,
                "direct_oft_success": direct[state_key],
                "direct_only_rescue": (not base) and minimum is None and direct[state_key],
            }
        )
    n = len(rows)
    successes = {
        str(length): sum(row["outcomes"][str(length)] for row in rows) for length in lengths
    }
    base_n = successes["0"]
    fixed_oracle_n = sum(
        any(row["outcomes"][str(length)] for length in lengths[1:]) for row in rows
    )
    fixed_rescues = sum(row["fixed_duration_rescue"] for row in rows)
    rescue_tasks = len({row["task_id"] for row in rows if row["fixed_duration_rescue"]})
    direct_n = sum(row["direct_oft_success"] for row in rows)
    direct_only = sum(row["direct_only_rescue"] for row in rows)
    harms = {
        str(length): sum(
            row["base_success"] and not row["outcomes"][str(length)] for row in rows
        )
        for length in lengths[1:]
    }
    nonmonotonic_states = [
        row["state_key"] for row in rows if row["nonmonotonic_finite_duration"]
    ]
    best_fixed_n = max(successes[str(length)] for length in lengths[1:])
    best_fixed_durations = [
        length for length in lengths[1:] if successes[str(length)] == best_fixed_n
    ]
    structured_signal = fixed_rescues >= 2 and (fixed_oracle_n - base_n) / n >= 0.08
    if structured_signal:
        status = "duration_structure_signal"
        next_step = (
            "Replicate on task-disjoint states, then fit a conservative competence/"
            "termination model against fixed-duration and always-OFT baselines."
        )
    elif direct_only >= 2 and direct_n > fixed_oracle_n:
        status = "episode_persistent_fallback"
        next_step = (
            "Do not fit a handback model yet; use episode-long fallback or distill a "
            "persistent recovery policy, and audit why finite durations fail."
        )
    else:
        status = "not_ready"
        next_step = "No stable duration structure; stop the learned-switching method line."
    return {
        "schema_version": "rase-recovery-duration-audit/v1",
        "status": status,
        "n_states": n,
        "durations": lengths,
        "successes_by_duration": successes,
        "base_successes": base_n,
        "fixed_duration_oracle_successes": fixed_oracle_n,
        "fixed_duration_rescues": fixed_rescues,
        "fixed_duration_rescue_tasks": rescue_tasks,
        "base_harmed_by_duration": harms,
        "nonmonotonic_finite_duration_states": nonmonotonic_states,
        "best_fixed_duration_successes": best_fixed_n,
        "best_fixed_durations": best_fixed_durations,
        "direct_oft_successes": direct_n,
        "direct_only_rescues": direct_only,
        "critic_gate": "closed" if not structured_signal else "replication_required",
        "world_model_gate": "closed",
        "next_step": next_step,
        "per_state": rows,
        "limitations": [
            "Development-only 12-state mechanism audit.",
            "Captured OFT actions are replayed from deterministic snapshots.",
            "The longest finite recovery duration is 64 environment steps.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-summary", type=Path, required=True)
    parser.add_argument("--fallback-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.duration_summary.read_text()),
        json.loads(args.fallback_analysis.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"per_state", "limitations"}
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

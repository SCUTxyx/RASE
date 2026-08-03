#!/usr/bin/env python3
"""Decide whether short expert replans or persistent fallback are supported."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze(prefix: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    direct = {str(row["state_key"]): bool(row["oft_only_success"]) for row in fallback["per_task"]}
    rows = []
    for row in prefix["per_state"]:
        state_key = str(row["state_key"])
        if state_key not in direct:
            raise ValueError(f"missing direct OFT outcome for {state_key}")
        arms = list(row["arms"])
        base = bool(arms[0]["success"])
        short = any(bool(arm["success"]) for arm in arms[1:])
        rows.append(
            {
                "state_key": state_key,
                "task_id": str(row["task_id"]),
                "base_smol": base,
                "short_oft_prefix_then_smol": short,
                "direct_oft": direct[state_key],
                "short_prefix_rescue": (not base) and short,
                "direct_only_rescue": (not base) and (not short) and direct[state_key],
            }
        )
    n = len(rows)
    base_n = sum(row["base_smol"] for row in rows)
    short_n = sum(row["short_oft_prefix_then_smol"] for row in rows)
    direct_n = sum(row["direct_oft"] for row in rows)
    short_rescues = sum(row["short_prefix_rescue"] for row in rows)
    direct_only = sum(row["direct_only_rescue"] for row in rows)
    rescue_tasks = len({row["task_id"] for row in rows if row["short_prefix_rescue"]})
    short_signal = short_rescues >= 2 and (short_n - base_n) / n >= 0.08
    if short_signal:
        status = "recovery_prefix_model_signal"
        recommendation = (
            "Distill or fine-tune a bounded recovery-prefix policy; keep direct OFT as "
            "the safety fallback and test on task-disjoint states before training a ranker."
        )
    elif direct_only >= 2 and direct_n > short_n:
        status = "persistent_fallback_required"
        recommendation = (
            "Do not train a short resample/replan model. Use persistent OFT escalation, "
            "or distill the full OFT policy with perturbation-aware data."
        )
    else:
        status = "not_ready"
        recommendation = "Increase generator diversity before any learned selector or world model."
    return {
        "schema_version": "rase-replan-mechanism-audit/v1",
        "status": status,
        "n_states": n,
        "base_smol_successes": base_n,
        "short_prefix_oracle_successes": short_n,
        "direct_oft_successes": direct_n,
        "short_prefix_rescues": short_rescues,
        "short_prefix_rescue_tasks": rescue_tasks,
        "direct_only_rescues": direct_only,
        "recommendation": recommendation,
        "world_model_gate": "closed",
        "per_state": rows,
        "limitations": [
            "Development-only 12-state mechanism audit.",
            "OFT prefixes are open-loop portions of one initial chunk.",
            "Direct OFT has a larger compute budget than prefix-transfer arms.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-summary", type=Path, required=True)
    parser.add_argument("--fallback-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.prefix_summary.read_text()),
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

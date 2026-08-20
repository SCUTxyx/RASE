#!/usr/bin/env python3
"""Calibrated safe-handback baselines over a PRE-B dataset (no world model)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def _fixed_duration_policy(rows: list[dict[str, Any]], h: int) -> dict[str, Any]:
    chosen = [row for row in rows if int(row["elapsed_recovery_steps"]) == int(h)]
    if not chosen:
        return {"h": h, "n": 0, "success_rate": None, "harm_rate": None}
    success = np.mean([row["handback_success"] for row in chosen])
    base = [row for row in chosen if row["base_success_at_h0"]]
    harm = (
        float(np.mean([row["false_handback_harm"] for row in base])) if base else 0.0
    )
    return {
        "h": h,
        "n": len(chosen),
        "success_rate": float(success),
        "harm_rate": harm,
        "mean_oft_steps": float(h),
    }


def _threshold_policy(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    feature: str = "elapsed_recovery_steps",
) -> dict[str, Any]:
    """Choose the smallest duration threshold that maximizes train net utility."""
    durations = sorted({int(row[feature]) for row in train})
    best = None
    for threshold in durations:
        # hand back at first duration >= threshold for each state
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train:
            by_state[row["state_key"]].append(row)
        utilities = []
        for state_rows in by_state.values():
            ordered = sorted(state_rows, key=lambda r: int(r[feature]))
            selected = next(
                (row for row in ordered if int(row[feature]) >= threshold),
                ordered[-1],
            )
            utilities.append(
                float(selected["handback_success"])
                - 0.25 * (int(selected[feature]) / max(durations))
                - 1.0 * float(selected["false_handback_harm"])
            )
        score = float(np.mean(utilities)) if utilities else float("-inf")
        if best is None or score > best["train_utility"]:
            best = {"threshold": threshold, "train_utility": score}
    assert best is not None

    by_state = defaultdict(list)
    for row in test:
        by_state[row["state_key"]].append(row)
    selected_rows = []
    for state_rows in by_state.values():
        ordered = sorted(state_rows, key=lambda r: int(r[feature]))
        selected_rows.append(
            next(
                (row for row in ordered if int(row[feature]) >= best["threshold"]),
                ordered[-1],
            )
        )
    success = float(np.mean([row["handback_success"] for row in selected_rows]))
    harm = float(np.mean([row["false_handback_harm"] for row in selected_rows]))
    cost = float(np.mean([row["elapsed_recovery_steps"] for row in selected_rows]))
    return {
        "name": "calibrated_duration_threshold",
        "threshold": best["threshold"],
        "train_utility": best["train_utility"],
        "test_n_states": len(selected_rows),
        "test_success_rate": success,
        "test_harm_rate": harm,
        "test_mean_oft_steps": cost,
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = list(dataset["rows"])
    train = _split_rows(rows, "train") + _split_rows(rows, "val")
    test = _split_rows(rows, "test")
    if not test:
        test = _split_rows(rows, "val") or rows
        test_name = "fallback_nonhidden"
    else:
        test_name = "test"
    durations = sorted({int(row["elapsed_recovery_steps"]) for row in rows})
    fixed = [_fixed_duration_policy(test, h) for h in durations]
    best_fixed = max(
        (row for row in fixed if row["success_rate"] is not None),
        key=lambda row: (row["success_rate"], -row["mean_oft_steps"]),
    )
    threshold = _threshold_policy(train or rows, test)
    always_oft = {
        "name": "always_oft_reference",
        "note": "episode-long OFT is an upper bound; not a handback policy",
        "test_success_rate": float(
            np.mean([row["direct_oft_success"] for row in test])
        )
        if test
        else None,
    }
    return {
        "schema_version": "rase-safe-handback-baselines/v1",
        "eval_split": test_name,
        "fixed_duration": fixed,
        "best_fixed_duration": best_fixed,
        "calibrated_threshold": threshold,
        "always_oft": always_oft,
        "world_model_used": False,
        "trainable_selector_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-gate", type=Path, default=None)
    args = parser.parse_args()

    if args.method_gate is not None:
        gate = json.loads(args.method_gate.read_text(encoding="utf-8"))
        if gate.get("termination_model_gate") != "open":
            raise SystemExit(
                "termination_model_gate is closed; refusing to train/evaluate "
                f"method baselines (decision={gate.get('decision')})"
            )

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = evaluate(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"eval_split": result["eval_split"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

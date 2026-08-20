#!/usr/bin/env python3
"""Compare two exact-state R5 boundary collections for repeatability."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = f"{row['state_key']}:{int(row['elapsed_oft_steps'])}"
        if key in rows:
            raise ValueError(f"duplicate boundary key: {key}")
        rows[key] = row
    return rows


def max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def first_success(rows: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        grouped[str(row["state_key"])].append(row)
    result: dict[str, int | None] = {}
    for state, items in grouped.items():
        successful = [
            int(row["elapsed_oft_steps"])
            for row in items
            if bool(row["success_if_handback_now"])
        ]
        result[state] = min(successful) if successful else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = read_jsonl(args.left)
    right = read_jsonl(args.right)
    left_keys, right_keys = set(left), set(right)
    common = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)

    label_fields = [
        "success_if_handback_now",
        "success_if_continue_oft",
        "student_failure_risk",
        "student_step_success",
        "oft_step_success",
    ]
    label_matches = {
        field: sum(bool(left[key][field]) == bool(right[key][field]) for key in common)
        for field in label_fields
    }
    by_boundary: dict[str, dict[str, int]] = {}
    for horizon in sorted({int(left[key]["elapsed_oft_steps"]) for key in common}):
        keys = [key for key in common if int(left[key]["elapsed_oft_steps"]) == horizon]
        matches = sum(
            bool(left[key]["success_if_handback_now"])
            == bool(right[key]["success_if_handback_now"])
            for key in keys
        )
        by_boundary[str(horizon)] = {"matches": matches, "total": len(keys)}

    left_first, right_first = first_success(left), first_success(right)
    state_keys = sorted(left_first)
    changed_states = [state for state in state_keys if left_first[state] != right_first[state]]
    vector_fields = [
        "latent",
        "next_latent_student",
        "next_latent_oft",
        "student_action",
        "student_action_chunk",
        "oft_action",
        "proprio",
    ]
    vector_diffs = {
        field: max(max_abs(left[key][field], right[key][field]) for key in common)
        for field in vector_fields
    }
    report = {
        "schema_version": "rase-pre-c0-r5-boundary-repeatability/v1",
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "n_rows": len(common),
        "n_rows_left": len(left),
        "n_rows_right": len(right),
        "left_only_boundary_keys": left_only,
        "right_only_boundary_keys": right_only,
        "boundary_key_sets_match": not left_only and not right_only,
        "n_states": len(state_keys),
        "handback_label_matches": label_matches["success_if_handback_now"],
        "handback_label_agreement": (
            label_matches["success_if_handback_now"] / len(common) if common else 0.0
        ),
        "label_matches": label_matches,
        "handback_agreement_by_boundary": by_boundary,
        "minimum_successful_boundary_exact_matches": len(state_keys) - len(changed_states),
        "minimum_successful_boundary_agreement": (
            (len(state_keys) - len(changed_states)) / len(state_keys) if state_keys else 0.0
        ),
        "changed_states": {
            state: {"left": left_first[state], "right": right_first[state]}
            for state in changed_states
        },
        "minimum_successful_boundary_counts_left": dict(Counter(map(str, left_first.values()))),
        "minimum_successful_boundary_counts_right": dict(Counter(map(str, right_first.values()))),
        "maximum_absolute_feature_difference": vector_diffs,
        "exact_feature_repeatability": all(value == 0.0 for value in vector_diffs.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare a deferred-switch replay with its calibration reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["state_key"]): row for row in payload.get("per_state") or []
    }


def audit(reference: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    reference_rows = _index(reference)
    replay_rows = _index(replay)
    if not replay_rows or not set(replay_rows).issubset(reference_rows):
        raise ValueError("replay state keys must be a non-empty reference subset")
    expected = {
        key
        for key, row in reference_rows.items()
        if row.get("classification") in {"direct_only", "deferred_only"}
    }
    if set(replay_rows) != expected:
        raise ValueError("replay coverage must equal the reference disagreement set")
    comparisons = []
    for key in sorted(replay_rows):
        before = reference_rows[key]
        after = replay_rows[key]
        outcome_match = all(
            bool(before[field]) == bool(after[field])
            for field in ("direct_oft_success", "decision_suffix_oft_success")
        )
        before_suffix = next(
            arm for arm in before["arms"] if arm["arm_label"] == "decision_suffix_oft"
        )
        after_suffix = next(
            arm for arm in after["arms"] if arm["arm_label"] == "decision_suffix_oft"
        )
        comparisons.append(
            {
                "state_key": key,
                "reference_classification": before["classification"],
                "replay_classification": after["classification"],
                "outcome_match": outcome_match,
                "prefix_sha256_match": before_suffix["prefix_sha256"]
                == after_suffix["prefix_sha256"],
            }
        )
    exact = all(
        row["outcome_match"] and row["prefix_sha256_match"] for row in comparisons
    )
    return {
        "schema_version": "rase-deferred-switch-replay-audit/v1",
        "status": "pass" if exact else "mismatch",
        "n_states": len(comparisons),
        "exact_outcome_and_prefix_match": exact,
        "comparisons": comparisons,
        "interpretation": (
            "Deterministic replay checks implementation stability only; it is not an "
            "independent seed or episode confirmation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(_read(args.reference), _read(args.replay))
    _write(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "n_states": result["n_states"],
                "exact_outcome_and_prefix_match": result[
                    "exact_outcome_and_prefix_match"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

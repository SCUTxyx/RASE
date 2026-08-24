#!/usr/bin/env python3
"""Summarize matched-suite OFT calibration repeats and enforce the B-1 gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUITES = ("spatial", "object", "goal", "long")


def load_summary(root: Path, repeat: str, suite: str) -> dict[str, Any]:
    path = root / repeat / suite / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", nargs="+", default=["a", "b"])
    parser.add_argument("--min-overall-success", type=float, default=0.70)
    parser.add_argument("--max-suite-unrecoverable", type=float, default=0.50)
    args = parser.parse_args()

    root = args.input_dir.resolve()
    rows: list[dict[str, Any]] = []
    state_outcomes: dict[str, dict[str, bool]] = {}
    for repeat in args.repeats:
        for suite in SUITES:
            summary = load_summary(root, repeat, suite)
            for row in summary["per_state"]:
                key = str(row["state_key"])
                success = bool(row["direct_oft_success"])
                rows.append(
                    {"repeat": repeat, "suite": suite, "state_key": key, "success": success}
                )
                state_outcomes.setdefault(key, {})[repeat] = success

    expected_per_repeat = 12
    repeat_counts = Counter(row["repeat"] for row in rows)
    complete = all(repeat_counts[repeat] == expected_per_repeat for repeat in args.repeats)
    first = args.repeats[0]
    first_rows = [row for row in rows if row["repeat"] == first]
    hits = sum(row["success"] for row in first_rows)
    overall_success = hits / len(first_rows)
    per_suite = {}
    for suite in SUITES:
        selected = [row for row in first_rows if row["suite"] == suite]
        suite_hits = sum(row["success"] for row in selected)
        per_suite[suite] = {
            "hits": suite_hits,
            "trials": len(selected),
            "success_rate": suite_hits / len(selected),
            "unrecoverable_rate": 1.0 - suite_hits / len(selected),
        }
    drift_states = {
        key: outcomes
        for key, outcomes in state_outcomes.items()
        if len(outcomes) == len(args.repeats) and len(set(outcomes.values())) > 1
    }
    checks = {
        "complete_repeats": complete,
        "state_sets_identical": all(len(v) == len(args.repeats) for v in state_outcomes.values()),
        "overall_success_ge_threshold": overall_success >= args.min_overall_success,
        "each_suite_unrecoverable_le_threshold": all(
            value["unrecoverable_rate"] <= args.max_suite_unrecoverable
            for value in per_suite.values()
        ),
        "zero_binary_outcome_drift": not drift_states,
    }
    artifact = {
        "schema_version": "rase-e3b-teacher-calibration/v1",
        "status": "complete",
        "input_dir": str(root),
        "repeats": args.repeats,
        "n_unique_states": len(state_outcomes),
        "overall": {
            "hits": hits,
            "trials": len(first_rows),
            "success_rate": overall_success,
        },
        "per_suite": per_suite,
        "outcome_drift": {"n_states": len(drift_states), "states": drift_states},
        "thresholds": {
            "min_overall_success": args.min_overall_success,
            "max_suite_unrecoverable": args.max_suite_unrecoverable,
        },
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": artifact["decision"], **artifact["overall"]}, sort_keys=True))
    return 0 if artifact["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

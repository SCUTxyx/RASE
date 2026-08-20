#!/usr/bin/env python3
"""Analyze the development-only K=8 multiscale temperature probe."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def analyze(
    keys: dict[str, Any],
    rollout: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    ordered_keys = [str(value) for value in keys.get("state_keys") or []]
    records = {str(row["state_key"]): dict(row) for row in keys.get("records") or []}
    if keys.get("selection_uses_outcomes") is not False:
        raise ValueError("probe cohort must be metadata-only")
    if not ordered_keys or set(records) != set(ordered_keys):
        raise ValueError("invalid frozen probe cohort")
    provenance = dict(rollout.get("state_keys_provenance") or {})
    if provenance.get("selected_state_keys_sha256") != keys.get("state_keys_sha256"):
        raise ValueError("rollout checksum does not match probe cohort")
    if rollout.get("continuation_seed_mode") != "common_root_rollout":
        raise ValueError("probe requires common_root_rollout")

    schedule = tuple(float(value) for value in generation.get("temperatures") or [])
    k = int(generation.get("k", 0))
    if k != 8 or len(schedule) != k:
        raise ValueError("G1 probe requires an explicit K=8 temperature schedule")
    if generation.get("state_keys") != ordered_keys:
        raise ValueError("generation state order does not match frozen cohort")
    rollout_rows = {
        str(row["state_key"]): dict(row) for row in rollout.get("per_state") or []
    }
    if set(rollout_rows) != set(ordered_keys):
        raise ValueError("rollout state set does not match frozen cohort")

    rows: list[dict[str, Any]] = []
    by_temperature: dict[float, list[bool]] = defaultdict(list)
    by_index: list[list[bool]] = [[] for _ in range(k)]
    for state_key in ordered_keys:
        candidates = list(rollout_rows[state_key].get("candidates") or [])
        if len(candidates) != k:
            raise ValueError(f"{state_key} expected K={k}, got {len(candidates)}")
        outcomes: list[bool] = []
        for index, candidate in enumerate(candidates):
            if int(candidate.get("trials", -1)) != 1:
                raise ValueError("development probe requires one rollout per candidate")
            success = int(candidate.get("successes", 0)) == 1
            outcomes.append(success)
            by_temperature[schedule[index]].append(success)
            by_index[index].append(success)
        first = outcomes[0]
        oracle = any(outcomes)
        rows.append({
            "state_key": state_key,
            "task_id": str(records[state_key]["task_id"]),
            "suite": str(records[state_key]["suite"]),
            "step": int(records[state_key]["step"]),
            "successes": outcomes,
            "first_success": first,
            "oracle_success": oracle,
            "first_fail_later_success": (not first) and oracle,
            "mixed_outcomes": len(set(outcomes)) > 1,
        })

    n = len(rows)
    first_hits = sum(row["first_success"] for row in rows)
    oracle_hits = sum(row["oracle_success"] for row in rows)
    rescues = [row for row in rows if row["first_fail_later_success"]]
    mixed = sum(row["mixed_outcomes"] for row in rows)
    proceed = len(rescues) >= 2
    return {
        "artifact_version": "rase-g1-multiscale-temperature-probe/v1",
        "status": (
            "pass_proceed_to_independent_confirmation"
            if proceed else "fail_do_not_scale"
        ),
        "decision_scope": (
            "development-only generator selection; these roots may not be used "
            "for held-out confirmation"
        ),
        "n_states": n,
        "n_tasks": len({row["task_id"] for row in rows}),
        "k": k,
        "temperature_schedule": list(schedule),
        "gate": {
            "name": "at_least_2_first_fail_later_success_roots",
            "threshold": 2,
            "observed": len(rescues),
            "passed": proceed,
        },
        "metrics": {
            "first_successes": first_hits,
            "first_success_rate": first_hits / n,
            "oracle_successes": oracle_hits,
            "oracle_success_rate": oracle_hits / n,
            "oracle_minus_first": (oracle_hits - first_hits) / n,
            "first_fail_later_successes": len(rescues),
            "rescue_tasks": len({row["task_id"] for row in rescues}),
            "mixed_roots": mixed,
            "mixed_root_rate": mixed / n,
            "per_temperature": {
                str(temperature): {
                    "successes": sum(values),
                    "trials": len(values),
                    "rate": sum(values) / len(values),
                }
                for temperature, values in sorted(by_temperature.items())
            },
            "per_candidate_index": [
                {
                    "index": index,
                    "temperature": schedule[index],
                    "successes": sum(values),
                    "trials": len(values),
                    "rate": sum(values) / len(values),
                }
                for index, values in enumerate(by_index)
            ],
        },
        "per_state": rows,
        "provenance": {
            "state_keys_sha256": keys.get("state_keys_sha256"),
            "selection_uses_outcomes": False,
            "continuation_seed_mode": rollout.get("continuation_seed_mode"),
            "candidate_artifact_version": 2,
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    m = result["metrics"]
    lines = [
        "# RASE G1 multiscale temperature probe",
        "",
        f"- Status: `{result['status']}`",
        f"- Development cohort: {result['n_states']} states / {result['n_tasks']} tasks",
        f"- Schedule: `{result['temperature_schedule']}`",
        f"- First candidate: {m['first_successes']}/{result['n_states']} ({m['first_success_rate']:.1%})",
        f"- Oracle@8: {m['oracle_successes']}/{result['n_states']} ({m['oracle_success_rate']:.1%})",
        f"- Oracle gain: {m['oracle_minus_first']:+.1%}",
        f"- First-fail/later-success roots: {m['first_fail_later_successes']} across {m['rescue_tasks']} tasks",
        f"- Mixed roots: {m['mixed_roots']}/{result['n_states']} ({m['mixed_root_rate']:.1%})",
        f"- Gate (>=2 rescues): `{'PASS' if result['gate']['passed'] else 'FAIL'}`",
        "",
        "## Per-temperature quality",
        "",
    ]
    for temperature, values in m["per_temperature"].items():
        lines.append(
            f"- T={temperature}: {values['successes']}/{values['trials']} "
            f"({values['rate']:.1%})"
        )
    lines.extend([
        "",
        "This is a development-only generator-selection result, not held-out evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        _read(args.keys), _read(args.rollout_summary), _read(args.generation_summary)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gate": result["gate"],
        "metrics": {
            "first_success_rate": result["metrics"]["first_success_rate"],
            "oracle_success_rate": result["metrics"]["oracle_success_rate"],
            "oracle_minus_first": result["metrics"]["oracle_minus_first"],
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

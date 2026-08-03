#!/usr/bin/env python3
"""Audit Phase 0H suffix-prefix curves and their Phase 0G endpoint identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use SUITE=PATH")
    suite, raw_path = value.split("=", 1)
    return suite, Path(raw_path)


def _source_endpoint(row: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [arm for arm in row.get("arms") or [] if arm.get("arm_label") == label]
    if len(matches) != 1:
        raise ValueError(f"source row requires exactly one {label} arm")
    return dict(matches[0])


def _curve_class(source_classification: str, flips: list[int]) -> str:
    if len(flips) == 1:
        return (
            "single_transition_direct_harm"
            if source_classification == "direct_only"
            else "single_transition_deferred_rescue"
        )
    return "nonmonotonic_multi_transition"


def analyze(
    cohort: dict[str, Any],
    source: dict[str, Any],
    summaries: list[tuple[str, dict[str, Any]]],
    *,
    expected_suffix_steps: int,
) -> dict[str, Any]:
    if cohort.get("schema_version") != "rase-timing-disagreement-cohort/v1":
        raise ValueError("unexpected cohort schema")
    if not cohort.get("selection_outcome_conditioned"):
        raise ValueError("Phase 0H cohort must disclose outcome-conditioned selection")
    if source.get("schema_version") != "rase-deferred-switch-analysis/v1":
        raise ValueError("unexpected source-analysis schema")
    source_rows = {str(row["state_key"]): row for row in source.get("per_state") or []}
    keys = [str(key) for key in cohort.get("state_keys") or []]
    expected_labels = {
        str(key): str(value)
        for key, value in (cohort.get("expected_source_classification") or {}).items()
    }
    if set(keys) != set(expected_labels):
        raise ValueError("cohort classification map does not match state keys")

    observed_rows: dict[str, dict[str, Any]] = {}
    for suite, summary in summaries:
        if summary.get("schema_version") != (
            "rase-oft-decision-suffix-prefix-grid/v1"
        ):
            raise ValueError(f"unexpected Phase 0H summary schema for {suite}")
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete Phase 0H summary for {suite}")
        if summary.get("state_keys_sha256") != cohort.get("state_keys_sha256"):
            raise ValueError(f"state-key provenance mismatch for {suite}")
        if summary.get("suite") not in {None, suite}:
            raise ValueError(f"suite mismatch: {summary.get('suite')} != {suite}")
        for row in summary.get("per_state") or []:
            key = str(row["state_key"])
            if key in observed_rows:
                raise ValueError(f"duplicate Phase 0H state: {key}")
            observed_rows[key] = dict(row)
    if set(observed_rows) != set(keys):
        raise ValueError(
            f"Phase 0H state union mismatch: expected={sorted(keys)} "
            f"observed={sorted(observed_rows)}"
        )

    per_state = []
    endpoint_mismatches = []
    for key in keys:
        source_row = source_rows.get(key)
        if source_row is None:
            raise ValueError(f"state missing from source analysis: {key}")
        source_class = str(source_row.get("classification"))
        if source_class != expected_labels[key]:
            raise ValueError(
                f"source classification changed for {key}: "
                f"{source_class} != {expected_labels[key]}"
            )
        row = observed_rows[key]
        arms = {int(arm["prefix_steps"]): arm for arm in row.get("arms") or []}
        expected_steps = set(range(expected_suffix_steps + 1))
        if set(arms) != expected_steps:
            raise ValueError(f"incomplete k-grid for {key}")
        if any(not bool(arms[step].get("prefix_completed")) for step in expected_steps):
            raise ValueError(f"incomplete executed prefix for {key}")
        source_direct = _source_endpoint(source_row, "direct_oft")
        source_deferred = _source_endpoint(source_row, "decision_suffix_oft")
        endpoint_checks = {
            "k0_success": bool(arms[0]["success"]) == bool(source_direct["success"]),
            "kT_success": bool(arms[expected_suffix_steps]["success"])
            == bool(source_deferred["success"]),
            "k0_prefix_sha256": arms[0]["prefix_sha256"]
            == source_direct["prefix_sha256"],
            "kT_prefix_sha256": arms[expected_suffix_steps]["prefix_sha256"]
            == source_deferred["prefix_sha256"],
        }
        if not all(endpoint_checks.values()):
            endpoint_mismatches.append({"state_key": key, **endpoint_checks})
        successes = [bool(arms[step]["success"]) for step in sorted(expected_steps)]
        flips = [
            step
            for step in range(1, expected_suffix_steps + 1)
            if successes[step] != successes[step - 1]
        ]
        per_state.append(
            {
                "state_key": key,
                "suite": row.get("suite"),
                "dim": row.get("dim"),
                "level": row.get("level"),
                "episode_id": row.get("episode_id"),
                "source_classification": source_class,
                "success_pattern": "".join("1" if hit else "0" for hit in successes),
                "success_flip_steps": flips,
                "n_success_flips": len(flips),
                "curve_classification": _curve_class(source_class, flips),
                "endpoint_checks": endpoint_checks,
                "prefix_translation_l2_sum": [
                    float(arms[step].get("prefix_translation_l2_sum", 0.0))
                    for step in sorted(expected_steps)
                ],
                "prefix_rotation_l2_sum": [
                    float(arms[step].get("prefix_rotation_l2_sum", 0.0))
                    for step in sorted(expected_steps)
                ],
                "prefix_gripper_abs_sum": [
                    float(arms[step].get("prefix_gripper_abs_sum", 0.0))
                    for step in sorted(expected_steps)
                ],
                "continuation_steps": [
                    int(arms[step]["continuation_steps"])
                    for step in sorted(expected_steps)
                ],
                "terminal_during_prefix": [
                    bool(arms[step]["terminal_during_prefix"])
                    for step in sorted(expected_steps)
                ],
            }
        )

    curve_counts = Counter(row["curve_classification"] for row in per_state)
    pattern_counts = Counter(row["success_pattern"] for row in per_state)
    single_boundaries = [
        {
            "state_key": row["state_key"],
            "source_classification": row["source_classification"],
            "prefix_steps": row["success_flip_steps"][0],
            "translation_l2_sum": row["prefix_translation_l2_sum"][
                row["success_flip_steps"][0]
            ],
            "rotation_l2_sum": row["prefix_rotation_l2_sum"][
                row["success_flip_steps"][0]
            ],
            "gripper_abs_sum": row["prefix_gripper_abs_sum"][
                row["success_flip_steps"][0]
            ],
        }
        for row in per_state
        if row["n_success_flips"] == 1
    ]
    boundary_steps = sorted({row["prefix_steps"] for row in single_boundaries})
    shared_scalar_boundary = (
        len(single_boundaries) == len(per_state) and len(boundary_steps) == 1
    )
    success_by_k = {
        str(step): {
            "hits": sum(row["success_pattern"][step] == "1" for row in per_state),
            "trials": len(per_state),
        }
        for step in range(expected_suffix_steps + 1)
    }
    status = "complete" if not endpoint_mismatches else "invalid_endpoint_parity"
    return {
        "schema_version": "rase-suffix-prefix-mechanism-analysis/v1",
        "status": status,
        "selection_outcome_conditioned": True,
        "claim_scope": "exploratory selected-subset mechanism audit only",
        "n_states": len(per_state),
        "expected_suffix_steps": expected_suffix_steps,
        "endpoint_parity": {
            "status": "pass" if not endpoint_mismatches else "fail",
            "n_pass": len(per_state) - len(endpoint_mismatches),
            "n_total": len(per_state),
            "mismatches": endpoint_mismatches,
        },
        "curve_classification_counts": dict(sorted(curve_counts.items())),
        "success_pattern_counts": dict(sorted(pattern_counts.items())),
        "success_by_prefix_steps": success_by_k,
        "single_transition_fraction": (
            sum(row["n_success_flips"] == 1 for row in per_state) / len(per_state)
        ),
        "single_transition_boundaries": single_boundaries,
        "shared_scalar_boundary": {
            "status": "pass" if shared_scalar_boundary else "fail",
            "boundary_prefix_steps": boundary_steps,
            "requires_all_states_single_transition": True,
        },
        "scientific_decision": {
            "status": (
                "targeted_independent_screen_may_be_designed"
                if shared_scalar_boundary
                else "close_timing_selector_use_immediate_oft"
            ),
            "reasons": (
                []
                if shared_scalar_boundary
                else [
                    f"{len(per_state) - len(single_boundaries)}/{len(per_state)} "
                    "selected states have non-monotonic multi-transition curves",
                    f"single-transition boundary steps are {boundary_steps}, not uniform",
                ]
            ),
            "authorizes_model_training": False,
        },
        "per_state": per_state,
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# RASE-UI Phase 0H suffix-prefix mechanism audit",
        "",
        f"Status: **{result['status']}**",
        "",
        "> Exploratory, outcome-selected six-state subset. These rates are not a "
        "population estimate and do not evaluate a selector.",
        "",
        "## Aggregate",
        "",
        f"- Endpoint identity: {result['endpoint_parity']['n_pass']}/"
        f"{result['endpoint_parity']['n_total']}.",
        f"- Single-transition fraction: {result['single_transition_fraction']:.3f}.",
        f"- Curve classes: `{json.dumps(result['curve_classification_counts'], sort_keys=True)}`.",
        f"- Success patterns: `{json.dumps(result['success_pattern_counts'], sort_keys=True)}`.",
        f"- Shared scalar boundary: {result['shared_scalar_boundary']['status']}; "
        f"observed single-transition steps "
        f"`{result['shared_scalar_boundary']['boundary_prefix_steps']}`.",
        f"- Decision: **{result['scientific_decision']['status']}**.",
        "",
        "## Per-state curves",
        "",
        "| state | suite | cell | Phase 0G | k=0..5 | flip k | class |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["per_state"]:
        flips = ",".join(str(step) for step in row["success_flip_steps"])
        lines.append(
            f"| `{row['state_key']}` | {row['suite']} | {row['dim']}:L{row['level']} | "
            f"{row['source_classification']} | `{row['success_pattern']}` | "
            f"{flips} | {row['curve_classification']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A single transition suggests a stable timing boundary on this selected state. "
            "Multiple transitions imply a non-monotonic/task-specific response and do not "
            "support a scalar wait-time mechanism. Endpoint SHA and success parity must pass "
            "before interpreting any interior k.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--summary", action="append", type=_named_path, required=True)
    parser.add_argument("--expected-suffix-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    source_path = args.source_analysis.resolve()
    cohort = _load(cohort_path)
    if cohort.get("source_analysis_sha256") != _file_sha256(source_path):
        raise SystemExit("source-analysis SHA-256 does not match frozen cohort")
    result = analyze(
        cohort,
        _load(source_path),
        [(suite, _load(path.resolve())) for suite, path in args.summary],
        expected_suffix_steps=args.expected_suffix_steps,
    )
    result["cohort_sha256"] = _file_sha256(cohort_path)
    result["source_analysis_sha256"] = _file_sha256(source_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "endpoint_parity": result["endpoint_parity"]["status"],
                "curve_classification_counts": result[
                    "curve_classification_counts"
                ],
                "success_pattern_counts": result["success_pattern_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

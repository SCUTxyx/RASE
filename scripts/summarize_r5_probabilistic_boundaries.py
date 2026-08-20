#!/usr/bin/env python3
"""Audit and summarize same-snapshot K-repeat R5 handback labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_boundary_is_violation(boundary: int, executed_oft_steps: int) -> bool:
    """A handback boundary is reachable only before the persistent run terminates."""
    return executed_oft_steps < 0 or boundary < executed_oft_steps


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional frozen selection manifest used to audit state/task/boundary coverage.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opportunity-min-savings", type=float, default=0.20)
    parser.add_argument("--opportunity-min-finite-states", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    collection = json.loads(args.collection_report.read_text())
    required = {
        "state_key", "task_id", "suite", "elapsed_oft_steps",
        "handback_repeats", "handback_repeat_seeds", "handback_repeat_successes",
        "handback_repeat_steps", "handback_repeat_stop_reasons",
        "handback_success_count", "handback_success_probability",
        "handback_success_wilson_lcb_95_one_sided",
        "handback_success_wilson_ucb_95_one_sided",
    }
    malformed: list[dict[str, Any]] = []
    duplicate_seeds = 0
    probabilities: list[float] = []
    lcbs: list[float] = []
    entropies: list[float] = []
    success_steps: list[float] = []
    failure_steps: list[float] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_boundary: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        repeats = int(row.get("handback_repeats", 0))
        seeds = list(row.get("handback_repeat_seeds", []))
        outcomes = [bool(value) for value in row.get("handback_repeat_successes", [])]
        steps = list(row.get("handback_repeat_steps", []))
        reasons = list(row.get("handback_repeat_stop_reasons", []))
        consistent = (
            repeats > 0
            and len(seeds) == len(outcomes) == len(steps) == len(reasons) == repeats
            and sum(outcomes) == int(row.get("handback_success_count", -1))
            and abs(sum(outcomes) / repeats - float(row.get("handback_success_probability", -1.0))) < 1e-12
        )
        if missing or not consistent:
            malformed.append({"row": index, "missing": missing, "repeat_fields_consistent": consistent})
            continue
        duplicate_seeds += int(len(set(seeds)) != repeats)
        probability = float(row["handback_success_probability"])
        probabilities.append(probability)
        lcbs.append(float(row["handback_success_wilson_lcb_95_one_sided"]))
        entropies.append(entropy(probability))
        for outcome, step in zip(outcomes, steps):
            (success_steps if outcome else failure_steps).append(float(step))
        by_suite[str(row["suite"])].append(row)
        by_boundary[int(row["elapsed_oft_steps"])].append(row)
        by_state[str(row["state_key"])].append(row)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        ps = [float(row["handback_success_probability"]) for row in group]
        group_lcbs = [float(row["handback_success_wilson_lcb_95_one_sided"]) for row in group]
        return {
            "n_boundaries": len(group),
            "n_trials": sum(int(row["handback_repeats"]) for row in group),
            "mean_success_probability": float(statistics.fmean(ps)) if ps else None,
            "mean_bernoulli_entropy_bits": (
                float(statistics.fmean(entropy(probability) for probability in ps))
                if ps else None
            ),
            "nondegenerate_boundaries": sum(0.0 < p < 1.0 for p in ps),
            "all_success_boundaries": sum(p == 1.0 for p in ps),
            "all_failure_boundaries": sum(p == 0.0 for p in ps),
            "max_wilson_lcb": max(group_lcbs) if group_lcbs else None,
        }

    suites = sorted(by_suite)
    expected_suites = {"Spatial", "Object", "Goal", "Long"}
    n_states = len({str(row["state_key"]) for row in rows})
    persistent_matches = int(collection.get("persistent_replay_matches", -1))
    persistent_total = int(collection.get("n_states", -1))
    protocol_reasons = []
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    expected_states = {
        str(row["state_key"]) for row in (manifest or {}).get("records", [])
    }
    observed_states = set(by_state)
    missing_manifest_states = sorted(expected_states - observed_states)
    unexpected_states = sorted(observed_states - expected_states) if manifest else []
    if malformed:
        protocol_reasons.append(f"{len(malformed)} malformed repeat rows")
    if duplicate_seeds:
        protocol_reasons.append(f"{duplicate_seeds} rows contain duplicate repeat seeds")
    if persistent_matches != persistent_total:
        protocol_reasons.append(f"persistent replay parity {persistent_matches}/{persistent_total}")
    if set(suites) != expected_suites:
        protocol_reasons.append(f"suite coverage is {suites}, expected {sorted(expected_suites)}")
    if manifest and missing_manifest_states:
        protocol_reasons.append(f"{len(missing_manifest_states)} manifest states have no recorded boundary")
    if manifest and unexpected_states:
        protocol_reasons.append(f"{len(unexpected_states)} states are absent from the frozen manifest")
    if manifest:
        expected_tasks = {str(row["task_id"]) for row in manifest["records"]}
        observed_tasks = {str(row["task_id"]) for row in rows}
        if observed_tasks != expected_tasks:
            protocol_reasons.append(
                f"task coverage mismatch: observed={sorted(observed_tasks)}, expected={sorted(expected_tasks)}"
            )
        expected_repeats = int(manifest["handback_repeats"])
        bad_repeat_rows = sum(int(row.get("handback_repeats", -1)) != expected_repeats for row in rows)
        if bad_repeat_rows:
            protocol_reasons.append(f"{bad_repeat_rows} rows differ from frozen K={expected_repeats}")
        paired_required = bool(manifest.get("paired_repeat_seeds_across_boundaries"))
        paired_seed_violations = []
        if paired_required:
            for state, group in sorted(by_state.items()):
                seed_sets = {
                    tuple(int(seed) for seed in row.get("handback_repeat_seeds", []))
                    for row in group
                }
                bad_modes = {
                    str(row.get("handback_repeat_seed_pairing")) for row in group
                } - {"common_across_boundaries"}
                if len(seed_sets) != 1 or bad_modes:
                    paired_seed_violations.append(state)
            if paired_seed_violations:
                protocol_reasons.append(
                    f"{len(paired_seed_violations)} states violate paired repeat seeds"
                )
        else:
            paired_seed_violations = []
    else:
        paired_required = False
        paired_seed_violations = []

    state_summaries = {
        str(state["state_key"]): state
        for suite_report in collection.get("suite_reports", [])
        for state in suite_report.get("state_summaries", [])
    }
    unexplained_missing_boundaries: list[dict[str, Any]] = []
    if manifest:
        frozen_boundaries = {int(value) for value in manifest["boundaries"]}
        for state in sorted(expected_states & observed_states):
            observed_boundaries = {
                int(row["elapsed_oft_steps"]) for row in by_state[state]
            }
            unexpected_boundaries = observed_boundaries - frozen_boundaries
            if unexpected_boundaries:
                unexplained_missing_boundaries.append({
                    "state_key": state,
                    "reason": "unexpected_boundaries",
                    "boundaries": sorted(unexpected_boundaries),
                })
            executed = int(state_summaries.get(state, {}).get("executed_oft_steps", -1))
            for boundary in sorted(frozen_boundaries - observed_boundaries):
                # A boundary can be absent only when the exact persistent OFT
                # trajectory terminated before reaching it.
                # When the episode terminates exactly on the boundary, there is
                # no post-action state at which a handback decision can be made.
                if missing_boundary_is_violation(boundary, executed):
                    unexplained_missing_boundaries.append({
                        "state_key": state,
                        "reason": "missing_reachable_boundary",
                        "boundary": boundary,
                        "executed_oft_steps": executed,
                    })
        if unexplained_missing_boundaries:
            protocol_reasons.append(
                f"{len(unexplained_missing_boundaries)} boundary coverage violations"
            )
    fixed_boundary_rates = {
        boundary: statistics.fmean(
            float(row["handback_success_probability"]) for row in group
        )
        for boundary, group in by_boundary.items()
    }
    best_fixed_boundary = min(
        fixed_boundary_rates,
        key=lambda boundary: (-fixed_boundary_rates[boundary], boundary),
    ) if fixed_boundary_rates else None
    probability_oracle_values: list[float] = []
    probability_oracle_steps: list[int] = []
    nonmonotonic_states: list[str] = []
    downward_transitions: list[dict[str, Any]] = []
    confidence_separated_downward_transitions: list[dict[str, Any]] = []
    state_curves: dict[str, list[dict[str, float | int]]] = {}
    for state, group in sorted(by_state.items()):
        curve_rows = sorted(group, key=lambda row: int(row["elapsed_oft_steps"]))
        curve = [
            (
                int(row["elapsed_oft_steps"]),
                float(row["handback_success_probability"]),
            )
            for row in curve_rows
        ]
        state_curves[state] = [
            {
                "elapsed_oft_steps": int(row["elapsed_oft_steps"]),
                "success_probability": float(row["handback_success_probability"]),
                "wilson_lcb": float(row["handback_success_wilson_lcb_95_one_sided"]),
                "wilson_ucb": float(row["handback_success_wilson_ucb_95_one_sided"]),
            }
            for row in curve_rows
        ]
        best_probability = max(probability for _, probability in curve)
        earliest_best = min(boundary for boundary, probability in curve if probability == best_probability)
        probability_oracle_values.append(best_probability)
        probability_oracle_steps.append(earliest_best)
        for left, right in zip(curve_rows, curve_rows[1:]):
            left_boundary = int(left["elapsed_oft_steps"])
            right_boundary = int(right["elapsed_oft_steps"])
            left_probability = float(left["handback_success_probability"])
            right_probability = float(right["handback_success_probability"])
            if right_probability < left_probability:
                transition = {
                    "state_key": state,
                    "from_boundary": left_boundary,
                    "to_boundary": right_boundary,
                    "probability_drop": left_probability - right_probability,
                    "confidence_separated": bool(
                        float(left["handback_success_wilson_lcb_95_one_sided"])
                        > float(right["handback_success_wilson_ucb_95_one_sided"])
                    ),
                }
                downward_transitions.append(transition)
                if transition["confidence_separated"]:
                    confidence_separated_downward_transitions.append(transition)
        if any(row["state_key"] == state for row in downward_transitions):
            nonmonotonic_states.append(state)
    probability_oracle_rate = statistics.fmean(probability_oracle_values) if probability_oracle_values else None
    best_fixed_rate = fixed_boundary_rates.get(best_fixed_boundary) if best_fixed_boundary is not None else None
    live_finite = int(collection.get("live_finite_safe_states", 0))
    live_tasks = int(collection.get("live_finite_safe_task_count", 0))
    live_savings = float(collection.get("live_oracle_oft_step_savings_fraction", 0.0))
    live_bins = {
        str(key): int(value)
        for key, value in collection.get("live_minimum_successful_boundary_counts", {}).items()
        if int(key) > 0 and int(value) >= 3
    }
    opportunity_reasons = []
    if live_finite < args.opportunity_min_finite_states:
        opportunity_reasons.append(
            f"only {live_finite} live finite-safe states (<{args.opportunity_min_finite_states})"
        )
    if live_tasks < 3:
        opportunity_reasons.append(f"only {live_tasks} tasks have live finite-safe states (<3)")
    if live_savings < args.opportunity_min_savings:
        opportunity_reasons.append(
            f"live conservative oracle savings {live_savings:.4f} (<{args.opportunity_min_savings:.4f})"
        )
    if len(live_bins) < 2:
        opportunity_reasons.append(f"only {len(live_bins)} populated finite stopping bins (<2)")
    report = {
        "schema_version": "rase-pre-c0-r5-probabilistic-boundary-summary/v1",
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": sha256(args.dataset),
        "source_collection_report": str(args.collection_report.resolve()),
        "source_collection_report_sha256": sha256(args.collection_report),
        "source_manifest": str(args.manifest.resolve()) if args.manifest else None,
        "source_manifest_sha256": sha256(args.manifest) if args.manifest else None,
        "n_rows": len(rows),
        "n_valid_rows": len(rows) - len(malformed),
        "n_states": n_states,
        "n_tasks": len({str(row["task_id"]) for row in rows}),
        "n_trials": sum(int(row.get("handback_repeats", 0)) for row in rows),
        "persistent_replay_matches": persistent_matches,
        "persistent_replay_total": persistent_total,
        "repeat_field_completeness": 1.0 - len(malformed) / max(1, len(rows)),
        "duplicate_seed_rows": duplicate_seeds,
        "manifest_state_coverage": {
            "expected": len(expected_states) if manifest else None,
            "observed": len(observed_states),
            "missing": missing_manifest_states,
            "unexpected": unexpected_states,
        },
        "unexplained_missing_boundaries": unexplained_missing_boundaries,
        "paired_repeat_seed_protocol": {
            "required": paired_required,
            "violating_states": paired_seed_violations,
            "status": "ready" if not paired_seed_violations else "not_ready",
        },
        "nondegenerate_boundaries": sum(0.0 < p < 1.0 for p in probabilities),
        "nondegenerate_boundary_fraction": sum(0.0 < p < 1.0 for p in probabilities) / max(1, len(probabilities)),
        "all_success_boundaries": sum(p == 1.0 for p in probabilities),
        "all_failure_boundaries": sum(p == 0.0 for p in probabilities),
        "success_probability": describe(probabilities),
        "bernoulli_entropy_bits": describe(entropies),
        "wilson_lcb": describe(lcbs),
        "boundaries_with_lcb_at_least": {
            str(threshold): sum(lcb >= threshold for lcb in lcbs)
            for threshold in (0.5, 0.8, 0.9, 0.95)
        },
        "successful_continuation_steps": describe(success_steps),
        "failed_continuation_steps": describe(failure_steps),
        "by_suite": {suite: group_summary(group) for suite, group in sorted(by_suite.items())},
        "by_boundary": {str(boundary): group_summary(group) for boundary, group in sorted(by_boundary.items())},
        "exploratory_probability_opportunity": {
            "fixed_boundary_success_rates": {
                str(boundary): rate for boundary, rate in sorted(fixed_boundary_rates.items())
            },
            "best_fixed_boundary": best_fixed_boundary,
            "best_fixed_empirical_success_rate": best_fixed_rate,
            "probability_oracle_empirical_success_rate": probability_oracle_rate,
            "probability_oracle_minus_best_fixed": (
                probability_oracle_rate - best_fixed_rate
                if probability_oracle_rate is not None and best_fixed_rate is not None else None
            ),
            "probability_oracle_total_oft_steps": sum(probability_oracle_steps),
            "nonmonotonic_states": nonmonotonic_states,
            "nonmonotonic_state_fraction": len(nonmonotonic_states) / max(1, len(by_state)),
            "downward_transitions": downward_transitions,
            "large_downward_transitions_at_least_0_4": sum(
                row["probability_drop"] >= 0.4 for row in downward_transitions
            ),
            "confidence_separated_downward_transitions": confidence_separated_downward_transitions,
            "state_curves": state_curves,
            "scope": "descriptive K-repeat smoke only; no inferential claim",
        },
        "malformed_rows": malformed,
        "protocol_gate_status": "ready" if not protocol_reasons else "not_ready",
        "protocol_gate_reasons": protocol_reasons,
        "probability_opportunity_gate_status": (
            "ready" if not opportunity_reasons else "not_ready"
        ),
        "probability_opportunity_gate_reasons": opportunity_reasons,
        "probability_opportunity_gate": {
            "live_finite_safe_states": live_finite,
            "live_finite_safe_task_count": live_tasks,
            "live_conservative_oracle_savings": live_savings,
            "populated_finite_stopping_bins": live_bins,
            "minimum_finite_states": args.opportunity_min_finite_states,
            "minimum_savings": args.opportunity_min_savings,
        },
        "scientific_scope": "label-entropy pilot only; repeats are not independent states",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["protocol_gate_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

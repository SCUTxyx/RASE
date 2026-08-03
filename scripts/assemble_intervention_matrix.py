#!/usr/bin/env python3
"""Assemble strict Smol and direct OFT outcomes into one intervention matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize_oft_rpc_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize timed OFT predict RPCs without conflating them with env rollout time."""
    measured = []
    for row in rows:
        result = row.get("result") or {}
        if (
            "oracle_predict_calls" not in result
            or "oracle_predict_elapsed_s" not in result
        ):
            continue
        measured.append(result)
    calls = sum(int(row["oracle_predict_calls"]) for row in measured)
    elapsed_s = sum(float(row["oracle_predict_elapsed_s"]) for row in measured)
    return {
        "measurement_scope": (
            "OFT action-prediction RPC transfer plus server inference; excludes "
            "environment stepping and client rollout control."
        ),
        "n_states": len(rows),
        "n_measured_states": len(measured),
        "coverage": len(measured) / len(rows) if rows else None,
        "predict_calls": calls,
        "predict_elapsed_s": elapsed_s,
        "mean_predict_calls_per_measured_state": (
            calls / len(measured) if measured else None
        ),
        "mean_ms_per_predict_call": 1000 * elapsed_s / calls if calls else None,
    }


def summarize_success_matrix(
    snapshots: list[Any], outcomes: list[Any], operator_ids: list[str]
) -> dict[str, Any]:
    snapshot_by_id = {row.snapshot_id: row for row in snapshots}
    if len(snapshot_by_id) != len(snapshots):
        raise ValueError("duplicate intervention snapshots")
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    observed_outcomes: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.snapshot_id not in snapshot_by_id:
            raise ValueError(f"outcome references unknown snapshot {outcome.snapshot_id}")
        if outcome.operator_id not in operator_ids:
            raise ValueError(f"unexpected operator {outcome.operator_id}")
        if outcome.observed and not outcome.proxy:
            observed_outcomes[(outcome.snapshot_id, outcome.operator_id)].append(outcome)
            values[(outcome.snapshot_id, outcome.operator_id)].append(
                float(bool(outcome.success))
            )
    per_state = []
    unique_winners: Counter[str] = Counter()
    unique_winner_tasks: dict[str, set[str]] = defaultdict(set)
    oracle_values = []
    lexicographic_oracle_steps = []
    lexicographic_winners: Counter[str] = Counter()
    success_pattern_counts: Counter[str] = Counter()
    n_no_operator_support = 0
    n_all_operator_success = 0
    complete_ids = []
    oracle_supported_ids = []
    for snapshot_id, snapshot in snapshot_by_id.items():
        row_values = {}
        for operator_id in operator_ids:
            arm = values[(snapshot_id, operator_id)]
            if arm:
                row_values[operator_id] = float(np.mean(arm))
        complete = len(row_values) == len(operator_ids)
        winners = []
        step_winners = []
        row_costs = {
            operator_id: {
                metric: float(
                    np.mean(
                        [
                            float(getattr(outcome.costs, metric))
                            for outcome in observed_outcomes[(snapshot_id, operator_id)]
                        ]
                    )
                )
                for metric in ("env_steps", "compute_seconds", "latency_seconds")
            }
            for operator_id in row_values
        }
        if complete:
            complete_ids.append(snapshot_id)
            best = max(row_values.values())
            winners = [
                operator_id
                for operator_id in operator_ids
                if np.isclose(row_values[operator_id], best)
            ]
            oracle_values.append(best)
            success_pattern_counts[
                "".join(
                    "1" if row_values[operator_id] > 0.5 else "0"
                    for operator_id in operator_ids
                )
            ] += 1
            n_no_operator_support += int(np.isclose(best, 0.0))
            n_all_operator_success += int(
                all(np.isclose(row_values[operator_id], 1.0) for operator_id in operator_ids)
            )
            if not np.isclose(best, 0.0):
                oracle_supported_ids.append(snapshot_id)
                best_steps = min(
                    row_costs[operator_id]["env_steps"] for operator_id in winners
                )
                step_winners = [
                    operator_id
                    for operator_id in winners
                    if np.isclose(row_costs[operator_id]["env_steps"], best_steps)
                ]
                lexicographic_oracle_steps.append(best_steps)
                for operator_id in step_winners:
                    lexicographic_winners[operator_id] += 1 / len(step_winners)
            if len(winners) == 1:
                unique_winners[winners[0]] += 1
                unique_winner_tasks[winners[0]].add(snapshot.task_id)
        per_state.append(
            {
                "snapshot_id": snapshot_id,
                "task_id": snapshot.task_id,
                "step": snapshot.step,
                "operator_success": row_values,
                "operator_costs": row_costs,
                "complete": complete,
                "winners": winners,
                "unique_winner": winners[0] if len(winners) == 1 else None,
                "success_then_env_steps_winners": step_winners,
            }
        )
    fixed_values = {
        operator_id: float(
            np.mean(
                [
                    float(np.mean(values[(snapshot_id, operator_id)]))
                    for snapshot_id in complete_ids
                ]
            )
        )
        for operator_id in operator_ids
    } if complete_ids else {}
    best_fixed_id = max(fixed_values, key=fixed_values.get) if fixed_values else None
    oracle = float(np.mean(oracle_values)) if oracle_values else None
    best_fixed = fixed_values.get(best_fixed_id) if best_fixed_id else None
    per_operator_costs = {
        operator_id: {
            metric: float(
                np.mean(
                    [
                        float(
                            np.mean(
                                [
                                    float(getattr(outcome.costs, metric))
                                    for outcome in observed_outcomes[
                                        (snapshot_id, operator_id)
                                    ]
                                ]
                            )
                        )
                        for snapshot_id in complete_ids
                    ]
                )
            )
            for metric in ("env_steps", "compute_seconds", "latency_seconds")
        }
        for operator_id in operator_ids
    } if complete_ids else {}
    success_tied_fixed_ids = (
        [
            operator_id
            for operator_id, value in fixed_values.items()
            if np.isclose(value, best_fixed)
        ]
        if best_fixed is not None
        else []
    )
    supported_mean_env_steps = {
        operator_id: float(
            np.mean(
                [
                    float(
                        np.mean(
                            [
                                outcome.costs.env_steps
                                for outcome in observed_outcomes[
                                    (snapshot_id, operator_id)
                                ]
                            ]
                        )
                    )
                    for snapshot_id in oracle_supported_ids
                ]
            )
        )
        for operator_id in operator_ids
    } if oracle_supported_ids else {}
    success_then_steps_fixed_id = (
        min(
            success_tied_fixed_ids,
            key=lambda operator_id: supported_mean_env_steps[operator_id],
        )
        if success_tied_fixed_ids and oracle_supported_ids
        else None
    )
    success_then_steps_fixed_mean = (
        supported_mean_env_steps[success_then_steps_fixed_id]
        if success_then_steps_fixed_id is not None
        else None
    )
    lexicographic_oracle_mean = (
        float(np.mean(lexicographic_oracle_steps))
        if lexicographic_oracle_steps
        else None
    )
    reference_id = "continue_smol_active_chunk"
    pairwise_vs_continue = {}
    if reference_id in operator_ids:
        for operator_id in operator_ids:
            if operator_id == reference_id:
                continue
            higher = lower = tied = 0
            for snapshot_id in complete_ids:
                reference = float(np.mean(values[(snapshot_id, reference_id)]))
                candidate = float(np.mean(values[(snapshot_id, operator_id)]))
                higher += int(candidate > reference and not np.isclose(candidate, reference))
                lower += int(candidate < reference and not np.isclose(candidate, reference))
                tied += int(np.isclose(candidate, reference))
            pairwise_vs_continue[operator_id] = {
                "higher_success_states": higher,
                "lower_success_states": lower,
                "tied_success_states": tied,
            }
    return {
        "n_snapshots": len(snapshots),
        "n_episodes": len({snapshot.episode_id for snapshot in snapshots}),
        "n_tasks": len({snapshot.task_id for snapshot in snapshots}),
        "n_complete_snapshots": len(complete_ids),
        "per_operator_success_rate": fixed_values,
        "best_fixed_operator": best_fixed_id,
        "best_fixed_success_rate": best_fixed,
        "same_state_oracle_success_rate": oracle,
        "oracle_minus_best_fixed": (
            oracle - best_fixed
            if oracle is not None and best_fixed is not None
            else None
        ),
        "n_no_operator_support": n_no_operator_support,
        "no_operator_support_rate": (
            n_no_operator_support / len(complete_ids) if complete_ids else None
        ),
        "n_all_operator_success": n_all_operator_success,
        "all_operator_success_rate": (
            n_all_operator_success / len(complete_ids) if complete_ids else None
        ),
        "success_pattern_counts": dict(sorted(success_pattern_counts.items())),
        "pairwise_vs_continue": pairwise_vs_continue,
        "per_operator_mean_costs": per_operator_costs,
        "success_then_env_steps": {
            "n_oracle_supported_states": len(oracle_supported_ids),
            "best_fixed_operator": success_then_steps_fixed_id,
            "best_fixed_mean_env_steps_on_supported_states": (
                success_then_steps_fixed_mean
            ),
            "same_state_oracle_mean_env_steps_on_supported_states": (
                lexicographic_oracle_mean
            ),
            "oracle_steps_saved_vs_best_fixed_on_supported_states": (
                success_then_steps_fixed_mean - lexicographic_oracle_mean
                if success_then_steps_fixed_mean is not None
                and lexicographic_oracle_mean is not None
                else None
            ),
            "winner_mass_on_supported_states": dict(
                sorted(lexicographic_winners.items())
            ),
            "semantics": (
                "On states where at least one arm succeeds, lexicographically maximize "
                "terminal success, then minimize env_steps."
            ),
        },
        "unique_winner_counts": dict(sorted(unique_winners.items())),
        "unique_winner_task_counts": {
            operator_id: len(tasks)
            for operator_id, tasks in sorted(unique_winner_tasks.items())
        },
        "per_state": per_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smol-run", type=Path, required=True)
    parser.add_argument("--oft-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()

    from rase.interventions.dataset import parse_registry, registry_payload
    from rase.interventions.schema import (
        CostVector,
        Feasibility,
        InterventionOutcome,
        InterventionSnapshot,
        OperatorFamily,
        OperatorSpec,
    )

    smol_run = args.smol_run.resolve()
    output_dir = args.output_dir.resolve()
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh run requires a new output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_specs = parse_registry(_read_json(smol_run / "operators.json"))
    source_spec_by_id = {spec.operator_id: spec for spec in source_specs}
    operator_ids = ["continue_smol_active_chunk", "replan_smol", "switch_oft"]
    missing_specs = set(operator_ids[:2]) - set(source_spec_by_id)
    if missing_specs:
        raise ValueError(f"Smol registry missing operators: {sorted(missing_specs)}")
    specs = [source_spec_by_id[name] for name in operator_ids[:2]]
    specs.append(
        OperatorSpec(
            operator_id="switch_oft",
            family=OperatorFamily.SWITCH_POLICY,
            executor="openvla_oft",
            recovery_target="current_observation",
            parameters={"handoff": "public_observation", "policy_reset": True},
            requires=("public_observation",),
        )
    )
    snapshots = [
        InterventionSnapshot.from_dict(row)
        for row in _read_jsonl(smol_run / "snapshots.jsonl")
    ]
    snapshot_ids = {snapshot.snapshot_id for snapshot in snapshots}
    outcomes = [
        InterventionOutcome.from_dict(row)
        for row in _read_jsonl(smol_run / "outcomes.jsonl")
    ]
    outcomes = [row for row in outcomes if row.operator_id in operator_ids[:2]]

    oft_by_state = {}
    source_summaries = []
    for summary_path in args.oft_summary:
        summary = _read_json(summary_path.resolve())
        if summary.get("status") != "complete":
            raise ValueError(f"OFT summary is not complete: {summary_path}")
        source_summaries.append(str(summary_path.resolve()))
        for row in summary.get("per_state") or []:
            state_key = str(row["state_key"])
            if state_key in oft_by_state:
                raise ValueError(f"duplicate OFT state across summaries: {state_key}")
            oft_by_state[state_key] = dict(row)
    if set(oft_by_state) != snapshot_ids:
        raise ValueError(
            "OFT coverage differs from Smol snapshots: "
            f"missing={sorted(snapshot_ids - set(oft_by_state))} "
            f"extra={sorted(set(oft_by_state) - snapshot_ids)}"
        )
    for snapshot in snapshots:
        row = oft_by_state[snapshot.snapshot_id]
        result = dict(row["result"])
        outcomes.append(
            InterventionOutcome(
                snapshot_id=snapshot.snapshot_id,
                operator_id="switch_oft",
                continuation_seed=0,
                feasibility=Feasibility(feasible=True),
                observed=True,
                success=bool(row["direct_oft_success"]),
                operator_completed=True,
                stop_reason=str(result["stop_reason"]),
                utility_cost=0.0,
                cost_source="phase0_zero_utility_cost",
                costs=CostVector(
                    compute_seconds=float(result["elapsed_s"]),
                    latency_seconds=float(result["elapsed_s"]),
                    env_steps=int(result["env_steps"]),
                ),
                outcome_semantics="direct_oft_from_same_snapshot",
            )
        )

    summary = summarize_success_matrix(snapshots, outcomes, operator_ids)
    oft_rows = [oft_by_state[snapshot_id] for snapshot_id in sorted(oft_by_state)]
    summary.update(
        {
            "schema_version": "rase-intervention-matrix-summary/v1",
            "status": "complete",
            "source_smol_run": str(smol_run),
            "source_oft_summaries": source_summaries,
            "diagnostic_only": True,
            "interpretation": (
                f"Diagnostic same-state intervention matrix over "
                f"{summary['n_complete_snapshots']} complete states, "
                f"{summary['n_episodes']} source episodes, and "
                f"{summary['n_tasks']} tasks. Evidentiary strength additionally "
                "depends on preregistration and held-out confirmation."
            ),
            "oft_rpc_inference": summarize_oft_rpc_metrics(oft_rows),
        }
    )
    _write_json(output_dir / "operators.json", registry_payload(specs))
    _write_jsonl(output_dir / "snapshots.jsonl", [row.to_dict() for row in snapshots])
    _write_jsonl(output_dir / "outcomes.jsonl", [row.to_dict() for row in outcomes])
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "n_complete_snapshots": summary["n_complete_snapshots"],
                "per_operator_success_rate": summary["per_operator_success_rate"],
                "oracle_minus_best_fixed": summary["oracle_minus_best_fixed"],
                "unique_winner_counts": summary["unique_winner_counts"],
                "output": str(output_dir / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit same-source regeneration headroom and cross-policy complementarity."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _cluster_bootstrap_difference(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(
            float(bool(row[left])) - float(bool(row[right]))
        )
    tasks = sorted(grouped)
    if not tasks:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        values = [value for task in sampled for value in grouped[str(task)]]
        estimates[index] = float(np.mean(values))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n_states": n,
        "n_tasks": len({row["task_id"] for row in rows}),
        "continue_rate": sum(row["continue_success"] for row in rows) / n,
        "first_resample_rate": sum(row["first_resample_success"] for row in rows) / n,
        "resample_oracle_rate": sum(row["resample_oracle_success"] for row in rows) / n,
        "fallback_rate": sum(row["fallback_success"] for row in rows) / n,
        "full_oracle_rate": sum(row["full_oracle_success"] for row in rows) / n,
        "mixed_outcome_roots": sum(row["mixed_resample_outcomes"] for row in rows),
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(
    keys: dict[str, Any],
    resample: dict[str, Any],
    source: dict[str, Any],
    fallback_summaries: list[dict[str, Any]],
    candidate_generation: dict[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 2_026_082_004,
) -> dict[str, Any]:
    if keys.get("artifact_version") != "rase-regeneration-state-keys/v1":
        raise ValueError("unexpected regeneration key artifact")
    if keys.get("selection_uses_outcomes") is not False:
        raise ValueError("regeneration cohort must be outcome-independent")
    ordered_keys = [str(value) for value in keys.get("state_keys") or []]
    records = {str(row["state_key"]): dict(row) for row in keys.get("records") or []}
    if not ordered_keys or len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("state_keys must be non-empty and unique")
    if set(records) != set(ordered_keys):
        raise ValueError("key records do not match state_keys")

    provenance = dict(resample.get("state_keys_provenance") or {})
    if provenance.get("selected_state_keys_sha256") != keys.get("state_keys_sha256"):
        raise ValueError("resample summary checksum does not match frozen cohort")
    if resample.get("continuation_seed_mode") != "common_root_rollout":
        raise ValueError("formal regeneration screen requires common_root_rollout")
    resample_rows = {
        str(row["state_key"]): dict(row) for row in resample.get("per_state") or []
    }
    if set(resample_rows) != set(ordered_keys):
        raise ValueError("resample summary state set does not match frozen cohort")

    source_rows: dict[str, dict[str, Any]] = {}
    for row in source.get("per_pair") or []:
        key = str(row["state_key"])
        if key in source_rows:
            raise ValueError(f"duplicate source result for {key}")
        source_rows[key] = dict(row)
    if not set(ordered_keys).issubset(source_rows):
        raise ValueError("source summary is missing frozen states")

    fallback_rows: dict[str, dict[str, Any]] = {}
    for summary in fallback_summaries:
        for row in summary.get("per_state") or []:
            key = str(row["state_key"])
            if key in fallback_rows:
                raise ValueError(f"duplicate fallback result for {key}")
            fallback_rows[key] = dict(row)
    if not set(ordered_keys).issubset(fallback_rows):
        raise ValueError("fallback summaries are missing frozen states")

    generation_keys = [str(value) for value in candidate_generation.get("state_keys") or []]
    if generation_keys != ordered_keys:
        raise ValueError("candidate generation state order does not match frozen cohort")
    diversity = dict(candidate_generation.get("diversity") or {})
    mean_chunk_l2 = float(diversity.get("mean_chunk_l2", 0.0))
    diversity_rows = list(diversity.get("per_state") or [])
    if len(diversity_rows) != len(ordered_keys):
        raise ValueError("candidate diversity rows must align with frozen cohort order")

    rows: list[dict[str, Any]] = []
    expected_k: int | None = None
    for state_index, state_key in enumerate(ordered_keys):
        candidates = list(resample_rows[state_key].get("candidates") or [])
        if expected_k is None:
            expected_k = len(candidates)
        if expected_k is None or expected_k < 2 or len(candidates) != expected_k:
            raise ValueError("all roots must have the same K >= 2")
        outcomes: list[bool] = []
        for candidate in candidates:
            if int(candidate.get("trials", -1)) != 1:
                raise ValueError("eligibility screen requires one trial per candidate")
            outcomes.append(int(candidate.get("successes", 0)) == 1)
        meta = records[state_key]
        continue_success = bool(
            source_rows[state_key].get("continue_smol_active_chunk")
        )
        fallback_success = bool(fallback_rows[state_key].get("direct_oft_success"))
        first = outcomes[0]
        resample_oracle = any(outcomes)
        source_or_resample = continue_success or resample_oracle
        rows.append({
            "state_key": state_key,
            "task_id": str(meta["task_id"]),
            "episode_id": str(meta["episode_id"]),
            "suite": str(meta["suite"]),
            "step": int(meta["step"]),
            "resample_successes": outcomes,
            "continue_success": continue_success,
            "first_resample_success": first,
            "resample_oracle_success": resample_oracle,
            "source_or_resample_success": source_or_resample,
            "fallback_success": fallback_success,
            "full_oracle_success": source_or_resample or fallback_success,
            "mixed_resample_outcomes": len(set(outcomes)) > 1,
            "resample_rescues_continue": (not continue_success) and resample_oracle,
            "resample_only_vs_fallback": resample_oracle and not fallback_success,
            "fallback_only_vs_resample": fallback_success and not resample_oracle,
            "mean_pairwise_chunk_l2": float(
                diversity_rows[state_index]["mean_pairwise_chunk_l2"]
            ),
        })

    n = len(rows)
    rates = {
        field: sum(bool(row[field]) for row in rows) / n
        for field in (
            "continue_success",
            "first_resample_success",
            "resample_oracle_success",
            "source_or_resample_success",
            "fallback_success",
            "full_oracle_success",
        )
    }
    fixed_fields = ("continue_success", "first_resample_success", "fallback_success")
    best_fixed_field = max(fixed_fields, key=lambda field: rates[field])
    best_fixed_rate = rates[best_fixed_field]
    mixed = sum(row["mixed_resample_outcomes"] for row in rows)
    rescued = sum(row["resample_rescues_continue"] for row in rows)
    rescue_tasks = len({row["task_id"] for row in rows if row["resample_rescues_continue"]})
    resample_only = sum(row["resample_only_vs_fallback"] for row in rows)
    fallback_only = sum(row["fallback_only_vs_resample"] for row in rows)
    oracle_minus_first = rates["resample_oracle_success"] - rates["first_resample_success"]
    union_minus_continue = rates["source_or_resample_success"] - rates["continue_success"]
    full_minus_best = rates["full_oracle_success"] - best_fixed_rate
    chunk_l2 = [float(row["mean_pairwise_chunk_l2"]) for row in rows]
    mixed_flags = [float(row["mixed_resample_outcomes"]) for row in rows]
    rescue_flags = [float(row["resample_rescues_continue"]) for row in rows]
    gain_flags = [
        float(row["resample_oracle_success"] and not row["first_resample_success"])
        for row in rows
    ]
    success_counts = [float(sum(row["resample_successes"])) for row in rows]
    mixed_l2 = [
        row["mean_pairwise_chunk_l2"] for row in rows
        if row["mixed_resample_outcomes"]
    ]
    uniform_l2 = [
        row["mean_pairwise_chunk_l2"] for row in rows
        if not row["mixed_resample_outcomes"]
    ]

    r0_conditions = {
        "candidate_diversity_mean_chunk_l2_ge_0_05": mean_chunk_l2 >= 0.05,
        "mixed_outcome_root_rate_ge_0_10": mixed / n >= 0.10,
        "oracle_at_k_minus_first_ge_0_05": oracle_minus_first >= 0.05,
        "continue_union_resample_minus_continue_ge_0_05": union_minus_continue >= 0.05,
        "resample_rescues_span_at_least_2_tasks": rescue_tasks >= 2,
    }
    r0_pass = all(r0_conditions.values())
    r0x_conditions = {
        "r0_candidate_learnability_pass": r0_pass,
        "resample_has_fallback_unique_success": resample_only >= 1,
        "full_oracle_minus_best_fixed_ge_0_05": full_minus_best >= 0.05,
    }
    r0x_pass = all(r0x_conditions.values())
    if r0x_pass:
        status = "pass_cross_policy_regeneration_eligibility"
    elif r0_pass:
        status = "pass_regeneration_only_cross_policy_gate_closed"
    else:
        status = "fail_regeneration_eligibility"

    by_suite: dict[str, Any] = {}
    for suite in sorted({str(row["suite"]) for row in rows}):
        by_suite[suite] = _group([row for row in rows if row["suite"] == suite])

    return {
        "artifact_version": "rase-regeneration-opportunity/v1",
        "status": status,
        "n_states": n,
        "n_tasks": len({row["task_id"] for row in rows}),
        "k": expected_k,
        "verifier_training_gate": "open" if r0_pass else "closed",
        "cross_policy_claim_gate": "open" if r0x_pass else "closed",
        "closed_loop_gate": (
            "closed_pending_offline_verifier"
            if r0_pass else "closed_candidate_eligibility_failed"
        ),
        "metrics": {
            "candidate_mean_chunk_l2": mean_chunk_l2,
            "continue_success_rate": rates["continue_success"],
            "first_resample_success_rate": rates["first_resample_success"],
            "resample_oracle_success_rate": rates["resample_oracle_success"],
            "source_or_resample_oracle_rate": rates["source_or_resample_success"],
            "fallback_success_rate": rates["fallback_success"],
            "full_oracle_success_rate": rates["full_oracle_success"],
            "mixed_outcome_roots": mixed,
            "mixed_outcome_root_rate": mixed / n,
            "resample_rescues_continue": rescued,
            "resample_rescue_tasks": rescue_tasks,
            "oracle_at_k_minus_first": oracle_minus_first,
            "oracle_at_k_minus_first_task_bootstrap_95_ci": _cluster_bootstrap_difference(
                rows, "resample_oracle_success", "first_resample_success",
                replicates=bootstrap_replicates, seed=bootstrap_seed,
            ),
            "continue_union_resample_minus_continue": union_minus_continue,
            "continue_union_resample_minus_continue_task_bootstrap_95_ci": _cluster_bootstrap_difference(
                rows, "source_or_resample_success", "continue_success",
                replicates=bootstrap_replicates, seed=bootstrap_seed + 1,
            ),
            "best_fixed_arm": best_fixed_field,
            "best_fixed_rate": best_fixed_rate,
            "full_oracle_minus_best_fixed": full_minus_best,
            "full_oracle_minus_best_fixed_task_bootstrap_95_ci": _cluster_bootstrap_difference(
                rows, "full_oracle_success", best_fixed_field,
                replicates=bootstrap_replicates, seed=bootstrap_seed + 2,
            ),
            "diversity_diagnostic": {
                "mean_chunk_l2_mixed_roots": float(np.mean(mixed_l2)) if mixed_l2 else None,
                "mean_chunk_l2_uniform_roots": float(np.mean(uniform_l2)) if uniform_l2 else None,
                "chunk_l2_vs_mixed_outcome_pearson": _correlation(chunk_l2, mixed_flags),
                "chunk_l2_vs_continue_rescue_pearson": _correlation(chunk_l2, rescue_flags),
                "chunk_l2_vs_oracle_gain_pearson": _correlation(chunk_l2, gain_flags),
                "chunk_l2_vs_candidate_success_count_pearson": _correlation(chunk_l2, success_counts),
                "interpretation": (
                    "Descriptive BOKBO-style diagnostic only; correlations are not "
                    "causal and the cohort is small."
                ),
            },
        },
        "portfolio": {
            "resample_only_vs_fallback": resample_only,
            "fallback_only_vs_resample": fallback_only,
            "both": sum(row["resample_oracle_success"] and row["fallback_success"] for row in rows),
            "neither": sum(not row["resample_oracle_success"] and not row["fallback_success"] for row in rows),
        },
        "r0_conditions": r0_conditions,
        "r0x_conditions": r0x_conditions,
        "by_suite": by_suite,
        "per_state": rows,
        "provenance": {
            "state_keys_sha256": keys.get("state_keys_sha256"),
            "continuation_seed_mode": resample.get("continuation_seed_mode"),
            "bootstrap_unit": "task",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _markdown(result: dict[str, Any]) -> str:
    m = result["metrics"]
    p = result["portfolio"]
    if result["verifier_training_gate"] == "open":
        final_note = (
            "The closed-loop gate remains closed until a task-held-out verifier "
            "beats action-shuffled and label-permuted controls."
        )
    else:
        final_note = (
            "The closed-loop and verifier-training gates remain closed: improve "
            "candidate generation, then repeat this frozen eligibility protocol."
        )
    return "\n".join([
        "# RASE same-source regeneration eligibility",
        "",
        f"- Status: `{result['status']}`",
        f"- Cohort: {result['n_states']} states / {result['n_tasks']} tasks; K={result['k']}",
        f"- Continue: {_pct(m['continue_success_rate'])}; first resample: {_pct(m['first_resample_success_rate'])}",
        f"- Oracle@K resample: {_pct(m['resample_oracle_success_rate'])}; gain over first {_pct(m['oracle_at_k_minus_first'])}",
        f"- Continue ∪ resample: {_pct(m['source_or_resample_oracle_rate'])}; gain over continue {_pct(m['continue_union_resample_minus_continue'])}",
        f"- Mixed roots: {m['mixed_outcome_roots']}/{result['n_states']} ({_pct(m['mixed_outcome_root_rate'])})",
        f"- Resample rescues: {m['resample_rescues_continue']} across {m['resample_rescue_tasks']} tasks",
        f"- Fallback: {_pct(m['fallback_success_rate'])}; full oracle: {_pct(m['full_oracle_success_rate'])}",
        f"- Full oracle gain over best fixed ({m['best_fixed_arm']}): {_pct(m['full_oracle_minus_best_fixed'])}",
        f"- Resample-only / fallback-only / both / neither: {p['resample_only_vs_fallback']} / {p['fallback_only_vs_resample']} / {p['both']} / {p['neither']}",
        (
            "- Diversity diagnostic: mean chunk L2 mixed/uniform = "
            f"{m['diversity_diagnostic']['mean_chunk_l2_mixed_roots']:.3f}/"
            f"{m['diversity_diagnostic']['mean_chunk_l2_uniform_roots']:.3f}; "
            "corr(L2, oracle-gain) = "
            f"{m['diversity_diagnostic']['chunk_l2_vs_oracle_gain_pearson']:.3f}"
        ),
        f"- Verifier training gate: `{result['verifier_training_gate']}`",
        f"- Cross-policy claim gate: `{result['cross_policy_claim_gate']}`",
        "",
        "## R0 candidate-learnability conditions",
        "",
        *[f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["r0_conditions"].items()],
        "",
        "## R0X cross-policy conditions",
        "",
        *[f"- [{'x' if passed else ' '}] `{name}`" for name, passed in result["r0x_conditions"].items()],
        "",
        final_note,
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--resample-summary", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--fallback-summary", type=Path, action="append", required=True)
    parser.add_argument("--candidate-generation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_082_004)
    args = parser.parse_args()
    result = analyze(
        _read(args.keys), _read(args.resample_summary), _read(args.source_summary),
        [_read(path) for path in args.fallback_summary],
        _read(args.candidate_generation_summary),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "verifier_training_gate": result["verifier_training_gate"],
        "cross_policy_claim_gate": result["cross_policy_claim_gate"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

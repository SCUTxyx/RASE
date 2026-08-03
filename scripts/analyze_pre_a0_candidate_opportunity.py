#!/usr/bin/env python3
"""Audit the PRE-A0 strict-resample and heterogeneous-fallback opportunity."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else 0.0


def _bootstrap_difference(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if not rows:
        return [0.0, 0.0]
    values = np.asarray(
        [float(bool(row[left])) - float(bool(row[right])) for row in rows]
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_states": len(rows),
        "first_sample_successes": sum(row["first_sample_success"] for row in rows),
        "strict_oracle_successes": sum(row["strict_oracle_success"] for row in rows),
        "oft_fallback_successes": sum(row["oft_fallback_success"] for row in rows),
        "heterogeneous_oracle_successes": sum(
            row["heterogeneous_oracle_success"] for row in rows
        ),
    }


def analyze(
    keys: dict[str, Any],
    strict: dict[str, Any],
    fallback: dict[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 2_026_080_302,
) -> dict[str, Any]:
    if keys.get("artifact_version") != "rase-pre-a0-state-keys/v1":
        raise ValueError("unexpected PRE-A0 key artifact")
    if keys.get("selection_uses_outcomes") is not False:
        raise ValueError("PRE-A0 key selection must be outcome-independent")
    key_records = {str(row["state_key"]): dict(row) for row in keys["records"]}
    ordered_keys = [str(value) for value in keys["state_keys"]]
    if set(key_records) != set(ordered_keys):
        raise ValueError("key records and state_keys do not match")

    provenance = dict(strict.get("state_keys_provenance") or {})
    if provenance.get("selected_state_keys_sha256") != keys.get("state_keys_sha256"):
        raise ValueError("strict summary does not match frozen state-key checksum")
    strict_rows = {str(row["state_key"]): dict(row) for row in strict["per_state"]}
    fallback_rows = {
        str(row["state_key"]): dict(row) for row in fallback.get("per_task") or []
    }
    if set(strict_rows) != set(ordered_keys):
        raise ValueError("strict summary state set does not match frozen keys")
    if not set(ordered_keys).issubset(fallback_rows):
        raise ValueError("fallback analysis is missing frozen states")

    rows: list[dict[str, Any]] = []
    expected_k: int | None = None
    for state_key in ordered_keys:
        candidates = list(strict_rows[state_key].get("candidates") or [])
        if expected_k is None:
            expected_k = len(candidates)
        if len(candidates) != expected_k or expected_k < 2:
            raise ValueError("all strict states must contain the same K >= 2")
        outcomes = []
        for candidate in candidates:
            if int(candidate.get("trials", -1)) != 1:
                raise ValueError("PRE-A0 screen requires one trial per candidate")
            outcomes.append(int(candidate.get("successes", 0)) == 1)
        meta = key_records[state_key]
        old = fallback_rows[state_key]
        if str(old.get("task_id")) != str(meta.get("task_id")):
            raise ValueError(f"task mismatch for {state_key}")
        first = outcomes[0]
        strict_oracle = any(outcomes)
        oft = bool(old["oft_only_success"])
        rows.append(
            {
                "state_key": state_key,
                "task_id": str(meta["task_id"]),
                "episode_id": str(meta["episode_id"]),
                "suite": str(meta["suite"]),
                "cell": (
                    f"{meta['perturbation_dimension']}:L{meta['perturbation_level']}"
                ),
                "strict_candidate_successes": outcomes,
                "first_sample_success": first,
                "strict_oracle_success": strict_oracle,
                "strict_rescue": (not first) and strict_oracle,
                "oft_fallback_success": oft,
                "heterogeneous_oracle_success": strict_oracle or oft,
                "heterogeneous_rescue": (not first) and (strict_oracle or oft),
                "source_only_success": bool(old["source_only_success"]),
                "source_to_oft_success": bool(old["source_to_oft_success"]),
            }
        )

    n = len(rows)
    first_n = sum(row["first_sample_success"] for row in rows)
    strict_n = sum(row["strict_oracle_success"] for row in rows)
    fallback_n = sum(row["oft_fallback_success"] for row in rows)
    heterogeneous_n = sum(row["heterogeneous_oracle_success"] for row in rows)
    base_failures = n - first_n
    strict_rescues = sum(row["strict_rescue"] for row in rows)
    strict_mixed_outcome_states = sum(
        len(set(row["strict_candidate_successes"])) > 1 for row in rows
    )
    heterogeneous_rescues = sum(row["heterogeneous_rescue"] for row in rows)
    strict_only = sum(
        row["strict_oracle_success"] and not row["oft_fallback_success"] for row in rows
    )
    fallback_only = sum(
        row["oft_fallback_success"] and not row["strict_oracle_success"] for row in rows
    )
    both = sum(
        row["oft_fallback_success"] and row["strict_oracle_success"] for row in rows
    )
    neither = n - strict_only - fallback_only - both
    rescue_tasks = len({row["task_id"] for row in rows if row["heterogeneous_rescue"]})

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("suite", "cell"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        grouped[field] = {
            name: _group_summary(group_rows) for name, group_rows in sorted(groups.items())
        }

    strict_headroom = (strict_n - first_n) / n
    heterogeneous_headroom = (heterogeneous_n - first_n) / n
    rescued_failure_fraction = (
        heterogeneous_rescues / base_failures if base_failures else 0.0
    )
    conditions = {
        "heterogeneous_oracle_headroom_ge_0_08": heterogeneous_headroom >= 0.08,
        "base_failures_rescued_ge_0_20": rescued_failure_fraction >= 0.20,
        "unique_successes_span_at_least_2_tasks": rescue_tasks >= 2,
        "two_generator_families_have_unique_success": strict_only > 0
        and fallback_only > 0,
        "heldout_direction_confirmed": False,
    }
    pilot_pass = all(
        value
        for key, value in conditions.items()
        if key != "heldout_direction_confirmed"
    )
    status = "pilot_signal_requires_scaled_heldout" if pilot_pass else "not_ready"
    reasons = [key for key, passed in conditions.items() if not passed]
    return {
        "artifact_version": "rase-pre-a0-candidate-opportunity/v1",
        "status": status,
        "decision_scope": "development-only opportunity screen; no training claim",
        "world_model_gate": "closed",
        "critic_training_gate": "closed",
        "n_states": n,
        "n_tasks": len({row["task_id"] for row in rows}),
        "k_strict_resamples": expected_k,
        "metrics": {
            "first_sample_successes": first_n,
            "first_sample_success_rate": first_n / n,
            "strict_oracle_successes": strict_n,
            "strict_oracle_success_rate": strict_n / n,
            "strict_oracle_headroom": strict_headroom,
            "strict_oracle_headroom_bootstrap_95_ci": _bootstrap_difference(
                rows,
                "strict_oracle_success",
                "first_sample_success",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            "strict_rescues": strict_rescues,
            "strict_mixed_outcome_states": strict_mixed_outcome_states,
            "oft_fallback_successes": fallback_n,
            "oft_fallback_success_rate": fallback_n / n,
            "heterogeneous_oracle_successes": heterogeneous_n,
            "heterogeneous_oracle_success_rate": heterogeneous_n / n,
            "heterogeneous_oracle_headroom": heterogeneous_headroom,
            "heterogeneous_oracle_headroom_bootstrap_95_ci": _bootstrap_difference(
                rows,
                "heterogeneous_oracle_success",
                "first_sample_success",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1,
            ),
            "base_failures": base_failures,
            "heterogeneous_rescues": heterogeneous_rescues,
            "base_failure_rescue_fraction": rescued_failure_fraction,
            "rescue_unique_tasks": rescue_tasks,
        },
        "portfolio": {
            "strict_only": strict_only,
            "fallback_only": fallback_only,
            "both": both,
            "neither": neither,
            "strict_vs_fallback_mcnemar_exact_p": _mcnemar_exact(
                strict_only, fallback_only
            ),
        },
        "gate_conditions": conditions,
        "reasons": reasons,
        "by_group": grouped,
        "per_state": rows,
        "provenance": {
            "state_keys_sha256": keys["state_keys_sha256"],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "limitations": [
            "The first strict sample is a preregistered sampling baseline, "
            "not the live active suffix.",
            "The OFT fallback is a full direct-policy rollout and is not "
            "compute-matched to a 10-step candidate prefix.",
            "Fresh replan, local corrections, abstention, and state-changing "
            "recovery are not yet implemented.",
            "These 12 development states were used by earlier Phase 1A work "
            "and are excluded from the flagship hidden test.",
            "One rollout per strict candidate measures opportunity, not "
            "calibrated success probability.",
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    metric = result["metrics"]
    portfolio = result["portfolio"]
    passed = sum(result["gate_conditions"].values())
    n = result["n_states"]
    return "\n".join(
        [
            "# PRE-A0 candidate opportunity audit",
            "",
            f"- Status: `{result['status']}`",
            (
                f"- Strict K: {result['k_strict_resamples']}; states/tasks: "
                f"{n}/{result['n_tasks']}"
            ),
            (
                f"- First sample: {metric['first_sample_successes']}/{n} "
                f"({metric['first_sample_success_rate']:.1%})"
            ),
            (
                f"- Strict oracle@K: {metric['strict_oracle_successes']}/{n} "
                f"({metric['strict_oracle_success_rate']:.1%}); headroom "
                f"{metric['strict_oracle_headroom']:+.1%}"
            ),
            (
                f"- OFT fallback: {metric['oft_fallback_successes']}/{n} "
                f"({metric['oft_fallback_success_rate']:.1%})"
            ),
            (
                "- Heterogeneous oracle: "
                f"{metric['heterogeneous_oracle_successes']}/{n} "
                f"({metric['heterogeneous_oracle_success_rate']:.1%}); headroom "
                f"{metric['heterogeneous_oracle_headroom']:+.1%}"
            ),
            (
                f"- Base-failure rescue: {metric['heterogeneous_rescues']}/"
                f"{metric['base_failures']} "
                f"({metric['base_failure_rescue_fraction']:.1%}) across "
                f"{metric['rescue_unique_tasks']} tasks"
            ),
            (
                "- Portfolio strict-only/fallback-only/both/neither: "
                f"{portfolio['strict_only']}/{portfolio['fallback_only']}/"
                f"{portfolio['both']}/{portfolio['neither']}"
            ),
            (
                f"- Pilot conditions passed: {passed}/"
                f"{len(result['gate_conditions'])}; held-out confirmation "
                "is deliberately false."
            ),
            "- World-model and critic-training gates remain closed.",
            "",
            "## Failed or pending conditions",
            "",
            *[f"- `{reason}`" for reason in result["reasons"]],
            "",
            "## Interpretation boundary",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--strict-summary", type=Path, required=True)
    parser.add_argument("--fallback-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_080_302)
    args = parser.parse_args()
    result = analyze(
        _read(args.keys),
        _read(args.strict_summary),
        _read(args.fallback_analysis),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

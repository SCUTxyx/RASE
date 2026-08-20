"""Scientific discovery diagnostics that do not modify the frozen gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from rase.vnext.opportunity import audit_opportunity, utility


ABORT_OPERATOR = "abort.safe"


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _operator_summary(rows: list[dict[str, Any]], weights: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("available") is True:
            grouped[str(row["operator_id"])].append(row)
    return {
        operator: {
            "trials": len(values),
            "success": _mean([float(row["success"]) for row in values]),
            "utility": _mean([utility(row, weights) for row in values]),
            "harm": _mean([float(row["harm"]) for row in values]),
            "query_cost": _mean([float(row["query_cost"]) for row in values]),
            "fallback_cost": _mean([float(row["fallback_cost"]) for row in values]),
            "latency_cost": _mean([float(row["latency_cost"]) for row in values]),
        }
        for operator, values in sorted(grouped.items())
    }


def _nondegenerate_cells(rows: list[dict[str, Any]], *, excluded: set[str]) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        operator = str(row["operator_id"])
        if row.get("available") is not True or operator in excluded:
            continue
        key = (str(row["root_id"]), str(row["policy_id"]), str(row["decision_point_id"]))
        cells[key][operator].append(float(row["success"]))
    eligible = 0
    nondegenerate = 0
    by_policy: dict[str, list[bool]] = defaultdict(list)
    by_suite: dict[str, list[bool]] = defaultdict(list)
    lookup = {
        (str(row["root_id"]), str(row["policy_id"]), str(row["decision_point_id"])): str(row["suite"])
        for row in rows
    }
    for key, by_operator in cells.items():
        if len(by_operator) < 2:
            continue
        eligible += 1
        means = [_mean(values) for values in by_operator.values()]
        flag = max(means) - min(means) > 0
        nondegenerate += int(flag)
        by_policy[key[1]].append(flag)
        by_suite[lookup[key]].append(flag)
    return {
        "eligible_cells": eligible,
        "nondegenerate_cells": nondegenerate,
        "fraction": nondegenerate / eligible if eligible else 0.0,
        "by_policy": {
            key: {"cells": len(values), "fraction": _mean([float(value) for value in values])}
            for key, values in sorted(by_policy.items())
        },
        "by_suite": {
            key: {"cells": len(values), "fraction": _mean([float(value) for value in values])}
            for key, values in sorted(by_suite.items())
        },
    }


def _repeat_stability(rows: list[dict[str, Any]], *, repeats: int) -> dict[str, Any]:
    branches: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("available") is True:
            key = (
                str(row["root_id"]), str(row["policy_id"]),
                str(row["decision_point_id"]), str(row["operator_id"]),
            )
            branches[key].append(row)
    invalid = []
    stable = 0
    by_operator: dict[str, list[bool]] = defaultdict(list)
    for key, values in branches.items():
        replicas = sorted(int(row["exact_repeat_replica"]) for row in values)
        if replicas != list(range(repeats)):
            invalid.append({"key": list(key), "replicas": replicas})
            continue
        outcomes = {bool(row["success"]) for row in values}
        flag = len(outcomes) == 1
        stable += int(flag)
        by_operator[key[3]].append(flag)
    valid = len(branches) - len(invalid)
    return {
        "expected_repeats": repeats,
        "valid_branches": valid,
        "invalid_branches": invalid,
        "stable_branches": stable,
        "stable_fraction": stable / valid if valid else 0.0,
        "by_operator": {
            key: {"branches": len(values), "stable_fraction": _mean([float(value) for value in values])}
            for key, values in sorted(by_operator.items())
        },
    }


def _candidate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("available") is True and row.get("operator_id") == "resample.source":
            grouped[str(row["policy_id"])].append(row)
    result = {}
    for policy, values in sorted(grouped.items()):
        pair_distinct = 0
        selected_second = 0
        differs_from_requery = 0
        comparable_requery = 0
        requery_lookup = {
            (
                str(row["root_id"]), str(row["policy_id"]), str(row["decision_point_id"]),
                int(row["exact_repeat_replica"]),
            ): row.get("requery_first_action_sha256")
            for row in rows if row.get("operator_id") == "requery.source"
        }
        for row in values:
            hashes = [str(item["first_action_sha256"]) for item in row.get("candidates", [])]
            pair_distinct += int(len(hashes) >= 2 and len(set(hashes)) > 1)
            selected_second += int(row.get("selected_candidate_id") == "candidate.1")
            key = (
                str(row["root_id"]), str(row["policy_id"]), str(row["decision_point_id"]),
                int(row["exact_repeat_replica"]),
            )
            requery_hash = requery_lookup.get(key)
            selected = next(
                (item for item in row.get("candidates", [])
                 if item.get("candidate_id") == row.get("selected_candidate_id")), None,
            )
            if requery_hash is not None and selected is not None:
                comparable_requery += 1
                differs_from_requery += int(str(selected["first_action_sha256"]) != str(requery_hash))
        total = len(values)
        result[policy] = {
            "trials": total,
            "distinct_candidate_pairs": pair_distinct,
            "distinct_candidate_pair_fraction": pair_distinct / total if total else 0.0,
            "selected_candidate_1": selected_second,
            "selected_candidate_1_fraction": selected_second / total if total else 0.0,
            "selected_differs_from_requery": differs_from_requery,
            "selected_differs_from_requery_fraction": (
                differs_from_requery / comparable_requery if comparable_requery else 0.0
            ),
        }
    return result


def audit_discovery_shadow(
    rows: list[dict[str, Any]], *, repeats: int, weights: dict[str, Any],
    opportunity_gate: dict[str, Any], minimum_nondegenerate_fraction: float,
    bootstrap_samples: int = 10_000, bootstrap_seed: int = 202708,
) -> dict[str, Any]:
    """Audit meaningful non-abort structure without changing preregistered feasibility."""
    all_available = [row for row in rows if row.get("available") is True]
    non_abort = [row for row in all_available if row.get("operator_id") != ABORT_OPERATOR]
    frozen_like = audit_opportunity(
        all_available, repeats=repeats, weights=weights, gate=opportunity_gate,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    non_abort_opportunity = audit_opportunity(
        non_abort, repeats=repeats, weights=weights, gate=opportunity_gate,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    all_nondegenerate = _nondegenerate_cells(all_available, excluded=set())
    non_abort_nondegenerate = _nondegenerate_cells(all_available, excluded={ABORT_OPERATOR})
    scientific_structure = (
        non_abort_nondegenerate["fraction"] >= minimum_nondegenerate_fraction
        and len(non_abort_opportunity["qualifying_winner_operators"]) >= 2
    )
    return {
        "schema_version": "rase-vnext-discovery-shadow-audit/v1",
        "status": "GO_CONFIRMATION" if scientific_structure else "STOP_REVISE_OPERATOR",
        "role": "scientific_shadow_only_does_not_modify_frozen_feasibility",
        "available_rows": len(all_available),
        "operator_summary": _operator_summary(all_available, weights),
        "all_operator_nondegeneracy": all_nondegenerate,
        "non_abort_nondegeneracy": non_abort_nondegenerate,
        "repeat_stability": _repeat_stability(all_available, repeats=repeats),
        "candidate_diagnostics": _candidate_diagnostics(all_available),
        "exploratory_opportunity_all_operators": frozen_like,
        "exploratory_opportunity_non_abort": non_abort_opportunity,
        "checks": {
            "non_abort_nondegenerate": (
                non_abort_nondegenerate["fraction"] >= minimum_nondegenerate_fraction
            ),
            "non_abort_root_winner_diversity": (
                len(non_abort_opportunity["qualifying_winner_operators"]) >= 2
            ),
        },
        "interpretation_lock": {
            "confirmation_gate_has_not_passed": True,
            "learned_models_remain_locked": True,
            "rl_and_opd_remain_locked": True,
        },
    }


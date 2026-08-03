"""Paired state-level analysis for direct policy escalation experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rase.collect.adaptive import wilson_interval
from rase.collect.policy_matrix import exact_mcnemar_p


PAIR_LABELS = ("both_success", "portfolio_only", "direct_only", "both_fail")


def _coverage(hits: int, trials: int) -> dict[str, Any]:
    lower, upper = wilson_interval(hits, trials) if trials else (None, None)
    return {
        "hits": hits,
        "trials": trials,
        "rate": hits / trials if trials else None,
        "wilson_95": {"lower": lower, "upper": upper, "unit": "state"},
    }


def _pair_label(portfolio: bool, direct: bool) -> str:
    if portfolio and direct:
        return "both_success"
    if portfolio:
        return "portfolio_only"
    if direct:
        return "direct_only"
    return "both_fail"


def aggregate_direct_escalation_pairing(
    matrix: Mapping[str, Any],
    direct_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare direct OFT with matched Smol and prefix-portfolio outcomes.

    The W7 portfolio is diagnostic because it takes an any-of-K maximum.  The
    direct arm is deployable.  This function preserves that distinction while
    measuring overlap, discordance, and suite concentration at the state level.
    """
    if matrix.get("schema_version") != "rase-one-shot-policy-matrix/v1":
        raise ValueError("unsupported policy matrix schema")
    if matrix.get("status") != "complete":
        raise ValueError("policy matrix is incomplete")

    matrix_rows: dict[str, Mapping[str, Any]] = {}
    for row in matrix.get("per_state") or ():
        key = str(row["state_key"])
        if key in matrix_rows:
            raise ValueError(f"duplicate matrix state: {key}")
        matrix_rows[key] = row

    direct_rows: dict[str, Mapping[str, Any]] = {}
    for summary in direct_summaries:
        if summary.get("schema_version") != "rase-oft-direct-escalation/v1":
            raise ValueError("unsupported direct escalation summary schema")
        if summary.get("status") != "complete":
            raise ValueError("direct escalation summary is incomplete")
        for row in summary.get("per_state") or ():
            key = str(row["state_key"])
            if key in direct_rows:
                raise ValueError(f"duplicate direct state: {key}")
            direct_rows[key] = row

    if not matrix_rows:
        raise ValueError("policy matrix has no states")
    if set(matrix_rows) != set(direct_rows):
        missing = sorted(set(matrix_rows) - set(direct_rows))
        extra = sorted(set(direct_rows) - set(matrix_rows))
        raise ValueError(f"matrix/direct state union mismatch: missing={missing}, extra={extra}")

    pair_counts: Counter[str] = Counter()
    suite_counts: dict[str, Counter[str]] = {}
    smol_hits = portfolio_hits = direct_hits = 0
    per_state: list[dict[str, Any]] = []
    for key, row in matrix_rows.items():
        smol = bool(row["smol_portfolio_hit"])
        portfolio = bool(row["oft_portfolio_hit"])
        direct = bool(direct_rows[key]["direct_oft_success"])
        label = _pair_label(portfolio, direct)
        pair_counts[label] += 1
        smol_hits += int(smol)
        portfolio_hits += int(portfolio)
        direct_hits += int(direct)
        suite = str(row.get("suite") or direct_rows[key].get("suite") or "unknown")
        suite_counts.setdefault(suite, Counter())[label] += 1
        per_state.append(
            {
                "state_key": key,
                "suite": suite,
                "dim": row.get("dim"),
                "level": row.get("level"),
                "episode_id": row.get("episode_id"),
                "smol_portfolio_success": smol,
                "prefix_portfolio_success": portfolio,
                "direct_oft_success": direct,
                "prefix_direct_pair_label": label,
            }
        )

    n_states = len(per_state)
    counts = {label: pair_counts[label] for label in PAIR_LABELS}
    direct_vs_smol = {
        "smol_only": sum(
            int(row["smol_portfolio_success"] and not row["direct_oft_success"])
            for row in per_state
        ),
        "direct_only": sum(
            int(row["direct_oft_success"] and not row["smol_portfolio_success"])
            for row in per_state
        ),
    }
    union_hits = n_states - counts["both_fail"]
    return {
        "schema_version": "rase-direct-escalation-pairing/v1",
        "status": "complete",
        "n_states": n_states,
        "smol_portfolio": _coverage(smol_hits, n_states),
        "prefix_oft_portfolio": _coverage(portfolio_hits, n_states),
        "direct_oft": _coverage(direct_hits, n_states),
        "prefix_direct_pair_counts": counts,
        "prefix_direct_mcnemar_exact_p_two_sided": exact_mcnemar_p(
            counts["portfolio_only"], counts["direct_only"]
        ),
        "direct_vs_smol_pair_counts": direct_vs_smol,
        "direct_vs_smol_mcnemar_exact_p_two_sided": exact_mcnemar_p(
            direct_vs_smol["smol_only"], direct_vs_smol["direct_only"]
        ),
        "direct_minus_prefix_risk_difference": (
            direct_hits - portfolio_hits
        ) / n_states,
        "prefix_direct_union": _coverage(union_hits, n_states),
        "prefix_direct_intersection_over_union": (
            counts["both_success"] / union_hits if union_hits else None
        ),
        "per_suite": [
            {
                "suite": suite,
                "n_states": sum(values.values()),
                **{label: values[label] for label in PAIR_LABELS},
                "portfolio_hits": values["both_success"] + values["portfolio_only"],
                "direct_hits": values["both_success"] + values["direct_only"],
            }
            for suite, values in sorted(suite_counts.items())
        ],
        "per_state": per_state,
        "interpretation": {
            "both_success": "direct escalation is sufficient; the prefix is not required",
            "portfolio_only": "prefix may help, but any-of-K selection prevents a deployable causal claim",
            "direct_only": "direct escalation avoids a harmful prefix or reflects one-shot variation",
            "both_fail": "neither tested OFT route recovers the state",
        },
        "warnings": [
            "The cohort is conditioned on SmolVLA episode failure.",
            "The W7 prefix result is an oracle any-of-K portfolio, not one deployable action.",
            "McNemar inference uses states, not candidate rollouts, as paired units.",
            "A marginal 9/24 versus 8/24 difference is not interpretable without the overlap table.",
        ],
    }

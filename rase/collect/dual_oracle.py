"""Testable aggregation for Wilson-triaged SmolVLA and one-shot OFT results."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rase.collect.adaptive import wilson_interval

SMOL_RECOVERABLE_LABELS = frozenset({"A", "B"})
SMOL_FAILED_LABELS = frozenset({"C"})
SMOL_UNCERTAIN_LABELS = frozenset({"uncertain", "incomplete"})
CROSS_LABELS = (
    "consensus_recoverable",
    "smol_only",
    "oft_only",
    "both_fail",
    "uncertain",
)


def _indexed_states(
    payload: Mapping[str, Any], *, source: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("per_state") or []:
        key = str(row["state_key"])
        if key in indexed:
            raise ValueError(f"duplicate state_key {key!r} in {source}")
        indexed[key] = row
    return indexed


def _counts(state: Mapping[str, Any] | None) -> tuple[int, int]:
    if state is None:
        return 0, 0
    return (
        sum(int(c.get("successes", 0)) for c in state.get("candidates") or []),
        sum(int(c.get("trials", 0)) for c in state.get("candidates") or []),
    )


def _smol_status(state: Mapping[str, Any] | None) -> bool | None:
    if state is None:
        return None
    label = state.get("set_label")
    if label in SMOL_RECOVERABLE_LABELS:
        return True
    if label in SMOL_FAILED_LABELS:
        return False
    return None


def _oft_status(state: Mapping[str, Any] | None) -> bool | None:
    if state is None:
        return None
    candidates = state.get("candidates") or []
    if not candidates or any(int(c.get("trials", 0)) <= 0 for c in candidates):
        return None
    return any(int(c.get("successes", 0)) > 0 for c in candidates)


def _cross_label(smol: bool | None, oft: bool | None) -> str:
    if smol is None or oft is None:
        return "uncertain"
    if smol and oft:
        return "consensus_recoverable"
    if smol:
        return "smol_only"
    if oft:
        return "oft_only"
    return "both_fail"


def _exact_mcnemar_p(smol_only: int, oft_only: int) -> float | None:
    """Two-sided exact McNemar p-value for discordant state-level pairs."""
    discordant = smol_only + oft_only
    if discordant == 0:
        return None
    tail = min(smol_only, oft_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def _oracle_agreement(split_keys: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    consensus = len(split_keys["consensus_recoverable"])
    smol_only = len(split_keys["smol_only"])
    oft_only = len(split_keys["oft_only"])
    both_fail = len(split_keys["both_fail"])
    n = consensus + smol_only + oft_only + both_fail
    if n == 0:
        return {
            "n_evaluable": 0,
            "confusion": {
                "both_recoverable": 0,
                "smolvla_only": 0,
                "oft_only": 0,
                "both_unrecoverable": 0,
            },
            "agreement": None,
            "agreement_wilson_95": {"lower": None, "upper": None, "unit": "state"},
            "cohen_kappa": None,
            "mcnemar_exact_p_two_sided": None,
        }
    agreements = consensus + both_fail
    agreement = agreements / n
    agreement_ci = wilson_interval(agreements, n)
    smol_positive = (consensus + smol_only) / n
    oft_positive = (consensus + oft_only) / n
    expected = smol_positive * oft_positive + (1 - smol_positive) * (1 - oft_positive)
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else None
    return {
        "n_evaluable": n,
        "confusion": {
            "both_recoverable": consensus,
            "smolvla_only": smol_only,
            "oft_only": oft_only,
            "both_unrecoverable": both_fail,
        },
        "agreement": agreement,
        "agreement_wilson_95": {
            "lower": agreement_ci[0],
            "upper": agreement_ci[1],
            "unit": "state",
        },
        "cohen_kappa": kappa,
        "mcnemar_exact_p_two_sided": _exact_mcnemar_p(smol_only, oft_only),
    }


def aggregate_dual_oracle(
    smolvla_summary: Mapping[str, Any],
    oft_summaries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    pool_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate the two oracles without assigning Wilson labels to OFT.

    OFT candidates are treated as deterministic one-shot observations. Its
    Wilson interval is computed only across independent state-level portfolio
    recovery indicators, never across candidate rows.
    """
    smol_by_key = _indexed_states(smolvla_summary, source="SmolVLA summary")
    smol_threshold = float(
        (smolvla_summary.get("protocol") or {}).get("threshold", 0.5)
    )
    oft_by_key: dict[str, Mapping[str, Any]] = {}
    oft_suite: dict[str, str] = {}
    suite_raw: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    for suite, payload in oft_summaries:
        states = _indexed_states(payload, source=f"OFT summary {suite!r}")
        hits = trials = 0
        for key, state in states.items():
            if key in oft_by_key:
                raise ValueError(f"duplicate OFT state_key {key!r} across suites")
            oft_by_key[key] = state
            oft_suite[key] = suite
            successes, n_trials = _counts(state)
            hits += successes
            trials += n_trials
        suite_raw[suite] = {
            "successes": hits,
            "trials": trials,
            "n_states": len(states),
            "rollouts_this_process": int(payload.get("rollouts_this_process") or 0),
        }

    all_keys = sorted(set(smol_by_key) | set(oft_by_key))
    per_state: list[dict[str, Any]] = []
    merged_candidates: list[dict[str, Any]] = []
    split_keys: dict[str, list[str]] = {label: [] for label in CROSS_LABELS}
    candidate_hits = candidate_trials = 0
    portfolio_recovered = portfolio_evaluable = 0

    for key in all_keys:
        smol = smol_by_key.get(key)
        oft = oft_by_key.get(key)
        smol_successes, smol_trials = _counts(smol)
        oft_successes, oft_trials = _counts(oft)
        smol_recoverable = _smol_status(smol)
        oft_recoverable = _oft_status(oft)
        label = _cross_label(smol_recoverable, oft_recoverable)
        split_keys[label].append(key)

        if oft is not None:
            candidates = oft.get("candidates") or []
            candidate_hits += sum(
                int(candidate.get("successes", 0)) > 0 for candidate in candidates
            )
            candidate_trials += sum(int(candidate.get("trials", 0)) for candidate in candidates)
            if any(
                int(candidate.get("trials", 0)) != 1
                or int(candidate.get("successes", 0)) not in {0, 1}
                for candidate in candidates
            ):
                warnings.append(
                    f"OFT state {key!r} is not strictly one trial with a binary "
                    "outcome per candidate"
                )
        if oft_recoverable is not None:
            portfolio_evaluable += 1
            portfolio_recovered += int(oft_recoverable)

        metadata = dict((pool_meta or {}).get(key) or {})
        suite = metadata.get("suite", oft_suite.get(key))
        if suite is None:
            warnings.append(f"state {key!r} has no suite metadata")
        t0 = metadata.get("t0", metadata.get("step", metadata.get("timestep")))
        row = {
            "state_key": key,
            "suite": suite,
            "dim": metadata.get("dim", metadata.get("perturb_dim")),
            "perturb_dim": metadata.get("perturb_dim", metadata.get("dim")),
            "level": metadata.get("level"),
            "t0": t0,
            "set_label_smolvla": (smol or {}).get("set_label"),
            "smolvla_successes": smol_successes,
            "smolvla_trials": smol_trials,
            "oft_successes": oft_successes,
            "oft_trials": oft_trials,
            "oft_candidate_hits": (
                sum(
                    int(candidate.get("successes", 0)) > 0
                    for candidate in (oft or {}).get("candidates") or []
                )
                if oft is not None
                else None
            ),
            "recoverable_smolvla": smol_recoverable,
            "recoverable_oft": oft_recoverable,
            "cross_label": label,
            "dual_track_label": label,
            "divergent_oft_only": label == "oft_only",
        }
        per_state.append(row)

        if smol is not None and oft is not None:
            smol_candidates = smol.get("candidates") or []
            oft_candidates = oft.get("candidates") or []
            if len(smol_candidates) != len(oft_candidates):
                warnings.append(f"candidate count mismatch for state {key!r}")
            for idx, (smol_cand, oft_cand) in enumerate(
                zip(smol_candidates, oft_candidates)
            ):
                merged_candidates.append(
                    {
                        "state_key": key,
                        "candidate_id": idx,
                        "successes_smolvla": int(smol_cand.get("successes", 0)),
                        "trials_smolvla": int(smol_cand.get("trials", 0)),
                        "successes_oft": int(oft_cand.get("successes", 0)),
                        "trials_oft": int(oft_cand.get("trials", 0)),
                        "recoverable_smolvla": (
                            float(smol_cand["lower"]) > smol_threshold
                            if smol_cand.get("lower") is not None
                            else int(smol_cand.get("successes", 0)) > 0
                        ),
                        "recoverable_oft": int(oft_cand.get("successes", 0)) > 0,
                    }
                )

    smol_recovered = sum(row["recoverable_smolvla"] is True for row in per_state)
    smol_evaluable = sum(row["recoverable_smolvla"] is not None for row in per_state)
    oft_only = len(split_keys["oft_only"])
    coverage = (
        portfolio_recovered / portfolio_evaluable if portfolio_evaluable else None
    )
    coverage_ci = (
        wilson_interval(portfolio_recovered, portfolio_evaluable)
        if portfolio_evaluable
        else (None, None)
    )
    hit_rate = candidate_hits / candidate_trials if candidate_trials else None
    hit_distribution = Counter(
        row["oft_candidate_hits"]
        for row in per_state
        if row["oft_candidate_hits"] is not None
    )
    warnings.extend(
        [
            "OFT uses deterministic one-shot verification; candidate_hit_rate is "
            "descriptive and must not be interpreted as a per-candidate success probability.",
            "OFT portfolio outcomes are not SmolVLA Wilson Set A/B certification.",
            "Candidate outcomes within a state may be non-independent; the portfolio "
            "Wilson interval uses states, not candidates, as trials.",
            "Cross-oracle agreement is descriptive because SmolVLA Wilson triage and "
            "OFT deterministic one-shot portfolios have different measurement semantics.",
        ]
    )

    n_states = len(all_keys)
    return {
        "schema_version": "rase-dual-oracle-summary/v2",
        "n_states": n_states,
        "deterministic_candidate_hits": candidate_hits,
        "deterministic_candidate_trials": candidate_trials,
        "candidate_hit_rate": hit_rate,
        "portfolio_recovered_states": portfolio_recovered,
        "portfolio_evaluable_states": portfolio_evaluable,
        "portfolio_coverage": coverage,
        "portfolio_coverage_wilson_95": {
            "lower": coverage_ci[0],
            "upper": coverage_ci[1],
            "unit": "state",
        },
        "cross_label_counts": {label: len(split_keys[label]) for label in CROSS_LABELS},
        "cross_oracle_agreement": _oracle_agreement(split_keys),
        "splits": split_keys,
        "candidate_success_count_distribution": {
            str(key): value for key, value in sorted(hit_distribution.items())
        },
        "metric_definitions": {
            "smolvla_set_label": "Preserved Wilson A/B/C/uncertain triage label.",
            "deterministic_candidate_hits": "OFT candidates with a successful one-shot outcome.",
            "candidate_hit_rate": "OFT deterministic hits divided by executed candidate trials.",
            "portfolio_recovered_states": "States with at least one successful OFT candidate.",
            "portfolio_coverage": "Recovered OFT states divided by OFT-evaluable states.",
            "cross_label": "Cross of SmolVLA Wilson recoverability and OFT portfolio recovery.",
        },
        "oracle_semantics": {
            "smolvla": "sequential_wilson_triage",
            "oft": "deterministic_one_shot",
            "oft_trials_per_candidate": 1,
        },
        "warnings": sorted(set(warnings)),
        "per_state": per_state,
        "per_candidate_gt": merged_candidates,
        "smolvla_raw": {
            "successes": sum(row["smolvla_successes"] for row in per_state),
            "trials": sum(row["smolvla_trials"] for row in per_state),
            "label_counts": smolvla_summary.get("label_counts"),
            "rollouts_this_process": smolvla_summary.get("rollouts_this_process"),
        },
        "oft_raw_by_suite": suite_raw,
        # Legacy headline aliases retained for readers of the v1 summary.
        "Y_Smol": smol_recovered / smol_evaluable if smol_evaluable else 0.0,
        "Y_OFT": coverage if coverage is not None else 0.0,
        "C_div": oft_only / n_states if n_states else 0.0,
        "n_recoverable_smolvla": smol_recovered,
        "n_recoverable_oft": portfolio_recovered,
        "n_divergent_oft_only": oft_only,
    }

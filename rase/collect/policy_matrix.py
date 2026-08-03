"""Paired aggregation for deterministic one-shot policy-matrix screens."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rase.collect.adaptive import wilson_interval


def _index_states(payload: Mapping[str, Any], *, source: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("per_state") or ():
        key = str(row["state_key"])
        if key in indexed:
            raise ValueError(f"duplicate state_key {key!r} in {source}")
        indexed[key] = row
    return indexed


def _one_shot_bits(row: Mapping[str, Any], *, source: str, k: int) -> list[bool]:
    candidates = list(row.get("candidates") or ())
    if len(candidates) != k:
        raise ValueError(f"{source} expected {k} candidates, got {len(candidates)}")
    bits: list[bool] = []
    for candidate, record in enumerate(candidates):
        trials = int(record.get("trials", 0))
        successes = int(record.get("successes", 0))
        if trials != 1 or successes not in {0, 1}:
            raise ValueError(
                f"{source} candidate {candidate} is not one binary trial: "
                f"successes={successes}, trials={trials}"
            )
        bits.append(bool(successes))
    return bits


def _pair_label(left: bool, right: bool, *, left_name: str, right_name: str) -> str:
    if left and right:
        return "both_hit"
    if left:
        return f"{left_name}_only"
    if right:
        return f"{right_name}_only"
    return "both_miss"


def exact_mcnemar_p(left_only: int, right_only: int) -> float | None:
    """Two-sided exact McNemar p-value for independent paired units."""
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def _coverage(successes: int, trials: int) -> dict[str, Any]:
    lower, upper = wilson_interval(successes, trials) if trials else (None, None)
    return {
        "hits": successes,
        "trials": trials,
        "rate": successes / trials if trials else None,
        "wilson_95": {"lower": lower, "upper": upper, "unit": "state"},
    }


def aggregate_one_shot_policy_matrix(
    frozen_keys: Sequence[str],
    smol_summary: Mapping[str, Any],
    oft_summaries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    pool_meta: Mapping[str, Mapping[str, Any]],
    state_keys_sha256: str,
    candidate_artifact_sha256: str,
) -> dict[str, Any]:
    """Validate and aggregate matched Smol/OFT one-shot candidate outcomes.

    State-level portfolio indicators are the inferential unit. Candidate-level
    pairs are retained as descriptive outcomes because candidates within a
    state are not independent.
    """
    frozen = [str(key) for key in frozen_keys]
    if not frozen or len(set(frozen)) != len(frozen):
        raise ValueError("frozen state keys must be non-empty and unique")
    expected = set(frozen)
    if smol_summary.get("mode") != "smolvla-screen":
        raise ValueError("Smol summary must have mode='smolvla-screen'")
    smol_provenance = dict(smol_summary.get("state_keys_provenance") or {})
    if smol_provenance.get("state_keys_sha256") != state_keys_sha256:
        raise ValueError("Smol summary state-key provenance does not match frozen keys")
    smol = _index_states(smol_summary, source="Smol screen")
    if set(smol) != expected:
        raise ValueError("Smol summary state keys do not exactly match frozen keys")

    oft: dict[str, Mapping[str, Any]] = {}
    oft_suite: dict[str, str] = {}
    for suite, payload in oft_summaries:
        if payload.get("mode") != "oft-verify":
            raise ValueError(f"OFT summary {suite!r} must have mode='oft-verify'")
        provenance = dict(payload.get("state_keys_provenance") or {})
        if provenance.get("state_keys_sha256") != state_keys_sha256:
            raise ValueError(f"OFT summary {suite!r} state-key provenance mismatch")
        for key, row in _index_states(payload, source=f"OFT {suite}").items():
            if key in oft:
                raise ValueError(f"duplicate OFT state_key {key!r} across suites")
            oft[key] = row
            oft_suite[key] = suite
    if set(oft) != expected:
        missing = sorted(expected - set(oft))
        extra = sorted(set(oft) - expected)
        raise ValueError(f"OFT summary union mismatch: missing={missing}, extra={extra}")

    k = int((smol_summary.get("protocol") or {}).get("k", 0))
    if k <= 0:
        raise ValueError("Smol summary has invalid candidate count")

    state_pairs: Counter[str] = Counter()
    candidate_pairs: Counter[str] = Counter()
    smol_candidate_hits = oft_candidate_hits = 0
    smol_state_hits = oft_state_hits = 0
    per_state: list[dict[str, Any]] = []
    per_cell_counts: dict[tuple[str, int], Counter[str]] = {}
    per_suite_counts: dict[str, Counter[str]] = {}

    for key in frozen:
        smol_bits = _one_shot_bits(smol[key], source=f"Smol state {key}", k=k)
        oft_bits = _one_shot_bits(oft[key], source=f"OFT state {key}", k=k)
        smol_hit = any(smol_bits)
        oft_hit = any(oft_bits)
        state_label = _pair_label(smol_hit, oft_hit, left_name="smol", right_name="oft")
        state_pairs[state_label] += 1
        smol_state_hits += int(smol_hit)
        oft_state_hits += int(oft_hit)

        candidate_labels = []
        for smol_value, oft_value in zip(smol_bits, oft_bits):
            label = _pair_label(
                smol_value, oft_value, left_name="smol", right_name="oft"
            )
            candidate_pairs[label] += 1
            candidate_labels.append(label)
            smol_candidate_hits += int(smol_value)
            oft_candidate_hits += int(oft_value)

        metadata = dict(pool_meta.get(key) or {})
        dim = str(metadata.get("perturb_dim", metadata.get("dim", "")))
        level = int(metadata.get("level", 0))
        cell = (dim, level)
        per_cell_counts.setdefault(cell, Counter())[state_label] += 1
        suite = str(metadata.get("suite", oft_suite.get(key, "")))
        per_suite_counts.setdefault(suite, Counter())[state_label] += 1
        per_state.append(
            {
                "state_key": key,
                "suite": suite,
                "dim": dim,
                "level": level,
                "episode_id": metadata.get("episode_id"),
                "smol_candidate_hits": sum(smol_bits),
                "oft_candidate_hits": sum(oft_bits),
                "smol_portfolio_hit": smol_hit,
                "oft_portfolio_hit": oft_hit,
                "state_pair_label": state_label,
                "candidate_pair_labels": candidate_labels,
            }
        )

    labels = ("both_hit", "smol_only", "oft_only", "both_miss")
    state_pair_counts = {label: state_pairs[label] for label in labels}
    candidate_pair_counts = {label: candidate_pairs[label] for label in labels}
    discordant = state_pair_counts["smol_only"] + state_pair_counts["oft_only"]
    return {
        "schema_version": "rase-one-shot-policy-matrix/v1",
        "status": "complete",
        "cohort": "failure_challenge",
        "conditioning": "smolvla_episode_outcome=failure",
        "n_states": len(frozen),
        "k": k,
        "state_keys_sha256": state_keys_sha256,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "smol_candidate": {
            "hits": smol_candidate_hits,
            "trials": len(frozen) * k,
            "rate_descriptive": smol_candidate_hits / (len(frozen) * k),
        },
        "oft_candidate": {
            "hits": oft_candidate_hits,
            "trials": len(frozen) * k,
            "rate_descriptive": oft_candidate_hits / (len(frozen) * k),
        },
        "smol_portfolio": _coverage(smol_state_hits, len(frozen)),
        "oft_portfolio": _coverage(oft_state_hits, len(frozen)),
        "state_pair_counts": state_pair_counts,
        "paired_state_effect": {
            "risk_difference_oft_minus_smol": (
                oft_state_hits - smol_state_hits
            ) / len(frozen),
            "discordant_pairs": discordant,
            "oft_win_fraction_among_discordant": (
                state_pair_counts["oft_only"] / discordant if discordant else None
            ),
        },
        "state_mcnemar_exact_p_two_sided": exact_mcnemar_p(
            state_pair_counts["smol_only"], state_pair_counts["oft_only"]
        ),
        "candidate_pair_counts_descriptive": candidate_pair_counts,
        "per_cell_state_pairs": [
            {
                "dim": dim,
                "level": level,
                "n_states": sum(counts.values()),
                **{label: counts[label] for label in labels},
            }
            for (dim, level), counts in sorted(per_cell_counts.items())
        ],
        "per_suite_state_pairs": [
            {
                "suite": suite,
                "n_states": sum(counts.values()),
                **{label: counts[label] for label in labels},
            }
            for suite, counts in sorted(per_suite_counts.items())
        ],
        "per_state": per_state,
        "warnings": [
            "This cohort is conditioned on SmolVLA episode failure and does not "
            "estimate an unconditional NGC rate.",
            "Both policy arms are deterministic one-shot screens; Wilson Set "
            "A/B/C labels do not apply.",
            "Candidate-level rates and pair counts are descriptive because "
            "candidates within a state are dependent.",
            "State-level portfolio indicators are the inferential unit; "
            f"n={len(frozen)} and its uncertainty must be reported.",
        ],
    }

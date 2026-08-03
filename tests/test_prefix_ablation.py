import numpy as np
import pytest

from rase.collect.prefix_ablation import (
    action_prefix_sha256,
    aggregate_prefix_summaries,
    build_decision_suffix_arms,
    build_prefix_arms,
    classify_prefix_ablation,
    summarize_decision_suffix_state,
    summarize_prefix_state,
)
from scripts.export_policy_matrix_split_keys import select_keys


def _outcomes(*, direct=False, zero=False, candidates=(False, False)):
    values = {"direct_oft": direct, "zero_10": zero}
    values.update(
        {f"candidate_{index}": value for index, value in enumerate(candidates)}
    )
    return values


def test_prefix_arms_keep_controls_distinct_from_frozen_candidates():
    candidates = np.arange(3 * 10 * 7, dtype=np.float32).reshape(3, 10, 7)
    arms = build_prefix_arms(candidates)
    assert [arm.label for arm in arms] == [
        "direct_oft",
        "zero_10",
        "candidate_0",
        "candidate_1",
        "candidate_2",
    ]
    assert arms[0].actions.shape == (0, 7)
    assert arms[1].actions.shape == (10, 7)
    assert np.all(arms[1].actions == 0)
    np.testing.assert_array_equal(arms[4].actions, candidates[2])


def test_decision_suffix_arms_preserve_exact_float32_actions_and_hash():
    suffix = np.arange(5 * 7, dtype=np.float64).reshape(5, 7) / 10
    arms = build_decision_suffix_arms(suffix)
    assert [arm.label for arm in arms] == ["direct_oft", "decision_suffix_oft"]
    assert arms[0].actions.shape == (0, 7)
    assert arms[1].actions.dtype == np.float32
    np.testing.assert_array_equal(arms[1].actions, suffix.astype(np.float32))
    assert action_prefix_sha256(arms[1].actions) == action_prefix_sha256(
        suffix.astype(np.float32)
    )


def test_decision_suffix_summary_reports_paired_classification():
    result = summarize_decision_suffix_state(
        "sp1_key",
        [
            {"arm_label": "direct_oft", "success": False},
            {"arm_label": "decision_suffix_oft", "success": True},
        ],
    )
    assert result["classification"] == "deferred_only"
    assert result["decision_suffix_oft_success"] is True


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (
            _outcomes(direct=True, zero=True, candidates=(True, True)),
            "continuation_sufficient_candidate_invariant",
        ),
        (
            _outcomes(direct=True, zero=True, candidates=(True, False)),
            "continuation_sufficient_candidate_harm_possible",
        ),
        (
            _outcomes(direct=False, zero=True, candidates=(False, False)),
            "passive_prefix_sufficient",
        ),
        (
            _outcomes(direct=False, zero=False, candidates=(False, True)),
            "candidate_specific_rescue",
        ),
        (
            _outcomes(direct=False, zero=False, candidates=(False, False)),
            "unrecovered",
        ),
    ],
)
def test_prefix_classification_is_conservative(outcomes, expected):
    assert classify_prefix_ablation(outcomes) == expected


def test_prefix_summary_reports_state_level_mechanism():
    outcomes = _outcomes(direct=False, zero=False, candidates=(True, False))
    records = [
        {"arm_label": label, "success": success}
        for label, success in outcomes.items()
    ]
    result = summarize_prefix_state("sp1_key", records, metadata={"suite": "Goal"})
    assert result["classification"] == "candidate_specific_rescue"
    assert result["candidate_hits"] == 1
    assert result["candidate_trials"] == 2


def test_export_policy_matrix_split_selects_only_requested_label():
    matrix = {
        "per_state": [
            {"state_key": "b", "state_pair_label": "oft_only"},
            {"state_key": "a", "state_pair_label": "oft_only"},
            {"state_key": "c", "state_pair_label": "both_miss"},
        ]
    }
    assert select_keys(matrix, "oft_only") == ["a", "b"]
    with pytest.raises(ValueError, match="no states"):
        select_keys(matrix, "smol_only")


def test_aggregate_prefix_summaries_requires_exact_state_union():
    summary = {
        "schema_version": "rase-oft-prefix-ablation/v1",
        "status": "complete",
        "per_state": [
            {"state_key": "a", "classification": "candidate_specific_rescue"}
        ],
    }
    result = aggregate_prefix_summaries(["a"], [summary])
    assert result["candidate_specific_rescue_states"] == ["a"]
    with pytest.raises(ValueError, match="union mismatch"):
        aggregate_prefix_summaries(["a", "b"], [summary])

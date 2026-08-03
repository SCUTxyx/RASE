import pytest

from rase.collect.escalation_analysis import aggregate_direct_escalation_pairing


def _matrix():
    rows = [
        ("a", "Goal", False, True),
        ("b", "Goal", False, True),
        ("c", "Long", False, False),
        ("d", "Long", False, False),
    ]
    return {
        "schema_version": "rase-one-shot-policy-matrix/v1",
        "status": "complete",
        "per_state": [
            {
                "state_key": key,
                "suite": suite,
                "smol_portfolio_hit": smol,
                "oft_portfolio_hit": oft,
            }
            for key, suite, smol, oft in rows
        ],
    }


def _direct(keys):
    return {
        "schema_version": "rase-oft-direct-escalation/v1",
        "status": "complete",
        "per_state": [
            {"state_key": key, "direct_oft_success": success}
            for key, success in keys
        ],
    }


def test_pairing_reports_overlap_not_only_marginals():
    result = aggregate_direct_escalation_pairing(
        _matrix(), [_direct([("a", True), ("b", False), ("c", True), ("d", False)])]
    )
    assert result["prefix_direct_pair_counts"] == {
        "both_success": 1,
        "portfolio_only": 1,
        "direct_only": 1,
        "both_fail": 1,
    }
    assert result["direct_oft"]["hits"] == 2
    assert result["prefix_oft_portfolio"]["hits"] == 2
    assert result["direct_minus_prefix_risk_difference"] == 0
    assert result["prefix_direct_mcnemar_exact_p_two_sided"] == 1.0
    assert result["direct_vs_smol_mcnemar_exact_p_two_sided"] == 0.5
    assert result["prefix_direct_union"]["hits"] == 3
    assert result["prefix_direct_intersection_over_union"] == pytest.approx(1 / 3)


def test_pairing_rejects_incomplete_state_union():
    with pytest.raises(ValueError, match="union mismatch"):
        aggregate_direct_escalation_pairing(_matrix(), [_direct([("a", True)])])

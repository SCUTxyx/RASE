import pytest

from rase.collect.dual_oracle import aggregate_dual_oracle


def _state(key, label, *, smol_trials=6, oft_hit=None):
    if oft_hit is None:
        return {
            "state_key": key,
            "set_label": label,
            "candidates": [
                {"successes": 0, "trials": smol_trials} for _ in range(8)
            ],
        }
    return {
        "state_key": key,
        "candidates": [
            {"successes": int(oft_hit and index == 0), "trials": 1}
            for index in range(8)
        ],
    }


def test_w4_zero_over_1536_and_17_over_32_semantics():
    smol = {
        "label_counts": {"C": 32},
        "per_state": [_state(f"s{index}", "C") for index in range(32)],
    }
    oft = {
        "per_state": [
            _state(f"s{index}", None, oft_hit=index < 17) for index in range(32)
        ]
    }
    metadata = {
        f"s{index}": {
            "suite": "Spatial",
            "perturb_dim": "camera",
            "level": 3 + index % 3,
            "step": 0,
        }
        for index in range(32)
    }

    result = aggregate_dual_oracle(smol, [("libero_spatial", oft)], pool_meta=metadata)

    assert result["smolvla_raw"]["successes"] == 0
    assert result["smolvla_raw"]["trials"] == 1536
    assert result["deterministic_candidate_hits"] == 17
    assert result["deterministic_candidate_trials"] == 256
    assert result["candidate_hit_rate"] == pytest.approx(17 / 256)
    assert result["portfolio_recovered_states"] == 17
    assert result["portfolio_coverage"] == pytest.approx(17 / 32)
    assert result["portfolio_coverage_wilson_95"]["lower"] == pytest.approx(
        0.3644952, abs=1e-6
    )
    assert result["portfolio_coverage_wilson_95"]["upper"] == pytest.approx(
        0.6913061, abs=1e-6
    )
    assert result["cross_label_counts"]["oft_only"] == 17
    assert result["cross_label_counts"]["both_fail"] == 15
    agreement = result["cross_oracle_agreement"]
    assert agreement["n_evaluable"] == 32
    assert agreement["confusion"] == {
        "both_recoverable": 0,
        "smolvla_only": 0,
        "oft_only": 17,
        "both_unrecoverable": 15,
    }
    assert agreement["agreement"] == pytest.approx(15 / 32)
    assert agreement["cohen_kappa"] == pytest.approx(0.0)
    assert agreement["mcnemar_exact_p_two_sided"] == pytest.approx(2 / 2**17)
    assert result["per_state"][0]["dim"] == "camera"
    assert result["per_state"][0]["t0"] == 0
    assert result["Y_OFT"] == result["portfolio_coverage"]
    assert any("not SmolVLA Wilson" in warning for warning in result["warnings"])


def test_all_cross_labels_and_unknown_are_explicit():
    labels = ["A", "B", "C", "C", "uncertain"]
    hits = [True, False, True, False, True]
    smol = {"per_state": [_state(str(i), label) for i, label in enumerate(labels)]}
    oft = {
        "per_state": [
            _state(str(i), None, oft_hit=hit) for i, hit in enumerate(hits)
        ]
    }

    result = aggregate_dual_oracle(smol, [("suite", oft)])

    assert [row["cross_label"] for row in result["per_state"]] == [
        "consensus_recoverable",
        "smol_only",
        "oft_only",
        "both_fail",
        "uncertain",
    ]
    assert all(result["cross_label_counts"][label] == 1 for label in result["splits"])
    assert result["per_state"][-1]["recoverable_smolvla"] is None
    assert result["cross_oracle_agreement"]["n_evaluable"] == 4


def test_duplicate_keys_are_rejected():
    duplicate = {
        "per_state": [
            _state("same", "C"),
            _state("same", "C"),
        ]
    }
    with pytest.raises(ValueError, match="duplicate state_key"):
        aggregate_dual_oracle(duplicate, [])

    one = {"per_state": [_state("same", "C")]}
    oft = {"per_state": [_state("same", None, oft_hit=False)]}
    with pytest.raises(ValueError, match="across suites"):
        aggregate_dual_oracle(one, [("a", oft), ("b", oft)])


def test_missing_suite_warns_and_legacy_headlines_remain():
    smol = {"per_state": [_state("x", "A")]}
    result = aggregate_dual_oracle(smol, [])

    assert result["per_state"][0]["suite"] is None
    assert result["per_state"][0]["cross_label"] == "uncertain"
    assert any("no suite metadata" in warning for warning in result["warnings"])
    for legacy_key in (
        "Y_Smol",
        "Y_OFT",
        "C_div",
        "n_recoverable_smolvla",
        "n_recoverable_oft",
        "n_divergent_oft_only",
        "per_candidate_gt",
    ):
        assert legacy_key in result


def test_candidate_recoverability_uses_wilson_lower_bound_when_available():
    smol_state = _state("x", "B")
    smol_state["candidates"][0].update({"successes": 5, "lower": 0.49})
    smol_state["candidates"][1].update({"successes": 5, "lower": 0.51})
    oft = {"per_state": [_state("x", None, oft_hit=False)]}
    result = aggregate_dual_oracle(
        {"protocol": {"threshold": 0.5}, "per_state": [smol_state]},
        [("suite", oft)],
    )
    candidates = result["per_candidate_gt"]
    assert candidates[0]["recoverable_smolvla"] is False
    assert candidates[1]["recoverable_smolvla"] is True
    assert result["per_state"][0]["dual_track_label"] == "smol_only"

import pytest

from rase.collect.policy_matrix import aggregate_one_shot_policy_matrix


def _summary(mode, rows, checksum="keys"):
    return {
        "mode": mode,
        "protocol": {"k": 2},
        "state_keys_provenance": {"state_keys_sha256": checksum},
        "per_state": [
            {
                "state_key": key,
                "candidates": [
                    {"successes": int(value), "trials": 1} for value in bits
                ],
            }
            for key, bits in rows.items()
        ],
    }


def test_one_shot_matrix_uses_state_as_inferential_unit():
    smol = _summary("smolvla-screen", {"a": [0, 0], "b": [1, 0]})
    oft = _summary("oft-verify", {"a": [0, 1], "b": [1, 0]})
    result = aggregate_one_shot_policy_matrix(
        ["a", "b"],
        smol,
        [("Spatial", oft)],
        pool_meta={
            "a": {"suite": "Spatial", "perturb_dim": "camera", "level": 1},
            "b": {"suite": "Spatial", "perturb_dim": "robot", "level": 2},
        },
        state_keys_sha256="keys",
        candidate_artifact_sha256="candidates",
    )
    assert result["smol_candidate"] == {"hits": 1, "trials": 4, "rate_descriptive": 0.25}
    assert result["oft_candidate"] == {"hits": 2, "trials": 4, "rate_descriptive": 0.5}
    assert result["state_pair_counts"] == {
        "both_hit": 1,
        "smol_only": 0,
        "oft_only": 1,
        "both_miss": 0,
    }
    assert result["candidate_pair_counts_descriptive"] == {
        "both_hit": 1,
        "smol_only": 0,
        "oft_only": 1,
        "both_miss": 2,
    }
    assert result["state_mcnemar_exact_p_two_sided"] == 1.0
    assert result["paired_state_effect"] == {
        "risk_difference_oft_minus_smol": 0.5,
        "discordant_pairs": 1,
        "oft_win_fraction_among_discordant": 1.0,
    }
    assert result["per_suite_state_pairs"] == [
        {
            "suite": "Spatial",
            "n_states": 2,
            "both_hit": 1,
            "smol_only": 0,
            "oft_only": 1,
            "both_miss": 0,
        }
    ]
    assert result["warnings"][-1].endswith(
        "n=2 and its uncertainty must be reported."
    )


def test_one_shot_matrix_rejects_incomplete_oft_union():
    smol = _summary("smolvla-screen", {"a": [0, 0], "b": [0, 0]})
    oft = _summary("oft-verify", {"a": [0, 0]})
    with pytest.raises(ValueError, match="union mismatch"):
        aggregate_one_shot_policy_matrix(
            ["a", "b"],
            smol,
            [("Spatial", oft)],
            pool_meta={},
            state_keys_sha256="keys",
            candidate_artifact_sha256="candidates",
        )

from scripts.analyze_pre_a0_candidate_opportunity import analyze


def test_pre_a0_opportunity_separates_generator_families() -> None:
    records = [
        {
            "state_key": f"s{i}",
            "task_id": f"task{i}",
            "episode_id": f"ep{i}",
            "suite": "Spatial" if i < 2 else "Goal",
            "perturbation_dimension": "clean",
            "perturbation_level": 0,
        }
        for i in range(4)
    ]
    keys = {
        "artifact_version": "rase-pre-a0-state-keys/v1",
        "selection_uses_outcomes": False,
        "state_keys": [f"s{i}" for i in range(4)],
        "state_keys_sha256": "frozen",
        "records": records,
    }
    candidate_outcomes = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    strict = {
        "state_keys_provenance": {"selected_state_keys_sha256": "frozen"},
        "per_state": [
            {
                "state_key": f"s{i}",
                "candidates": [
                    {"successes": int(value), "trials": 1}
                    for value in candidate_outcomes[i]
                ],
            }
            for i in range(4)
        ],
    }
    fallback = {
        "per_task": [
            {
                "state_key": f"s{i}",
                "task_id": f"task{i}",
                "oft_only_success": i == 2,
                "source_only_success": False,
                "source_to_oft_success": False,
            }
            for i in range(4)
        ]
    }
    result = analyze(keys, strict, fallback, bootstrap_replicates=100, bootstrap_seed=1)
    assert result["status"] == "pilot_signal_requires_scaled_heldout"
    assert result["metrics"]["strict_oracle_headroom"] == 0.25
    assert result["metrics"]["strict_mixed_outcome_states"] == 2
    assert result["metrics"]["heterogeneous_oracle_headroom"] == 0.5
    assert result["metrics"]["base_failure_rescue_fraction"] == 2 / 3
    assert result["portfolio"] == {
        "strict_only": 2,
        "fallback_only": 1,
        "both": 0,
        "neither": 1,
        "strict_vs_fallback_mcnemar_exact_p": 1.0,
    }


def test_pre_a0_rejects_non_one_shot_summary() -> None:
    keys = {
        "artifact_version": "rase-pre-a0-state-keys/v1",
        "selection_uses_outcomes": False,
        "state_keys": ["s0"],
        "state_keys_sha256": "x",
        "records": [
            {
                "state_key": "s0",
                "task_id": "t0",
                "episode_id": "e0",
                "suite": "Goal",
                "perturbation_dimension": "clean",
                "perturbation_level": 0,
            }
        ],
    }
    strict = {
        "state_keys_provenance": {"selected_state_keys_sha256": "x"},
        "per_state": [
            {
                "state_key": "s0",
                "candidates": [
                    {"successes": 1, "trials": 2},
                    {"successes": 0, "trials": 1},
                ],
            }
        ],
    }
    fallback = {
        "per_task": [
            {
                "state_key": "s0",
                "task_id": "t0",
                "oft_only_success": False,
                "source_only_success": False,
                "source_to_oft_success": False,
            }
        ]
    }
    try:
        analyze(keys, strict, fallback)
    except ValueError as exc:
        assert "one trial" in str(exc)
    else:
        raise AssertionError("expected non-one-shot input to fail")

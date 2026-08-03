from scripts.analyze_recovery_duration import analyze


def _inputs(outcomes, direct):
    duration = {
        "prefix_lengths": [0, 8, 32],
        "per_state": [
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "suite": "Goal",
                "perturbation_dimension": "camera",
                "perturbation_level": 1,
                "arms": [{"success": value} for value in row],
            }
            for i, row in enumerate(outcomes)
        ],
    }
    fallback = {
        "per_task": [
            {"state_key": f"s{i}", "oft_only_success": value}
            for i, value in enumerate(direct)
        ]
    }
    return duration, fallback


def test_duration_structure_signal_and_minimum_duration():
    duration, fallback = _inputs(
        [[False, True, True], [False, False, True], [False, False, False]],
        [True, True, True],
    )
    result = analyze(duration, fallback)
    assert result["status"] == "duration_structure_signal"
    assert result["fixed_duration_rescues"] == 2
    assert [row["minimum_successful_duration"] for row in result["per_state"]] == [8, 32, None]
    assert result["best_fixed_duration_successes"] == 2
    assert result["best_fixed_durations"] == [32]
    assert result["base_harmed_by_duration"] == {"8": 0, "32": 0}
    assert result["nonmonotonic_finite_duration_states"] == []


def test_episode_persistent_fallback_when_all_fixed_durations_fail():
    duration, fallback = _inputs(
        [[False, False, False], [False, False, False], [False, False, False]],
        [True, True, False],
    )
    result = analyze(duration, fallback)
    assert result["status"] == "episode_persistent_fallback"
    assert result["direct_only_rescues"] == 2


def test_reports_harm_and_nonmonotonic_finite_duration():
    duration, fallback = _inputs(
        [[True, True, False], [True, False, True], [False, False, False]],
        [True, True, False],
    )
    result = analyze(duration, fallback)
    assert result["base_harmed_by_duration"] == {"8": 1, "32": 1}
    assert result["nonmonotonic_finite_duration_states"] == ["s0"]
    assert result["per_state"][0]["harmed_durations"] == [32]

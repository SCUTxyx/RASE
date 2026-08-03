from scripts.analyze_replan_mechanism import analyze


def _inputs(short_successes: list[bool], direct_successes: list[bool]):
    prefix = {
        "per_state": [
            {
                "state_key": f"s{i}",
                "task_id": f"t{i}",
                "arms": [{"success": False}, {"success": success}],
            }
            for i, success in enumerate(short_successes)
        ]
    }
    fallback = {
        "per_task": [
            {"state_key": f"s{i}", "oft_only_success": success}
            for i, success in enumerate(direct_successes)
        ]
    }
    return prefix, fallback


def test_recovery_prefix_signal_requires_two_tasks_and_headroom():
    prefix, fallback = _inputs([True, True, False, False], [True] * 4)
    result = analyze(prefix, fallback)
    assert result["status"] == "recovery_prefix_model_signal"
    assert result["short_prefix_rescues"] == 2


def test_persistent_fallback_when_short_prefix_cannot_rescue():
    prefix, fallback = _inputs([False] * 4, [True, True, True, False])
    result = analyze(prefix, fallback)
    assert result["status"] == "persistent_fallback_required"
    assert result["direct_only_rescues"] == 3

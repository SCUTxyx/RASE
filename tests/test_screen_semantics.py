from scripts.rollout_pool_candidates import apply_screen_semantics


def test_screen_semantics_never_emits_formal_set_labels():
    summary = {
        "label_counts": {"A": 1, "uncertain": 1},
        "per_state": [
            {
                "state_key": "hit",
                "set_label": "A",
                "candidates": [{"successes": 1}, {"successes": 0}],
            },
            {
                "state_key": "miss",
                "set_label": "uncertain",
                "candidates": [{"successes": 0}, {"successes": 0}],
            },
        ],
    }
    apply_screen_semantics(summary)
    assert summary["formal_set_labels"] is False
    assert "label_counts" not in summary
    assert summary["diagnostic_label_counts"] == {"A": 1, "uncertain": 1}
    assert summary["screen_candidate_hits"] == 1
    assert summary["screen_state_hits"] == 1
    assert [row["set_label"] for row in summary["per_state"]] == [None, None]
    assert summary["per_state"][0]["diagnostic_set_label"] == "A"

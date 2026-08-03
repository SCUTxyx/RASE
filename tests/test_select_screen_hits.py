import hashlib
import json

import pytest

from scripts.select_screen_hits import candidate_hit_counts


def test_candidate_hit_counts():
    payload = {
        "per_state": [
            {
                "state_key": "a",
                "candidates": [
                    {"successes": 0},
                    {"successes": 1},
                    {"successes": 2},
                ],
            },
            {"state_key": "b", "candidates": [{"successes": 0}]},
        ]
    }
    assert candidate_hit_counts(payload) == {"a": 2, "b": 0}


def test_candidate_hit_counts_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        candidate_hit_counts(
            {
                "per_state": [
                    {"state_key": "same", "candidates": []},
                    {"state_key": "same", "candidates": []},
                ]
            }
        )


def test_frozen_key_checksum_matches_runner_contract():
    keys = ["sp1_a", "sp1_b"]
    expected = hashlib.sha256(
        json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    from scripts.rollout_pool_candidates import _state_keys_checksum

    assert _state_keys_checksum(keys) == expected

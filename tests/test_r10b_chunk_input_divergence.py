from __future__ import annotations

from scripts.audit_r10b_chunk_input_divergence import (
    classify_cell,
    compare_replica_pair,
    first_difference,
)


def _record(index: int, *, image: str = "image", action: str = "action") -> dict:
    return {
        "query_index": index, "action_offset": index * 8,
        "agentview_sha256": image, "agentview_shape": [2, 2, 3],
        "wrist_sha256": "wrist", "wrist_shape": [2, 2, 3],
        "proprio_sha256": "proprio", "proprio_shape": [9],
        "action_chunk_sha256": action, "action_chunk_shape": [8, 7],
    }


def test_first_difference_identifies_later_input_change() -> None:
    result = first_difference([_record(0), _record(1)], [_record(0), _record(1, image="changed")])
    assert result == {
        "query_index": 1, "action_offset": 8, "input_diff": True,
        "action_diff": False, "changed_fields": ["agentview_sha256"],
    }


def test_first_difference_identifies_matched_input_output_change() -> None:
    result = first_difference([_record(0)], [_record(0, action="changed")])
    assert result == {
        "query_index": 0, "action_offset": 0, "input_diff": False,
        "action_diff": True, "changed_fields": ["action_chunk_sha256"],
    }


def test_pairwise_classification_keeps_later_input_and_matched_output_causes() -> None:
    left = [_record(0), _record(1), _record(2)]
    right = [
        _record(0),
        _record(1, action="output-changed"),
        _record(2, image="input-changed"),
    ]
    result = compare_replica_pair(left, right)
    assert result["categories"] == [
        "B_CLOSED_LOOP_INPUT_DIVERGENCE",
        "C_MATCHED_INPUT_OUTPUT_DIVERGENCE",
    ]
    assert result["first_input_difference"]["query_index"] == 2
    assert result["first_matched_input_output_difference"]["query_index"] == 1


def test_cell_compares_all_three_replica_pairs() -> None:
    base = [_record(0), _record(1)]
    initial_changed = [_record(0, image="initial-changed"), _record(1)]
    shorter = [_record(0)]
    result = classify_cell([base, initial_changed, shorter])
    assert set(result["pairwise"]) == {
        "rep0_vs_rep1", "rep0_vs_rep2", "rep1_vs_rep2",
    }
    assert "A_INITIAL_INPUT_DIVERGENCE" in result["categories"]
    assert "B_CLOSED_LOOP_INPUT_DIVERGENCE" in result["categories"]

from __future__ import annotations

from scripts.audit_r10b_chunk_input_divergence import first_difference


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

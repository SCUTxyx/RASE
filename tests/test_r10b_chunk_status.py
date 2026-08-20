from __future__ import annotations

import json

from scripts.status_r10b_chunk_diagnostic import metadata_path, snapshot


def _manifest() -> dict:
    return {
        "status": "frozen_diagnostic", "expected_trajectories": 3,
        "records": [{
            "state_key": "state", "seed_index": 1, "suite": "Goal",
            "policy_id": "pi05_libero", "group_id": "state:pi05:seed1",
        }],
    }


def test_progress_counts_only_contract_valid_chunk_traces(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    root = tmp_path / "out"
    for replica in (0, 1):
        path = metadata_path(root, _manifest()["records"][0], replica)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"elapsed_source_steps": boundary, "persistent_chunk_query_records": [{"q": 1}]} for boundary in (8, 16)]
        path.write_text(json.dumps({"rows": rows}))
    invalid = metadata_path(root, _manifest()["records"][0], 2)
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text(json.dumps({"rows": []}))
    result = snapshot(manifest_path, root, tmp_path / "audit.json")
    assert result["complete"] == 2
    assert result["expected"] == 3
    assert result["reason_counts"] == {"complete": 2, "missing_trace_t8": 1}

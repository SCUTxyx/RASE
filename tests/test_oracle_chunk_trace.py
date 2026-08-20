from __future__ import annotations

import numpy as np

from rase.collect.oracle_continuation import OracleChunkContinuation


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, inputs, *, payload):
        del inputs, payload
        self.calls += 1
        start = (self.calls - 1) * 2
        return {"actions": np.arange(start, start + 14, dtype=np.float32).reshape(1, 2, 7)}


def test_chunk_trace_records_only_query_inputs(monkeypatch) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)
    proprio = np.arange(9, dtype=np.float32)
    monkeypatch.setattr(
        "rase.collect.oracle_continuation.raw_libero_to_oracle_arrays",
        lambda control_env: (image, wrist, proprio),
    )
    client = _Client()
    policy = OracleChunkContinuation(
        client, instruction="task", control_env=object(), record_chunk_trace=True,
    )

    for _ in range(3):
        policy.act({}, task="task")

    assert client.calls == 2
    assert [row["query_index"] for row in policy.chunk_query_records] == [0, 1]
    assert [row["action_offset"] for row in policy.chunk_query_records] == [0, 2]
    assert all(row["agentview_shape"] == [8, 8, 3] for row in policy.chunk_query_records)
    assert all(row["wrist_shape"] == [8, 8, 3] for row in policy.chunk_query_records)
    assert all(row["proprio_shape"] == [9] for row in policy.chunk_query_records)
    assert all(row["action_chunk_shape"] == [2, 7] for row in policy.chunk_query_records)
    assert all(set(row) == {
        "query_index", "action_offset", "agentview_sha256", "agentview_shape",
        "wrist_sha256", "wrist_shape", "proprio_sha256", "proprio_shape",
        "action_chunk_sha256", "action_chunk_shape",
    } for row in policy.chunk_query_records)

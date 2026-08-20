from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rase.vnext.candidate_capture import (
    CAPTURE_SCHEMA,
    array_sha256,
    audit_candidate_capture,
    write_candidate_capture,
)


def _first(action: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(action, dtype=np.float32).reshape(-1, 7)[0].tobytes()).hexdigest()


def _chunk(capability="executable", origin="executed_trace", actions=None, **kwargs):
    return type(
        "CaptureChunk",
        (),
        {
            "capability": capability,
            "chunk_origin": origin,
            "actions": actions,
            "mask_reason": kwargs.pop("mask_reason", None),
            "inference_event_id": kwargs.pop("inference_event_id", None),
            "queue_cursor_at_boundary": kwargs.pop("queue_cursor_at_boundary", None),
            "candidate_generation_seed": kwargs.pop("candidate_generation_seed", None),
            "native_chunk_sha256": kwargs.pop("native_chunk_sha256", None),
            "full_env_chunk": kwargs.pop("full_env_chunk", None),
            "boundary_action": kwargs.pop("boundary_action", None),
        },
    )()


def _common_kwargs(tmp_path, chunks, firsts, instruction="pick up the bowl"):
    return dict(
        output_dir=tmp_path,
        group_key=("root", "policy", "point", 0),
        operator_chunks=chunks,
        instruction=instruction,
        task_id="task", suite="Spatial", policy_id="policy",
        decision_point_id="source.step.8", replica=0,
        seed_ledger={"seed": 1},
        proprio=np.zeros(8, dtype=np.float32),
        proprio_mask=np.ones(8, dtype=np.bool_),
        images={"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
        executed_first_action_sha256=firsts,
        fallback_full_action_trace=None,
    )


def test_capture_v2_roundtrip_full_chunks(tmp_path) -> None:
    cont = np.arange(21, dtype=np.float32).reshape(3, 7)
    requery = np.ones((10, 7), dtype=np.float32)
    fallback = -np.ones((4, 7), dtype=np.float32)
    chunks = {
        "continue.source": _chunk(
            "executable", "inference_event", cont,
            full_env_chunk=cont, queue_cursor_at_boundary=0,
            boundary_action=cont[0], inference_event_id="ev-1",
            native_chunk_sha256="abc",
        ),
        "requery.source": _chunk(
            "executable", "forced_inference", requery,
            full_env_chunk=requery, queue_cursor_at_boundary=0,
            boundary_action=requery[0], inference_event_id="ev-2",
        ),
        "resample.source": _chunk("incapable_missing", "contract_mask", mask_reason="no_diversity"),
        "fallback.persistent": _chunk(
            "executable", "executed_trace", fallback,
            full_env_chunk=fallback, queue_cursor_at_boundary=0,
            boundary_action=fallback[0],
        ),
        "abort.safe": _chunk("control_only_abort", "control_only", mask_reason="safe_abort_control_event"),
    }
    firsts = {op: _first(chunks[op].actions) for op in ("continue.source", "requery.source", "fallback.persistent")}
    result = write_candidate_capture(**_common_kwargs(tmp_path, chunks, firsts))
    assert result["schema_version"] == CAPTURE_SCHEMA
    assert result["status"] == "COMPLETE"
    assert all(result["capture_execution_alignment"].values()), result["capture_execution_alignment"]
    audit = audit_candidate_capture(tmp_path / (result["metadata_path"].split("/")[-1]))
    assert audit == {"status": "PASS", "failures": []}


def test_capture_v2_continue_cursor_suffix(tmp_path) -> None:
    """Boundary inside a partially consumed chunk: actions must be the suffix."""
    full = np.arange(70, dtype=np.float32).reshape(10, 7)
    cursor = 6
    suffix = full[cursor:]
    boundary = full[cursor - 1]
    chunks = {
        "continue.source": _chunk(
            "executable", "inference_event", suffix,
            full_env_chunk=full, queue_cursor_at_boundary=cursor,
            boundary_action=boundary, inference_event_id="ev-1",
        ),
        "requery.source": _chunk(
            "executable", "forced_inference", np.ones((10, 7), dtype=np.float32),
            full_env_chunk=np.ones((10, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=np.ones(7, dtype=np.float32),
        ),
        "resample.source": _chunk("incapable_missing", "contract_mask", mask_reason="no_diversity"),
        "fallback.persistent": _chunk(
            "executable", "executed_trace", -np.ones((3, 7), dtype=np.float32),
            full_env_chunk=-np.ones((3, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=-np.ones(7, dtype=np.float32),
        ),
        "abort.safe": _chunk("control_only_abort", "control_only", mask_reason="safe_abort"),
    }
    firsts = {
        "continue.source": _first(boundary),
        "requery.source": _first(chunks["requery.source"].actions),
        "fallback.persistent": _first(chunks["fallback.persistent"].actions),
    }
    result = write_candidate_capture(**_common_kwargs(tmp_path, chunks, firsts))
    assert all(result["capture_execution_alignment"].values()), result["capture_execution_alignment"]
    audit = audit_candidate_capture(tmp_path / (result["metadata_path"].split("/")[-1]))
    assert audit == {"status": "PASS", "failures": []}
    meta = result["operators"]["continue.source"]
    assert meta["queue_cursor_at_boundary"] == cursor
    assert meta["chunk_origin"] == "inference_event"


def test_capture_v2_mismatched_executed_first_fails(tmp_path) -> None:
    cont = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunks = {
        "continue.source": _chunk(
            "executable", "inference_event", cont,
            full_env_chunk=cont, queue_cursor_at_boundary=0,
            boundary_action=cont[0], inference_event_id="ev-1",
        ),
        "requery.source": _chunk(
            "executable", "forced_inference", np.ones((10, 7), dtype=np.float32),
            full_env_chunk=np.ones((10, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=np.ones(7, dtype=np.float32),
        ),
        "resample.source": _chunk("incapable_missing", "contract_mask", mask_reason="no_diversity"),
        "fallback.persistent": _chunk(
            "executable", "executed_trace", -np.ones((4, 7), dtype=np.float32),
            full_env_chunk=-np.ones((4, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=-np.ones(7, dtype=np.float32),
        ),
        "abort.safe": _chunk("control_only_abort", "control_only", mask_reason="safe_abort"),
    }
    wrong = {"continue.source": "0" * 64, "requery.source": "0" * 64, "fallback.persistent": "0" * 64}
    result = write_candidate_capture(**_common_kwargs(tmp_path, chunks, wrong))
    assert not all(result["capture_execution_alignment"].values())
    audit = audit_candidate_capture(tmp_path / (result["metadata_path"].split("/")[-1]))
    assert audit["status"] == "FAIL"
    assert "capture_execution_alignment" in audit["failures"]


def test_capture_v2_incapable_requires_reason(tmp_path) -> None:
    chunks = {
        "continue.source": _chunk(
            "executable", "inference_event", np.arange(21, dtype=np.float32).reshape(3, 7),
            full_env_chunk=np.arange(21, dtype=np.float32).reshape(3, 7),
            queue_cursor_at_boundary=0,
            boundary_action=np.arange(7, dtype=np.float32),
        ),
        "requery.source": _chunk(
            "executable", "forced_inference", np.ones((10, 7), dtype=np.float32),
            full_env_chunk=np.ones((10, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=np.ones(7, dtype=np.float32),
        ),
        "resample.source": _chunk("incapable_missing", "contract_mask", mask_reason=None),
        "fallback.persistent": _chunk(
            "executable", "executed_trace", -np.ones((4, 7), dtype=np.float32),
            full_env_chunk=-np.ones((4, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=-np.ones(7, dtype=np.float32),
        ),
        "abort.safe": _chunk("control_only_abort", "control_only", mask_reason="safe_abort"),
    }
    firsts = {op: _first(chunks[op].actions) for op in ("continue.source", "requery.source", "fallback.persistent")}
    result = write_candidate_capture(**_common_kwargs(tmp_path, chunks, firsts))
    audit = audit_candidate_capture(tmp_path / (result["metadata_path"].split("/")[-1]))
    assert audit["status"] == "FAIL"
    assert "missing_mask_reason:resample.source" in audit["failures"]


def test_capture_v2_abort_control_only_record(tmp_path) -> None:
    cont = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunks = {
        "continue.source": _chunk(
            "executable", "inference_event", cont,
            full_env_chunk=cont, queue_cursor_at_boundary=0,
            boundary_action=cont[0], inference_event_id="ev-1",
        ),
        "requery.source": _chunk(
            "executable", "forced_inference", np.ones((10, 7), dtype=np.float32),
            full_env_chunk=np.ones((10, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=np.ones(7, dtype=np.float32),
        ),
        "resample.source": _chunk("incapable_missing", "contract_mask", mask_reason="no_diversity"),
        "fallback.persistent": _chunk(
            "executable", "executed_trace", -np.ones((4, 7), dtype=np.float32),
            full_env_chunk=-np.ones((4, 7), dtype=np.float32),
            queue_cursor_at_boundary=0, boundary_action=-np.ones(7, dtype=np.float32),
        ),
        "abort.safe": _chunk("control_only_abort", "control_only", mask_reason="safe_abort"),
    }
    firsts = {op: _first(chunks[op].actions) for op in ("continue.source", "requery.source", "fallback.persistent")}
    result = write_candidate_capture(**_common_kwargs(tmp_path, chunks, firsts))
    meta = result["operators"]["abort.safe"]
    assert meta["capability"] == "control_only_abort"
    assert meta["chunk_origin"] == "control_only"
    assert "abort.safe" not in result["candidate_chunk_sha256"]


def test_capture_v2_requires_continue_and_requery(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires continue.source and requery.source"):
        write_candidate_capture(**_common_kwargs(tmp_path, {"fallback.persistent": _chunk()}, {}))

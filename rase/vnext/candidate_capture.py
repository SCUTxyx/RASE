"""Immutable, contemporaneous candidate-action capture artifacts (K3-E0 v2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CAPTURE_SCHEMA = "rase-vnext-candidate-capture/v2"

# Frozen capability schema (K3 protocol §4): never treat an incapable or
# control-only candidate as an ordinary failure.
CAPABILITIES = (
    "executable",
    "incapable_missing",
    "incapable_short_chunk",
    "incapable_invalid_action",
    "incapable_actuator_mismatch",
    "control_only_abort",
    "execution_error",
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_digest(group_key: Sequence[object]) -> str:
    token = "\x1f".join(map(str, group_key)).encode()
    return hashlib.sha256(token).hexdigest()[:24]


def pad_action_chunk(
    actions: np.ndarray, *, horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != 7 or len(value) < 1:
        raise ValueError(f"action chunk must be non-empty [T,7], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("action chunk contains non-finite values")
    count = min(len(value), horizon)
    padded = np.zeros((horizon, 7), dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.bool_)
    padded[:count] = value[:count]
    mask[:count] = True
    return padded, mask


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _chunk_record(chunk: Any) -> dict[str, Any]:
    """Normalize a CaptureChunk-like object into a plain record."""
    if chunk is None:
        raise ValueError("capture chunk record is required for every operator slot")
    if isinstance(chunk, dict):
        return chunk
    capability = str(getattr(chunk, "capability", "executable"))
    record: dict[str, Any] = {
        "capability": capability,
        "chunk_origin": str(getattr(chunk, "chunk_origin", "")),
        "mask_reason": getattr(chunk, "mask_reason", None),
        "inference_event_id": getattr(chunk, "inference_event_id", None),
        "queue_cursor_at_boundary": getattr(chunk, "queue_cursor_at_boundary", None),
        "candidate_generation_seed": getattr(chunk, "candidate_generation_seed", None),
        "native_chunk_sha256": getattr(chunk, "native_chunk_sha256", None),
        "actions": getattr(chunk, "actions", None),
        "full_env_chunk": getattr(chunk, "full_env_chunk", None),
        "boundary_action": getattr(chunk, "boundary_action", None),
    }
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown capability {capability!r}")
    return record


def write_candidate_capture(
    output_dir: Path,
    *,
    group_key: Sequence[object],
    operator_chunks: Mapping[str, Any],
    instruction: str,
    task_id: str,
    suite: str,
    policy_id: str,
    decision_point_id: str,
    replica: int,
    seed_ledger: Mapping[str, Any],
    proprio: np.ndarray,
    proprio_mask: np.ndarray,
    images: Mapping[str, np.ndarray],
    executed_first_action_sha256: Mapping[str, str],
    fallback_full_action_trace: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write one immutable NPZ+JSON pair before branch rows are committed.

    Every frozen operator slot must appear (executable or not) so capability
    status is auditable and incapable slots can never be silently dropped.
    """
    if not instruction.strip():
        raise ValueError("candidate capture requires a non-empty instruction")
    operators = tuple(sorted(operator_chunks))
    if not {"continue.source", "requery.source"}.issubset(operators):
        raise ValueError("capture requires continue.source and requery.source")

    records = {operator: _chunk_record(operator_chunks[operator]) for operator in operators}
    chunks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    chunk_hashes: dict[str, str] = {}
    first_hashes: dict[str, str] = {}
    alignment: dict[str, bool] = {}
    extra_arrays: dict[str, np.ndarray] = {}
    for operator in operators:
        record = records[operator]
        capability = str(record["capability"])
        actions = record.get("actions")
        full = record.get("full_env_chunk")
        cursor = record.get("queue_cursor_at_boundary")
        boundary = record.get("boundary_action")
        if capability == "executable":
            if actions is None or len(actions) == 0:
                raise ValueError(f"executable operator {operator} has no actions")
            padded, mask = pad_action_chunk(np.asarray(actions, dtype=np.float32))
            chunks.append(padded)
            masks.append(mask)
            valid = padded[mask]
            chunk_hashes[operator] = array_sha256(valid)
            first_hashes[operator] = array_sha256(valid[0].astype(np.float32, copy=False))
            if full is not None:
                full_array = np.asarray(full, dtype=np.float32)
                extra_arrays[f"full_env_chunk_{operator}"] = full_array
                if cursor is not None:
                    cursor_int = int(cursor)
                    if not 0 <= cursor_int <= len(full_array):
                        raise ValueError(f"{operator}: cursor {cursor_int} outside full chunk {len(full_array)}")
                    if cursor_int < len(full_array):
                        suffix = full_array[cursor_int:]
                        if not np.array_equal(suffix, valid):
                            raise ValueError(f"{operator}: actions != full_env_chunk[cursor:]")
            if boundary is not None:
                boundary_array = np.asarray(boundary, dtype=np.float32).reshape(-1, 7)
                extra_arrays[f"boundary_action_{operator}"] = boundary_array
            executed = executed_first_action_sha256.get(operator)
            if executed is not None:
                if boundary is not None and cursor is not None and int(cursor) >= 1 and full is not None:
                    ok = (
                        array_sha256(np.asarray(boundary, dtype=np.float32).reshape(-1, 7)[0]) == str(executed)
                        and array_sha256(np.asarray(full, dtype=np.float32)[int(cursor) - 1]) == str(executed)
                    )
                elif boundary is not None:
                    ok = (
                        array_sha256(np.asarray(boundary, dtype=np.float32).reshape(-1, 7)[0]) == str(executed)
                        and array_sha256(valid[0]) == str(executed)
                    )
                else:
                    ok = first_hashes[operator] == str(executed)
                alignment[operator] = bool(ok)
        else:
            chunks.append(np.zeros((10, 7), dtype=np.float32))
            masks.append(np.zeros(10, dtype=np.bool_))
            if capability == "control_only_abort":
                alignment[operator] = executed_first_action_sha256.get(operator) is None

    digest = capture_digest(group_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{digest}.npz"
    meta_path = output_dir / f"{digest}.json"
    if npz_path.exists() or meta_path.exists():
        raise FileExistsError(f"candidate capture already exists for {group_key}")
    arrays: dict[str, np.ndarray] = {
        "actions": np.stack(chunks),
        "action_step_mask": np.stack(masks),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "proprio_mask": np.asarray(proprio_mask, dtype=np.bool_),
    }
    for role, image in sorted(images.items()):
        arrays[f"image_{role}"] = np.asarray(image, dtype=np.uint8)
    if fallback_full_action_trace is not None:
        arrays["fallback_full_action_trace"] = np.asarray(
            fallback_full_action_trace, dtype=np.float32,
        )
    for name, value in extra_arrays.items():
        arrays[name] = value
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(npz_path)

    serializable_records: dict[str, dict[str, Any]] = {}
    for operator in operators:
        record = dict(records[operator])
        for key in ("actions", "full_env_chunk", "boundary_action"):
            record.pop(key, None)
        serializable_records[operator] = record
    payload = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "COMPLETE",
        "group_key": list(group_key),
        "task_id": task_id,
        "suite": suite,
        "policy_id": policy_id,
        "decision_point_id": decision_point_id,
        "exact_repeat_replica": int(replica),
        "instruction": instruction,
        "operator_order": list(operators),
        "seed_ledger": dict(seed_ledger),
        "operators": serializable_records,
        "candidate_chunk_sha256": chunk_hashes,
        "candidate_first_action_sha256": first_hashes,
        "executed_first_action_sha256": dict(executed_first_action_sha256),
        "capture_execution_alignment": alignment,
        "fallback_full_action_trace_sha256": (
            array_sha256(fallback_full_action_trace)
            if fallback_full_action_trace is not None else None
        ),
        "arrays_path": str(npz_path.resolve()),
        "arrays_sha256": file_sha256(npz_path),
    }
    _atomic_json(meta_path, payload)
    return {**payload, "metadata_path": str(meta_path.resolve())}


def audit_candidate_capture(meta_path: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text())
    failures: list[str] = []
    if meta.get("schema_version") != CAPTURE_SCHEMA:
        failures.append("schema_version")
    arrays_path = Path(str(meta.get("arrays_path", "")))
    if not arrays_path.exists():
        failures.append("arrays_missing")
        return {"status": "FAIL", "failures": failures}
    if file_sha256(arrays_path) != meta.get("arrays_sha256"):
        failures.append("arrays_sha256")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        actions = arrays["actions"]
        masks = arrays["action_step_mask"]
        for index, operator in enumerate(meta["operator_order"]):
            record = meta.get("operators", {}).get(operator, {})
            capability = str(record.get("capability", "executable"))
            if capability == "executable":
                valid = actions[index][masks[index]]
                if len(valid) < 1:
                    failures.append(f"empty_chunk:{operator}")
                if array_sha256(valid) != meta["candidate_chunk_sha256"][operator]:
                    failures.append(f"chunk_hash:{operator}")
                if array_sha256(valid[0].astype(np.float32, copy=False)) != meta[
                    "candidate_first_action_sha256"
                ][operator]:
                    failures.append(f"first_hash:{operator}")
            elif capability != "control_only_abort":
                # incapable / execution_error slots must carry a reason.
                if not record.get("mask_reason"):
                    failures.append(f"missing_mask_reason:{operator}")
        if "fallback_full_action_trace" in arrays:
            if array_sha256(arrays["fallback_full_action_trace"]) != meta.get(
                "fallback_full_action_trace_sha256"
            ):
                failures.append("fallback_full_trace_hash")
    if not all(meta.get("capture_execution_alignment", {}).values()):
        failures.append("capture_execution_alignment")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}

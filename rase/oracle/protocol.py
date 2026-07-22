"""Versioned ZeroMQ wire protocol: one JSON frame plus raw array frames."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_NAME = "rase-oracle"
PROTOCOL_VERSION = 1
OPERATIONS = frozenset({"health", "model-info", "predict"})
MAX_ARRAY_BYTES = 512 * 1024 * 1024


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    header: dict[str, Any]
    arrays: dict[str, np.ndarray]


def request(
    operation: str,
    *,
    request_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if operation not in OPERATIONS:
        raise ProtocolError(f"unsupported operation: {operation}")
    return {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "kind": "request",
        "operation": operation,
        "request_id": request_id or uuid.uuid4().hex,
        "payload": dict(payload or {}),
    }


def response(
    request_header: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    header = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "kind": "response",
        "operation": request_header.get("operation"),
        "request_id": request_header.get("request_id"),
        "ok": error is None,
        "payload": dict(payload or {}),
    }
    if error is not None:
        header["error"] = str(error)
    return header


def validate_header(header: Mapping[str, Any], *, kind: str | None = None) -> None:
    if header.get("protocol") != PROTOCOL_NAME:
        raise ProtocolError("unknown protocol")
    if header.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {header.get('version')}")
    if header.get("kind") not in {"request", "response"}:
        raise ProtocolError("message kind must be request or response")
    if kind is not None and header.get("kind") != kind:
        raise ProtocolError(f"expected {kind} message")
    if header.get("operation") not in OPERATIONS:
        raise ProtocolError(f"unsupported operation: {header.get('operation')}")
    if not isinstance(header.get("request_id"), str) or not header["request_id"]:
        raise ProtocolError("request_id must be a non-empty string")
    if not isinstance(header.get("payload"), dict):
        raise ProtocolError("payload must be a JSON object")
    if header.get("kind") == "response" and not isinstance(header.get("ok"), bool):
        raise ProtocolError("response ok must be boolean")


def encode_message(
    header: Mapping[str, Any], arrays: Mapping[str, Any] | None = None
) -> list[bytes]:
    """Encode a multipart message without pickle or base64."""
    value = dict(header)
    validate_header(value)
    descriptors = []
    frames: list[bytes] = []
    for name, raw in (arrays or {}).items():
        if not isinstance(name, str) or not name:
            raise ProtocolError("array names must be non-empty strings")
        array = np.asarray(raw)
        if array.dtype.hasobject:
            raise ProtocolError("object arrays are forbidden")
        array = np.ascontiguousarray(array)
        if array.nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError(f"array {name!r} exceeds size limit")
        descriptors.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
            }
        )
        frames.append(array.tobytes(order="C"))
    value["arrays"] = descriptors
    try:
        json_frame = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"header is not valid JSON: {exc}") from exc
    return [json_frame, *frames]


def decode_message(frames: Sequence[bytes]) -> Message:
    if not frames:
        raise ProtocolError("empty multipart message")
    try:
        header = json.loads(bytes(frames[0]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(header, dict):
        raise ProtocolError("JSON frame must contain an object")
    validate_header(header)
    descriptors = header.get("arrays")
    if not isinstance(descriptors, list):
        raise ProtocolError("arrays descriptor must be a list")
    if len(frames) != len(descriptors) + 1:
        raise ProtocolError("array descriptor/frame count mismatch")
    arrays: dict[str, np.ndarray] = {}
    for descriptor, frame in zip(descriptors, frames[1:]):
        if not isinstance(descriptor, dict):
            raise ProtocolError("invalid array descriptor")
        try:
            name = descriptor["name"]
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(value) for value in descriptor["shape"])
            nbytes = int(descriptor["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("invalid array descriptor fields") from exc
        if not isinstance(name, str) or not name or name in arrays:
            raise ProtocolError("array names must be unique non-empty strings")
        if dtype.hasobject or any(value < 0 for value in shape):
            raise ProtocolError("unsafe dtype or invalid shape")
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != nbytes or nbytes != len(frame) or nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError(f"invalid byte length for array {name!r}")
        arrays[name] = np.frombuffer(bytes(frame), dtype=dtype).reshape(shape)
    return Message(header=header, arrays=arrays)

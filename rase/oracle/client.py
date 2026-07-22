"""Synchronous client for the versioned oracle service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .protocol import ProtocolError, decode_message, encode_message, request


class OracleRemoteError(RuntimeError):
    pass


class OracleClient:
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        timeout_ms: int = 30_000,
        context: Any | None = None,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required to use OracleClient") from exc
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._owned_context = context is None
        self._context = context or zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        socket = self._context.socket(self._zmq.REQ)
        socket.linger = 0
        socket.rcvtimeo = self.timeout_ms
        socket.sndtimeo = self.timeout_ms
        socket.connect(self.endpoint)
        self._socket = socket

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._owned_context and self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self) -> OracleClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _call(
        self,
        operation: str,
        *,
        payload: Mapping[str, Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self._socket is None:
            raise RuntimeError("client is closed")
        outgoing = request(operation, payload=payload)
        try:
            self._socket.send_multipart(encode_message(outgoing, arrays))
            incoming = decode_message(self._socket.recv_multipart())
        except self._zmq.Again as exc:
            # REQ sockets enter an invalid state after timeout; rebuild.
            self._connect()
            raise TimeoutError(f"oracle {operation} timed out") from exc
        if incoming.header["request_id"] != outgoing["request_id"]:
            raise ProtocolError("response request_id does not match request")
        if incoming.header["operation"] != operation:
            raise ProtocolError("response operation does not match request")
        if not incoming.header["ok"]:
            raise OracleRemoteError(incoming.header.get("error", "oracle request failed"))
        return incoming.header["payload"], incoming.arrays

    def health(self) -> dict[str, Any]:
        payload, _ = self._call("health")
        return payload

    def model_info(self) -> dict[str, Any]:
        payload, _ = self._call("model-info")
        model = payload.get("model")
        if not isinstance(model, dict):
            raise ProtocolError("model-info response lacks model object")
        return model

    def predict(
        self,
        arrays: Mapping[str, Any],
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        _, outputs = self._call("predict", payload=payload, arrays=arrays)
        if not outputs:
            raise ProtocolError("predict response contains no arrays")
        return outputs

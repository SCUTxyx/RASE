"""Injectable ZeroMQ oracle server; no model framework is imported here."""

from __future__ import annotations

import argparse
import importlib
import signal
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    response,
    validate_header,
)


class ModelAdapter(Protocol):
    def model_info(self) -> Mapping[str, Any]: ...

    def predict(
        self, arrays: Mapping[str, np.ndarray], payload: Mapping[str, Any]
    ) -> Mapping[str, np.ndarray] | np.ndarray: ...


class OracleServer:
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def dispatch(self, frames: list[bytes]) -> list[bytes]:
        """Handle one request. Errors are returned as protocol responses."""
        request_header: Mapping[str, Any] = {
            "operation": "health",
            "request_id": "invalid",
        }
        try:
            message = decode_message(frames)
            validate_header(message.header, kind="request")
            request_header = message.header
            operation = request_header["operation"]
            if operation == "health":
                header = response(
                    request_header,
                    payload={"status": "ok", "protocol_version": PROTOCOL_VERSION},
                )
                return encode_message(header)
            if operation == "model-info":
                info = dict(self.adapter.model_info())
                header = response(request_header, payload={"model": info})
                return encode_message(header)
            if not message.arrays:
                raise ProtocolError("predict requires at least one input array")
            output = self.adapter.predict(message.arrays, request_header["payload"])
            output_arrays = (
                {"actions": output} if isinstance(output, np.ndarray) else dict(output)
            )
            if not output_arrays:
                raise ProtocolError("adapter returned no arrays")
            header = response(
                request_header, payload={"output_names": list(output_arrays)}
            )
            return encode_message(header, output_arrays)
        except Exception as exc:
            # Preserve correlation when the request JSON was valid enough to parse.
            header = response(request_header, error=f"{type(exc).__name__}: {exc}")
            return encode_message(header)

    def serve(
        self,
        endpoint: str,
        *,
        context: Any | None = None,
        stop_after: int | None = None,
    ) -> None:
        """Run a REP socket. ``stop_after`` exists for smoke tests."""
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required to run the oracle server") from exc
        owned_context = context is None
        context = context or zmq.Context()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(endpoint)
        served = 0
        try:
            while stop_after is None or served < stop_after:
                socket.send_multipart(self.dispatch(socket.recv_multipart()))
                served += 1
        finally:
            socket.close()
            if owned_context:
                context.term()


def load_adapter(specification: str) -> ModelAdapter:
    """Load ``module:factory`` only in the heavyweight server environment."""
    if ":" not in specification:
        raise ValueError("adapter must be specified as module:factory")
    module_name, factory_name = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    adapter = factory()
    if not hasattr(adapter, "predict") or not hasattr(adapter, "model_info"):
        raise TypeError("adapter must implement predict() and model_info()")
    return adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--adapter", required=True, help="module:zero_argument_factory")
    args = parser.parse_args(argv)
    server = OracleServer(load_adapter(args.adapter))
    signal.signal(signal.SIGTERM, lambda *_: raise_system_exit())
    server.serve(args.endpoint)
    return 0


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())

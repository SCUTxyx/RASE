"""Timeout rebuilds the REQ socket so the next call is not stuck in EFSM."""

from __future__ import annotations

import pytest

from rase.oracle.client import OracleClient


class _Again(Exception):
    pass


class _FakeSocket:
    def __init__(self):
        self.closed = False
        self.sent = 0

    def send_multipart(self, frames):
        del frames
        self.sent += 1

    def recv_multipart(self):
        raise _Again()

    def close(self, _linger=0):
        self.closed = True

    def connect(self, _endpoint):
        return None


class _FakeContext:
    def __init__(self):
        self.sockets = []

    def socket(self, _type):
        sock = _FakeSocket()
        self.sockets.append(sock)
        return sock

    def term(self):
        return None


def test_timeout_rebuilds_socket(monkeypatch):
    zmq = type(
        "zmq",
        (),
        {"REQ": 1, "Again": _Again, "Context": _FakeContext},
    )
    monkeypatch.setitem(__import__("sys").modules, "zmq", zmq)

    ctx = _FakeContext()
    client = OracleClient("tcp://127.0.0.1:9", timeout_ms=10, context=ctx)
    assert len(ctx.sockets) == 1
    with pytest.raises(TimeoutError, match="timed out"):
        client.health()
    assert len(ctx.sockets) == 2
    assert ctx.sockets[0].closed is True
    client.close()

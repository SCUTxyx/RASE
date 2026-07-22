import json

import numpy as np
import pytest

from rase.oracle.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    request,
)
from rase.oracle.server import OracleServer


class FakeAdapter:
    def model_info(self):
        return {"name": "fake", "action_dim": 7}

    def predict(self, arrays, payload):
        return {"actions": arrays["observations"][:, :7] * payload.get("scale", 1)}


def test_json_and_raw_array_round_trip():
    header = request("predict", request_id="r1", payload={"temperature": 0.5})
    array = np.arange(24, dtype=np.float32).reshape(3, 8)
    frames = encode_message(header, {"observations": array})
    assert json.loads(frames[0])["version"] == PROTOCOL_VERSION
    assert len(frames) == 2

    decoded = decode_message(frames)
    assert decoded.header["request_id"] == "r1"
    np.testing.assert_array_equal(decoded.arrays["observations"], array)


def test_health_model_info_and_predict_dispatch():
    server = OracleServer(FakeAdapter())
    health = decode_message(server.dispatch(encode_message(request("health"))))
    assert health.header["ok"]
    assert health.header["payload"]["status"] == "ok"

    info = decode_message(server.dispatch(encode_message(request("model-info"))))
    assert info.header["payload"]["model"]["name"] == "fake"

    observations = np.ones((2, 9), dtype=np.float32)
    predicted = decode_message(
        server.dispatch(
            encode_message(
                request("predict", payload={"scale": 2}),
                {"observations": observations},
            )
        )
    )
    np.testing.assert_array_equal(
        predicted.arrays["actions"], np.full((2, 7), 2, dtype=np.float32)
    )


def test_rejects_wrong_version_and_bad_frame_length():
    header = request("health")
    header["version"] = 99
    with pytest.raises(ProtocolError):
        encode_message(header)

    valid = encode_message(request("predict"), {"x": np.zeros(2)})
    with pytest.raises(ProtocolError):
        decode_message(valid[:-1])


def test_adapter_error_is_protocol_response():
    server = OracleServer(FakeAdapter())
    message = decode_message(server.dispatch(encode_message(request("predict"))))
    assert message.header["ok"] is False
    assert "requires at least one input array" in message.header["error"]

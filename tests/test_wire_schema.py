import numpy as np
import pytest

from rase.oracle.wire_schema import (
    WireSchemaError,
    proprio_to_policy_state,
    validate_predict_inputs,
)


def test_validate_predict_inputs_ok():
    arrays = {
        "agentview": np.zeros((2, 64, 64, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 64, 64, 3), dtype=np.uint8),
        "proprio": np.zeros((2, 8), dtype=np.float32),
    }
    batch, instructions, fmt = validate_predict_inputs(
        arrays,
        {"instructions": ["a", "b"], "proprio_format": "policy_state", "max_batch": 2},
    )
    assert batch == 2
    assert instructions == ["a", "b"]
    assert fmt == "policy_state"


def test_validate_rejects_bad_dtype_and_batch():
    arrays = {
        "agentview": np.zeros((1, 8, 8, 3), dtype=np.float32),
        "wrist": np.zeros((1, 8, 8, 3), dtype=np.float32),
        "proprio": np.zeros((1, 8), dtype=np.float32),
    }
    with pytest.raises(WireSchemaError, match="uint8"):
        validate_predict_inputs(arrays, {"instructions": ["x"]})


def test_proprio_policy_state_passthrough_and_raw_quat():
    state = np.arange(8, dtype=np.float32)
    assert np.allclose(proprio_to_policy_state(state), state)
    # Identity-ish quat (0,0,0,1) → zero axis-angle.
    raw = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    out = proprio_to_policy_state(raw, proprio_format="raw_quat")
    assert out.shape == (8,)
    assert np.allclose(out[:3], [0.1, 0.2, 0.3])
    assert np.allclose(out[3:6], 0.0, atol=1e-6)

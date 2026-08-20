from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_r8a1_rep3_pilot", ROOT / "scripts" / "audit_r8a1_rep3_pilot.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wilson_gate_is_frozen_at_three_disagreements() -> None:
    assert MODULE.wilson_upper(0, 96) < 0.10
    assert MODULE.wilson_upper(3, 96) < 0.10
    assert MODULE.wilson_upper(4, 96) > 0.10


def test_array_equal_requires_shape_and_dtype() -> None:
    import numpy as np

    value = np.asarray([1.0, 2.0], dtype=np.float32)
    assert MODULE.array_equal(value, value.copy(), 1e-6)
    assert not MODULE.array_equal(value, value.astype(np.float64), 1e-6)
    assert not MODULE.array_equal(value, value.reshape(2, 1), 1e-6)

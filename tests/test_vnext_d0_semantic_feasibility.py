from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_rase_vnext_d0_semantic_feasibility.py"
SPEC = importlib.util.spec_from_file_location("d0_semantic", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_d0_perturbations_are_finite_bounded_and_distinct() -> None:
    actions = np.linspace(-0.9, 0.9, 70, dtype=np.float32).reshape(10, 7)
    transformed = [MODULE.transform(actions, name) for name in MODULE.PERTURBATIONS]
    assert all(value.shape == (10, 7) for value in transformed)
    assert all(np.isfinite(value).all() and np.max(np.abs(value)) <= 1 for value in transformed)
    assert len({value.tobytes() for value in transformed}) == len(transformed)


def test_gripper_phase_shift_inverts_constant_gripper() -> None:
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    shifted = MODULE.transform(actions, "gripper_phase_shift")
    np.testing.assert_array_equal(shifted[:, 6], 1.0)

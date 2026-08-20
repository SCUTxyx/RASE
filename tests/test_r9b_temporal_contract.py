from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_probe_uses_causal_history_shapes() -> None:
    module = load("audit_r9c_information_gate.py")
    n = 6
    data = {
        "image_history": np.zeros((n, 4, 2, 3, 8, 8), dtype=np.uint8),
        "proprio_history": np.zeros((n, 4, 8), dtype=np.float32),
        "proprio_delta_history": np.zeros((n, 4, 8), dtype=np.float32),
        "action_history": np.zeros((n, 4, 7), dtype=np.float32),
        "language_hash": np.zeros((n, 256), dtype=np.float32),
    }
    groups = module.probe_features(data)
    assert groups["image_sequence"].shape[0] == n
    assert groups["temporal_state"].shape[0] == n
    assert groups["action_history"].shape[0] == n
    assert groups["all_causal"].shape[0] == n


def test_auc_is_tie_half_and_rejects_single_class() -> None:
    module = load("audit_r9c_information_gate.py")
    assert module.auc(np.asarray([0, 1]), np.asarray([0.2, 0.8])) == 1.0
    assert module.auc(np.asarray([0, 1]), np.asarray([0.5, 0.5])) == 0.5
    assert np.isnan(module.auc(np.asarray([1, 1]), np.asarray([0.1, 0.2])))


def test_repro_accepts_only_success_prefixes() -> None:
    module = load("audit_r9b_temporal_repro.py")
    assert module.valid_boundary_set({0}, True)
    assert module.valid_boundary_set({0, 4, 8}, True)
    assert module.valid_boundary_set({0, 4, 8, 12, 16}, False)
    assert not module.valid_boundary_set({0, 8}, True)
    assert not module.valid_boundary_set({0, 4}, False)

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


def test_seed_parser_and_deterministic_rank() -> None:
    module = load("freeze_r10b_case_control_manifest.py")
    assert module.seed_from_group("state:pi05_libero:seed3") == 3
    assert module.rank("same") == module.rank("same")
    assert module.rank("same") != module.rank("different")


def test_case_control_feature_contract() -> None:
    module = load("audit_r10c_case_control_information.py")
    n = 10
    data = {
        "image_history": np.zeros((n, 8, 2, 3, 8, 8), np.uint8),
        "proprio_history": np.zeros((n, 8, 8), np.float32),
        "proprio_delta_history": np.zeros((n, 8, 8), np.float32),
        "proprio_accel_history": np.zeros((n, 8, 8), np.float32),
        "action_history": np.zeros((n, 8, 7), np.float32),
        "action_delta_history": np.zeros((n, 8, 7), np.float32),
        "language_hash": np.zeros((n, 256), np.float32),
    }
    groups = module.summarize_features(data)
    assert set(groups) == {"image_sequence", "temporal_state", "action_history",
                           "semantic", "temporal_plus_action", "all_causal"}
    assert all(value.shape[0] == n for value in groups.values())


def test_auc_ties_and_task_bootstrap() -> None:
    module = load("audit_r10c_case_control_information.py")
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.0, 1.0, 0.0, 1.0])
    tasks = np.asarray(["a", "a", "b", "b"])
    assert module.auc(labels, scores) == 1.0
    interval = module.task_bootstrap_auc(labels, scores, tasks, trials=20)
    assert interval == (1.0, 1.0, 1.0)

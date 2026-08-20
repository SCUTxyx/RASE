from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def load_module():
    # The logic under test does not instantiate the neural network.  Stub the
    # project-only imports so this unit test also runs in the lightweight audit
    # environment.
    risk = types.ModuleType("rase.risk.light_risk_student")
    risk.CandidateArmStudent = object
    encoder = types.ModuleType("rase.risk.tiny_universal_state_encoder")
    encoder.TinyUniversalStateEncoder = object
    sys.modules.setdefault("rase", types.ModuleType("rase"))
    sys.modules.setdefault("rase.risk", types.ModuleType("rase.risk"))
    sys.modules["rase.risk.light_risk_student"] = risk
    sys.modules["rase.risk.tiny_universal_state_encoder"] = encoder
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_r6c1_early_selector.py"
    spec = importlib.util.spec_from_file_location("r6c1_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_data():
    # g1 succeeds under source; g2 is rescuable; g3 fails under both arms.
    groups = np.asarray(["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3)
    elapsed = np.asarray([0, 8, 16] * 3)
    source = np.asarray([1] * 3 + [0] * 6, dtype=np.float32)
    persistent = np.asarray([1] * 6 + [0] * 3, dtype=np.float32)
    steps = np.asarray([30] * 3 + [50] * 3 + [60] * 3, dtype=np.float32)
    return {
        "group_id": groups,
        "elapsed_source_steps": elapsed,
        "arm_success": np.stack([source, persistent], axis=-1),
        "arm_teacher_steps": np.stack([np.zeros(9), steps], axis=-1),
        "state_key": np.asarray(["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3),
        "task_id": np.asarray(["t1"] * 3 + ["t2"] * 3 + ["t3"] * 3),
        "suite": np.asarray(["a"] * 9),
        "policy_id": np.asarray(["p"] * 9),
    }


def test_group_scores_and_episode_records_are_aligned():
    module = load_module()
    data = synthetic_data()
    # g1 safe, g2 high-risk at t0, g3 safe.  Group-local position reuse would
    # incorrectly apply g1 scores to g2 and miss the rescue.
    lcb = np.asarray([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    advantage = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    metrics = module.controller_early_window(
        data, np.arange(9), lcb, advantage, risk_thr=0.5, adv_thr=0.5)
    assert metrics["entered"] == 1
    assert metrics["successes"] == 2
    assert metrics["teacher_steps"] == 50
    records = {row["group_id"]: row for row in metrics["trajectories"]}
    assert records["g1"]["controller_success"] is True
    assert records["g1"]["controller_teacher_steps"] == 0
    assert records["g2"]["controller_success"] is True
    assert records["g2"]["controller_teacher_steps"] == 50
    assert records["g2"]["enter_elapsed_source_steps"] == 0
    assert records["g3"]["controller_success"] is False


def test_conditional_missed_rescue_uses_rescue_opportunities():
    module = load_module()
    data = synthetic_data()
    lcb = np.ones(9)
    advantage = np.zeros(9)
    metrics = module.controller_early_window(
        data, np.arange(9), lcb, advantage, risk_thr=0.5, adv_thr=0.5)
    # Baseline succeeds on g1 and g2, but only g2 is a rescue opportunity.
    assert metrics["false_continue_rate"] == 0.5
    assert metrics["rescue_opportunities"] == 1
    assert metrics["conditional_missed_rescue_rate"] == 1.0


def test_record_aggregation_preserves_bootstrap_multiplicity():
    module = load_module()
    data = synthetic_data()
    metrics = module.controller_early_window(
        data, np.arange(9), np.ones(9), np.zeros(9), risk_thr=0.5, adv_thr=0.5)
    records = metrics["trajectories"]
    doubled = module.metrics_from_trajectory_records(records + records)
    assert doubled["episodes"] == 6
    assert doubled["false_continue"] == 2
    assert doubled["false_continue_rate"] == metrics["false_continue_rate"]
    assert doubled["conditional_missed_rescue_rate"] == metrics["conditional_missed_rescue_rate"]


def test_absolute_harm_counts_late_entry_rescue_decay():
    module = load_module()
    data = synthetic_data()
    # g2 is rescuable at t0 but no longer rescuable at t8.  The controller
    # waits at t0 and enters at t8, so false-continue is zero but paired harm
    # relative to t0-persistent must be counted.
    data["arm_success"][4, 1] = 0.0
    lcb = np.asarray([0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
    advantage = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    metrics = module.controller_early_window(
        data, np.arange(9), lcb, advantage, risk_thr=0.5, adv_thr=0.5)
    assert metrics["false_continue"] == 0
    assert metrics["absolute_paired_harm"] == 1 / 3
    records = {row["group_id"]: row for row in metrics["trajectories"]}
    assert records["g2"]["enter_elapsed_source_steps"] == 8
    assert records["g2"]["paired_harm"] is True

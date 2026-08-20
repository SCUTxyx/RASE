from __future__ import annotations

from pathlib import Path

from rase.adapt.pre_c1 import (
    analyze_pre_c1_recovery_gate,
    episode_grouped_split,
    load_protocol_lock,
    validate_protocol_lock,
)


def test_protocol_lock_schema():
    path = Path("artifacts/pre_c1/pre_c1_protocol_lock.yaml")
    payload = load_protocol_lock(path)
    assert validate_protocol_lock(payload) == []
    assert payload["sealed"]["world_model_gate"] == "closed"
    assert payload["gate"]["recovery_gain_pp"] == 8.0


def test_protocol_lock_c1_1_schema():
    path = Path("artifacts/pre_c1/pre_c1_1_protocol_lock.yaml")
    payload = load_protocol_lock(path)
    assert validate_protocol_lock(payload) == []
    assert payload["phase"] == "PRE-C1.1"
    assert payload["dataset"]["teacher_horizon_mode"] == "persistent_min128_from_fork"
    assert int(payload["dataset"]["teacher_horizon_steps"]) == 0
    assert int(payload["dataset"]["teacher_min_steps_from_fork"]) == 128
    assert payload["dataset"]["stages"] == ["T0", "T1", "T2", "T3", "T4"]
    assert payload["gate"]["recovery_gain_pp"] == 8.0
    assert payload["gate"]["clean_retention_drop_pp"] == 2.0


def test_episode_grouped_split_no_leakage():
    rows = [
        {"episode_id": f"e{i // 2}", "state_key": f"s{i}", "clean_flag": False}
        for i in range(8)
    ]
    splits = episode_grouped_split(rows, seed=2026080405, val_fraction=0.25)
    assert splits["leakage_episode_overlap"] == []
    assert splits["n_train_rows"] + splits["n_val_rows"] == 8


def test_recovery_gate_pass_and_abstention():
    recovery = [
        {
            "episode_id": f"e{i}",
            "state_key": f"r{i}",
            "base_success": False,
            "adapted_success": True,
        }
        for i in range(10)
    ]
    retention = [
        {
            "episode_id": f"c{i}",
            "state_key": f"c{i}",
            "base_success": True,
            "adapted_success": True,
        }
        for i in range(10)
    ]
    audit = analyze_pre_c1_recovery_gate(
        recovery_rows=recovery,
        retention_rows=retention,
        recovery_gain_pp=8.0,
        clean_retention_drop_pp=2.0,
        bootstrap_replicates=200,
    )
    assert audit["gate_pass"] is True
    assert audit["decision"] == "same_backbone_recovery_method_eligible"

    bad_ret = [
        {
            "episode_id": f"c{i}",
            "state_key": f"c{i}",
            "base_success": True,
            "adapted_success": False,
        }
        for i in range(10)
    ]
    audit2 = analyze_pre_c1_recovery_gate(
        recovery_rows=recovery,
        retention_rows=bad_ret,
        bootstrap_replicates=200,
    )
    assert audit2["decision"] == "abstention_track_required"

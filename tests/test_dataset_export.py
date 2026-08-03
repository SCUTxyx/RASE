from pathlib import Path

import numpy as np
import pytest

from rase.collect.candidates import make_artifact
from rase.collect.dataset_export import (
    audit_split_support,
    build_grouped_benchmark_splits,
    build_leave_one_suite_out_splits,
    build_recovery_rows,
    split_state_keys,
)
from rase.collect.schema import StateMetadata


def _meta():
    return StateMetadata(
        task_id="libero_spatial_000001",
        instruction="do task",
        suite="Spatial",
        episode_id="ep-1",
        step=4,
        perturb_dim="camera",
        perturb_sub="viewpoint",
        level=3,
        episode_outcome="failure",
        seed=1,
    )


def _artifact():
    return make_artifact(
        np.zeros((8, 10, 7), dtype=np.float32),
        seeds=range(8),
        temperature=0.7,
        policy_hash="hash",
    )


def test_build_recovery_rows_joins_candidate_gt():
    key = _meta().state_key
    summary = {
        "per_state": [
            {
                "state_key": key,
                "set_label_smolvla": "C",
                "dual_track_label": "oft_only",
                "recoverable_smolvla": False,
                "recoverable_oft": True,
            }
        ],
        "per_candidate_gt": [
            {
                "state_key": key,
                "candidate_id": 2,
                "successes_smolvla": 0,
                "trials_smolvla": 6,
                "successes_oft": 1,
                "trials_oft": 1,
                "recoverable_smolvla": False,
                "recoverable_oft": True,
            }
        ],
    }
    rows = build_recovery_rows(
        summary,
        metadata_for=lambda _key: _meta(),
        artifact_for=lambda _key: (Path("/tmp/candidate.npz"), _artifact()),
    )
    assert len(rows) == 8
    assert rows[2]["successes_oft"] == 1
    assert rows[2]["trials_smolvla"] == 6
    assert rows[2]["dual_track_label"] == "oft_only"
    assert rows[2]["t0"] == 4
    assert rows[2]["episode_id"] == "ep-1"
    assert rows[2]["perturb_sub"] == "viewpoint"
    assert rows[2]["state_seed"] == 1
    assert split_state_keys(rows) == {"oft_only": [key]}


def test_build_recovery_rows_rejects_duplicate_states():
    key = _meta().state_key
    state = {"state_key": key}
    with pytest.raises(ValueError, match="duplicate state"):
        build_recovery_rows(
            {"per_state": [state, state]},
            metadata_for=lambda _key: _meta(),
            artifact_for=lambda _key: (Path("/tmp/candidate.npz"), _artifact()),
        )


def _split_row(state_key, task_id, episode_id, *, suite, dim, level, label):
    return {
        "state_key": state_key,
        "candidate_id": 0,
        "task_id": task_id,
        "episode_id": episode_id,
        "suite": suite,
        "perturb_dim": dim,
        "perturb_sub": "x",
        "level": level,
        "dual_track_label": label,
        "episode_outcome": "failure",
    }


def test_grouped_benchmark_split_is_deterministic_and_episode_disjoint():
    rows = []
    for episode in range(12):
        for step in range(2):
            key = f"state-{episode}-{step}"
            row = _split_row(
                key,
                f"task-{episode % 4}",
                f"ep-{episode}",
                suite=("Spatial", "Object")[episode % 2],
                dim=("camera", "robot")[episode % 2],
                level=3 + episode % 3,
                label=("oft_only", "both_fail")[episode % 2],
            )
            rows.extend([row, {**row, "candidate_id": 1}])

    first = build_grouped_benchmark_splits(rows, seed=7)
    second = build_grouped_benchmark_splits(list(reversed(rows)), seed=7)
    assert first == second
    assert first["audit"]["n_rows"] == 48
    assert first["audit"]["n_states"] == 24
    assert first["audit"]["n_groups"] == 12
    assert first["audit"]["group_leakage"] is False

    state_split = {key: split for split, keys in first["splits"].items() for key in keys}
    for episode in range(12):
        assert state_split[f"state-{episode}-0"] == state_split[f"state-{episode}-1"]
    expected = {f"state-{episode}-{step}" for episode in range(12) for step in range(2)}
    assert set(state_split) == expected


def test_grouped_benchmark_split_rejects_inconsistent_candidate_metadata():
    row = _split_row(
        "state", "task", "episode", suite="Spatial", dim="camera", level=3, label="oft_only"
    )
    with pytest.raises(ValueError, match="inconsistent metadata"):
        build_grouped_benchmark_splits([row, {**row, "level": 4}])


def test_leave_one_suite_out_has_no_suite_or_group_leakage():
    rows = [
        _split_row("a", "ta", "ea", suite="Spatial", dim="camera", level=3, label="x"),
        _split_row("b", "tb", "eb", suite="Object", dim="camera", level=3, label="x"),
        _split_row("c", "tc", "ec", suite="Goal", dim="camera", level=3, label="x"),
    ]
    result = build_leave_one_suite_out_splits(rows)
    assert result["suites"] == ["Goal", "Object", "Spatial"]
    for suite, fold in result["folds"].items():
        test = set(fold["splits"]["test"])
        train = set(fold["splits"]["train"])
        assert not test & train
        assert {row["suite"] for row in rows if row["state_key"] in test} == {suite}
        assert suite not in {row["suite"] for row in rows if row["state_key"] in train}


def test_split_support_failure_is_explicit_and_does_not_resample():
    rows = [
        {
            **_split_row("a", "ta", "ea", suite="Spatial", dim="camera", level=3, label="x"),
            "arms": {
                "continue_smol": {"observed": True, "success": True, "cost": 0.02},
                "escalate_oft": {"observed": True, "success": True, "cost": 0.10},
                "abstain": {"observed": True, "success": False, "cost": 0.0},
            },
        }
    ]
    splits = {"splits": {"train": ["a"], "val": [], "test": []}}
    original = {name: list(keys) for name, keys in splits["splits"].items()}
    audit = audit_split_support(
        rows,
        splits,
        requirements={"min_states_per_split": 1, "min_train_optimal_actions": 2},
    )
    assert audit["status"] == "NOT_READY"
    assert audit["ready"] is False
    assert audit["reasons"]
    assert splits["splits"] == original


def _support_row(key, task, episode, suite, cohort, smol, oft):
    return {
        **_split_row(key, task, episode, suite=suite, dim="camera", level=3, label="x"),
        "cohort": cohort,
        "arms": {
            "continue_smol": {"observed": True, "success": smol, "cost": 0.02},
            "escalate_oft": {"observed": True, "success": oft, "cost": 0.10},
            "abstain": {"observed": True, "success": False, "cost": 0.0},
        },
    }


def test_split_support_rejects_task_test_without_failure_cohort():
    rows = [
        _support_row("train-f", "train-task", "ep-f", "Spatial", "failure_challenge", False, True),
        _support_row("train-c", "train-task", "ep-c", "Spatial", "clean_control", True, False),
        *[
            _support_row(
                f"test-{index}",
                "heldout-task",
                f"ep-{index}",
                "Object",
                "clean_control",
                True,
                False,
            )
            for index in range(8)
        ],
    ]
    splits = {
        "splits": {
            "train": ["train-f", "train-c"],
            "test": [f"test-{index}" for index in range(8)],
        }
    }
    audit = audit_split_support(
        rows,
        splits,
        requirements={
            "required_splits": ["train", "test"],
            "required_cohorts": {"test": ["clean_control", "failure_challenge"]},
            "min_states_per_cohort": {"test": 1},
            "min_episode_groups_per_split": {"train": 2, "test": 8},
            "required_optimal_actions": {"train": ["continue_smol", "escalate_oft"]},
            "min_train_states": 1,
            "min_train_suites": 1,
            "min_train_optimal_actions": 2,
        },
    )
    assert audit["status"] == "NOT_READY"
    assert audit["per_split"]["test"]["cohort_counts"] == {
        "clean_control": 8,
        "failure_challenge": 0,
    }
    assert audit["per_split"]["test"]["n_episode_groups"] == 8
    assert audit["per_split"]["test"]["optimal_action_counts"]["escalate_oft"] == 0
    assert any(
        "split test cohort 'failure_challenge' has 0 states" in reason
        for reason in audit["reasons"]
    )


def test_leave_suite_out_fold_requirements_gate_each_train_and_test():
    rows = [
        _support_row("sp-c", "sp", "c", "Spatial", "clean_control", True, False),
        _support_row("sp-f", "sp", "f", "Spatial", "failure_challenge", False, True),
        _support_row("ob-c", "ob", "c", "Object", "clean_control", True, False),
    ]
    loso = build_leave_one_suite_out_splits(rows)
    requirements = {
        "required_splits": ["train", "test"],
        "required_cohorts": {
            "train": ["clean_control", "failure_challenge"],
            "test": ["clean_control", "failure_challenge"],
        },
        "min_states_per_cohort": 1,
        "min_train_states": 1,
        "min_train_suites": 1,
        "min_train_optimal_actions": 1,
    }
    audits = {
        suite: audit_split_support(rows, fold, requirements=requirements)
        for suite, fold in loso["folds"].items()
    }
    assert audits["Object"]["status"] == "NOT_READY"
    assert audits["Spatial"]["status"] == "NOT_READY"
    assert any(
        "split test cohort 'failure_challenge'" in reason for reason in audits["Object"]["reasons"]
    )
    assert any(
        "split train cohort 'failure_challenge'" in reason
        for reason in audits["Spatial"]["reasons"]
    )

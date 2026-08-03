import hashlib
import json

import numpy as np
import pytest

from rase.collect.prefix_ablation import (
    action_prefix_sha256,
    build_decision_suffix_prefix_arms,
    summarize_decision_suffix_prefix_state,
)
from scripts.analyze_suffix_prefix_mechanism import analyze
from scripts.freeze_timing_disagreement_keys import freeze_cohort


def _record(steps: int, success: bool, suffix: np.ndarray) -> dict:
    actions = suffix[:steps]
    return {
        "arm_label": f"suffix_prefix_{steps}",
        "prefix_steps": steps,
        "prefix_completed": True,
        "prefix_sha256": action_prefix_sha256(actions),
        "success": success,
        "continuation_steps": 10,
        "terminal_during_prefix": False,
        "prefix_translation_l2_sum": 0.0,
        "prefix_rotation_l2_sum": 0.0,
        "prefix_gripper_abs_sum": 0.0,
    }


def _source_row(key: str, classification: str, suffix: np.ndarray) -> dict:
    direct_success = classification == "direct_only"
    return {
        "state_key": key,
        "classification": classification,
        "arms": [
            {
                "arm_label": "direct_oft",
                "success": direct_success,
                "prefix_sha256": action_prefix_sha256(suffix[:0]),
            },
            {
                "arm_label": "decision_suffix_oft",
                "success": not direct_success,
                "prefix_sha256": action_prefix_sha256(suffix),
            },
        ],
    }


def test_build_suffix_prefix_grid_preserves_every_exact_prefix() -> None:
    suffix = np.arange(35, dtype=np.float32).reshape(5, 7)
    arms = build_decision_suffix_prefix_arms(suffix)
    assert [arm.label for arm in arms] == [f"suffix_prefix_{k}" for k in range(6)]
    for steps, arm in enumerate(arms):
        assert arm.candidate_index == steps
        np.testing.assert_array_equal(arm.actions, suffix[:steps])


def test_suffix_prefix_summary_counts_nonmonotonic_flips() -> None:
    suffix = np.zeros((5, 7), dtype=np.float32)
    records = [
        _record(steps, success, suffix)
        for steps, success in enumerate([True, False, True, True, False, False])
    ]
    result = summarize_decision_suffix_prefix_state(
        "state-a", records, expected_suffix_steps=5
    )
    assert result["success_pattern"] == "101100"
    assert result["success_flip_steps"] == [1, 2, 4]
    assert result["single_transition"] is False


def test_suffix_prefix_summary_rejects_incomplete_prefix() -> None:
    suffix = np.zeros((5, 7), dtype=np.float32)
    records = [_record(k, k < 3, suffix) for k in range(6)]
    records[2]["prefix_completed"] = False
    with pytest.raises(ValueError, match="incomplete prefix"):
        summarize_decision_suffix_prefix_state(
            "state-a", records, expected_suffix_steps=5
        )


def test_freeze_cohort_requires_frozen_disagreement_counts(tmp_path) -> None:
    source_path = tmp_path / "analysis.json"
    analysis = {
        "schema_version": "rase-deferred-switch-analysis/v1",
        "status": "complete",
        "state_keys_sha256": "source-keys",
        "per_state": [
            {"state_key": "a", "classification": "direct_only", "suite": "Goal"},
            {"state_key": "b", "classification": "deferred_only", "suite": "Goal"},
        ],
    }
    source_path.write_text(json.dumps(analysis), encoding="utf-8")
    result = freeze_cohort(
        analysis,
        source_path=source_path,
        expected_direct_only=1,
        expected_deferred_only=1,
    )
    assert result["selection_outcome_conditioned"] is True
    assert result["state_keys"] == ["a", "b"]
    with pytest.raises(ValueError, match="counts changed"):
        freeze_cohort(
            analysis,
            source_path=source_path,
            expected_direct_only=2,
            expected_deferred_only=1,
        )


def test_analysis_requires_endpoint_success_and_hash_parity() -> None:
    suffix = np.arange(35, dtype=np.float32).reshape(5, 7)
    keys = ["a", "b"]
    checksum = hashlib.sha256(
        json.dumps(keys, separators=(",", ":")).encode()
    ).hexdigest()
    cohort = {
        "schema_version": "rase-timing-disagreement-cohort/v1",
        "selection_outcome_conditioned": True,
        "state_keys": keys,
        "state_keys_sha256": checksum,
        "expected_source_classification": {
            "a": "direct_only",
            "b": "deferred_only",
        },
    }
    source = {
        "schema_version": "rase-deferred-switch-analysis/v1",
        "per_state": [
            _source_row("a", "direct_only", suffix),
            _source_row("b", "deferred_only", suffix),
        ],
    }
    rows = []
    for key, pattern in (("a", "111000"), ("b", "000111")):
        rows.append(
            {
                "state_key": key,
                "suite": "Goal",
                "dim": "robot",
                "level": 1,
                "arms": [
                    _record(k, hit == "1", suffix) for k, hit in enumerate(pattern)
                ],
            }
        )
    summary = {
        "schema_version": "rase-oft-decision-suffix-prefix-grid/v1",
        "status": "complete",
        "suite": "Goal",
        "state_keys_sha256": checksum,
        "per_state": rows,
    }
    result = analyze(
        cohort, source, [("Goal", summary)], expected_suffix_steps=5
    )
    assert result["status"] == "complete"
    assert result["endpoint_parity"]["n_pass"] == 2
    assert result["single_transition_fraction"] == 1.0
    assert result["curve_classification_counts"] == {
        "single_transition_deferred_rescue": 1,
        "single_transition_direct_harm": 1,
    }
    assert result["shared_scalar_boundary"]["status"] == "pass"
    assert result["shared_scalar_boundary"]["boundary_prefix_steps"] == [3]
    assert result["scientific_decision"]["status"] == (
        "targeted_independent_screen_may_be_designed"
    )

    summary["per_state"][0]["arms"][5]["success"] = True
    invalid = analyze(
        cohort, source, [("Goal", summary)], expected_suffix_steps=5
    )
    assert invalid["status"] == "invalid_endpoint_parity"

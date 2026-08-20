from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_deviation_timeline import analyze
from scripts.mine_deviation_snapshots import (
    MiningConfig,
    compute_component_timeline,
    extract_active_suffix,
    label_stage_indices,
)


def _timeline(
    visual_deltas,
    *,
    config: MiningConfig,
    proprios=None,
    suffixes=None,
):
    count = len(visual_deltas)
    return compute_component_timeline(
        steps=list(range(count)),
        proprios=proprios or [np.asarray([0.0])] * count,
        suffixes=suffixes or [np.asarray([0.0])] * count,
        agentview_deltas=visual_deltas,
        wrist_deltas=[None] * count,
        config=config,
    )


def test_component_timeline_and_threshold_labels_are_deterministic():
    config = MiningConfig(
        visual_delta_threshold=0.1,
        deviation_score_threshold=1.5,
        failure_score_threshold=2.5,
        sustained_steps=2,
        visual_weight=1.0,
        velocity_weight=0.0,
        jerk_weight=0.0,
        suffix_weight=0.0,
        no_progress_weight=0.0,
    )
    timeline = _timeline(
        [None, 0.01, 0.20, 0.20, 0.25, 0.30, 0.40, 0.40],
        config=config,
    )

    labels, fallback, reasons = label_stage_indices(timeline, config)

    assert labels == {"T0": 1, "T1": 2, "T2": 3, "T3": 4, "T4": 7}
    assert not fallback
    assert reasons == []
    assert timeline[2]["visual_frame_delta"] == pytest.approx(0.20)
    assert timeline[2]["deviation_score"] == pytest.approx(2.0)


def test_no_progress_streak_is_a_deviation_component():
    config = MiningConfig(
        no_progress_streak_threshold=2,
        deviation_score_threshold=1.0,
        failure_score_threshold=2.0,
        sustained_steps=2,
        visual_weight=0.0,
        velocity_weight=0.0,
        jerk_weight=0.0,
        suffix_weight=0.0,
        no_progress_weight=1.0,
    )
    timeline = _timeline([None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], config=config)

    labels, fallback, _ = label_stage_indices(timeline, config)

    assert [row["no_progress_streak"] for row in timeline] == [0, 1, 2, 3, 4, 5, 6]
    assert labels == {"T0": 1, "T1": 2, "T2": 3, "T3": 4, "T4": 6}
    assert not fallback


def test_temporal_fallback_is_explicit_strict_and_unique():
    config = MiningConfig(
        visual_weight=1.0,
        velocity_weight=0.0,
        jerk_weight=0.0,
        suffix_weight=0.0,
        no_progress_weight=0.0,
    )
    timeline = _timeline([None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], config=config)

    labels, fallback, reasons = label_stage_indices(timeline, config)

    indices = list(labels.values())
    assert indices == [0, 1, 3, 4, 6]
    assert fallback
    assert reasons == ["first_deviation_not_detected_in_valid_window"]
    assert indices == sorted(set(indices))


def test_fewer_than_five_snapshots_cannot_satisfy_stage_contract():
    config = MiningConfig()
    timeline = _timeline([None, 0.0, 0.0, 0.0], config=config)
    with pytest.raises(ValueError, match="at least 5 snapshots"):
        label_stage_indices(timeline, config)


def test_velocity_jerk_and_nested_suffix_norm_are_computed_without_gpu():
    suffix = extract_active_suffix(
        {"policy": {"cache": {"active_suffix": [[3.0, 4.0], [0.0, 0.0]]}}}
    )
    assert suffix is not None
    assert np.linalg.norm(suffix) == pytest.approx(5.0)
    config = MiningConfig()
    timeline = compute_component_timeline(
        steps=[0, 2, 4],
        proprios=[np.asarray([0.0]), np.asarray([2.0]), np.asarray([6.0])],
        suffixes=[suffix, suffix, suffix],
        agentview_deltas=[None, 0.1, 0.1],
        wrist_deltas=[None, None, None],
        config=config,
    )
    assert timeline[1]["proprio_velocity"] == pytest.approx(1.0)
    assert timeline[2]["proprio_velocity"] == pytest.approx(2.0)
    assert timeline[2]["proprio_jerk"] == pytest.approx(0.5)
    assert timeline[0]["active_suffix_norm"] == pytest.approx(5.0)


def test_design_threshold_aliases_are_loaded():
    config = MiningConfig.from_design(
        {
            "deviation_mining": {
                "thresholds": {
                    "visual_delta": 0.2,
                    "no_progress_streak": 4,
                    "failure_score": 3.0,
                },
                "weights": {"visual": 1.0, "no_progress": 0.0},
            }
        }
    )
    assert config.visual_delta_threshold == pytest.approx(0.2)
    assert config.no_progress_streak_threshold == 4
    assert config.failure_score_threshold == pytest.approx(3.0)
    assert config.visual_weight == pytest.approx(1.0)
    assert config.no_progress_weight == pytest.approx(0.0)


def test_analyzer_summarizes_order_fallback_and_reliability():
    stages = {
        stage: {
            "index": index,
            "step": index * 2,
            "state_key": f"key-{index}",
        }
        for index, stage in enumerate(("T0", "T1", "T2", "T3", "T4"))
    }
    keys = {
        "schema_version": "rase-pre-c0-deviation-keys/v1",
        "provenance": {"design_sha256": "design", "pool_manifest_sha256": "pool"},
        "episodes": [
            {
                "episode_id": "ep-1",
                "suite": "Spatial",
                "cell": "camera:L1",
                "temporal_fallback": False,
                "temporal_fallback_reasons": [],
                "reliability": {
                    "reliable": True,
                    "score": 0.9,
                    "signal_coverage": 0.9,
                },
                "stages": stages,
            },
            {
                "episode_id": "ep-2",
                "suite": "Spatial",
                "cell": "robot:L1",
                "temporal_fallback": True,
                "temporal_fallback_reasons": ["T3_threshold_not_detected"],
                "reliability": {
                    "reliable": False,
                    "score": 0.4,
                    "signal_coverage": 0.8,
                },
                "stages": stages,
            },
        ],
    }

    summary = analyze(keys)

    assert summary["stage_counts"] == {stage: 2 for stage in stages}
    assert summary["ordering"]["all_strict"]
    assert summary["temporal_fallback"]["rate"] == pytest.approx(0.5)
    assert summary["reliability"]["reliable_rate"] == pytest.approx(0.5)
    assert summary["by_suite"]["Spatial"]["episodes"] == 2
    assert summary["qc_pass"]

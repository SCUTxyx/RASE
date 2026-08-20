from __future__ import annotations

import numpy as np

from rase.vnext.adapter_parity import (
    audit_action_roundtrip,
    audit_motion_trace_conversion,
    audit_resample_capability,
    build_capability_report,
)
from rase.vnext.libero import (
    LIBERO_ACTION_SEMANTICS,
    LIBERO_MOTION_SEMANTIC_MAP,
    LiberoPolicyAdapter,
)
from rase.vnext.motion_trace import MotionSemanticMap, action_to_motion_trace
from rase.vnext.schema import CanonicalActionToken


def _token(values: np.ndarray, *, step_mask: np.ndarray | None = None) -> CanonicalActionToken:
    return CanonicalActionToken.from_array(
        values,
        semantics=LIBERO_ACTION_SEMANTICS,
        control_hz=10.0,
        coordinate_frame="robot.base.relative",
        policy_id="pi05.libero",
        step_mask=step_mask,
    )


def test_motion_trace_integrates_translation_rotation_and_derivatives() -> None:
    values = np.array([
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 1.0],
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    trace = action_to_motion_trace(
        _token(values),
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
        workspace_bounds=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(trace.integrated_ee_pose_rel[:, 0], [0.005, 0.01, 0.005])
    np.testing.assert_allclose(trace.velocity[:, 0], [0.05, 0.05, -0.05])
    assert trace.derivative_valid_order.tolist() == [1, 2, 3]
    assert np.isclose(trace.path_length, 0.015)
    assert trace.direction_reversal_count == 1
    assert trace.gripper_events.tolist() == [0, 1, 0]
    assert trace.workspace_margin_valid.all()
    assert trace.kinematic_map_valid is True


def test_invalid_steps_are_masked_not_invented() -> None:
    values = np.ones((2, 7), dtype=np.float32)
    trace = action_to_motion_trace(
        _token(values, step_mask=np.array([True, False])),
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
    )
    assert trace.valid_mask.tolist() == [True, False]
    assert trace.pose_valid_mask.tolist() == [True, False]
    assert trace.motion_dimension_mask[0].all()
    assert not trace.motion_dimension_mask[1].any()
    np.testing.assert_allclose(
        trace.integrated_ee_pose_rel[1], trace.integrated_ee_pose_rel[0],
    )


def test_missing_physical_semantics_produces_invalid_trace() -> None:
    token = CanonicalActionToken.from_array(
        np.ones((2, 3), dtype=np.float32),
        semantics=("joint_0", "joint_1", "joint_2"),
        control_hz=20.0,
        coordinate_frame="robot.joint.relative",
        policy_id="joint.policy",
    )
    trace = action_to_motion_trace(token)
    assert trace.kinematic_map_valid is False
    assert not trace.valid_mask.any()
    assert not trace.motion_dimension_mask.any()
    np.testing.assert_array_equal(trace.ee_delta, np.zeros((2, 6), dtype=np.float32))


def test_projection_is_only_valid_with_positive_depth() -> None:
    token = _token(np.array([
        [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    projection = np.array([
        [100.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    trace = action_to_motion_trace(
        token,
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
        projection_matrices={"agentview": projection},
    )
    assert trace.camera_roles == ("agentview",)
    assert trace.camera_projection_valid[0, 0]
    np.testing.assert_allclose(trace.projected_uv[0, 0], [10.0, 0.0])


def test_chunk_consistency_compares_previous_last_to_current_first() -> None:
    previous = _token(np.array([
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    current = _token(np.array([
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    trace = action_to_motion_trace(
        current,
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
        previous_action=previous,
    )
    assert trace.chunk_to_prev_consistency_valid
    assert np.isclose(trace.chunk_to_prev_consistency, -1.0)


def test_eef_frame_translation_is_rotated_by_current_pose() -> None:
    token = _token(np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    initial = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4),
    ])
    trace = action_to_motion_trace(
        token,
        semantic_map=MotionSemanticMap(
            translation=LIBERO_MOTION_SEMANTIC_MAP.translation,
            rotation=LIBERO_MOTION_SEMANTIC_MAP.rotation,
            gripper=LIBERO_MOTION_SEMANTIC_MAP.gripper,
            translation_frame="eef",
            rotation_representation="axis_angle",
        ),
        initial_ee_pose=initial,
    )
    np.testing.assert_allclose(
        trace.integrated_ee_pose_rel[0, :3], [0.0, 1.0, 0.0], atol=1e-6,
    )


def test_roundtrip_and_motion_parity_are_cpu_only() -> None:
    adapter = LiberoPolicyAdapter(policy_id="pi05.libero", family="pi05")
    samples = [
        np.zeros((2, 7), dtype=np.float32),
        np.linspace(-0.5, 0.5, 21, dtype=np.float32).reshape(3, 7),
    ]
    roundtrip = audit_action_roundtrip(adapter, samples)
    tokens = [adapter.raw_to_canonical(sample) for sample in samples]
    motion = audit_motion_trace_conversion(
        tokens, semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
    )
    assert roundtrip.status == "PASS"
    assert motion.status == "PASS"


def test_empirical_resample_capability_can_override_declared_support() -> None:
    identical = [
        [np.zeros((2, 7)), np.zeros((2, 7))],
        [np.ones((2, 7)), np.ones((2, 7))],
    ]
    audit = audit_resample_capability(
        identical, minimum_distinct_fraction=0.1,
    )
    adapter = LiberoPolicyAdapter(
        policy_id="claimed.stochastic",
        family="claimed",
        supports_resample=True,
        stochastic_sampling=True,
    )
    report = build_capability_report(
        adapter.descriptor,
        resample_audit=audit,
        fallback_available=True,
    )
    assert audit.status == "FAIL"
    assert audit.details["recommended_capability_mask"] is True
    assert report["effective_mask"]["resample.source"] is False
    assert report["status"] == "REVIEW_REQUIRED"


def test_distinct_resample_groups_pass() -> None:
    groups = [
        [np.zeros((1, 7)), np.ones((1, 7))],
        [np.zeros((1, 7)), np.eye(1, 7)],
    ]
    result = audit_resample_capability(groups, minimum_distinct_fraction=1.0)
    assert result.status == "PASS"
    assert result.details["distinct_fraction"] == 1.0


def test_libero_axis_angle_scaling_and_base_frame_composition() -> None:
    # A normalized +1 z rotation is +0.5 rad, and translation is 0.05 m.
    token = _token(np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
    ], dtype=np.float32))
    initial = np.array([
        0.0, 0.0, 0.0,
        np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4),
    ])
    trace = action_to_motion_trace(
        token,
        semantic_map=LIBERO_MOTION_SEMANTIC_MAP,
        initial_ee_pose=initial,
    )
    np.testing.assert_allclose(trace.ee_delta[0, :3], [0.05, 0.0, 0.0])
    np.testing.assert_allclose(trace.ee_delta[0, 3:], [0.0, 0.0, 0.5])
    # Base-frame composition differs from local composition for this pose.
    expected = np.array([
        np.sin(np.pi / 4) * np.cos(0.25),
        np.sin(np.pi / 4) * np.sin(0.25),
        np.cos(np.pi / 4) * np.sin(0.25),
        np.cos(np.pi / 4) * np.cos(0.25),
    ])
    np.testing.assert_allclose(
        trace.integrated_ee_pose_rel[0, 3:], expected, atol=1e-6,
    )

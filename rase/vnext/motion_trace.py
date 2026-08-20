"""Policy-agnostic physical motion traces derived from canonical action chunks.

The conversion is intentionally conservative: an unavailable or unverified
physical quantity is masked rather than invented.  This module imports NumPy
only and is safe to use in CPU-only parity tests while a GPU rollout is active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import CanonicalActionToken


def _readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _normalise_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must have a finite non-zero norm")
    return value / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64)


def _euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert intrinsic xyz Euler increments to xyzw quaternion."""
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalise_quaternion(np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]))


def _axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a scaled axis-angle / rotation vector to xyzw quaternion."""
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("axis_angle must be a finite shape-[3] rotation vector")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    half = angle / 2.0
    xyz = vector * (math.sin(half) / angle)
    return _normalise_quaternion(np.append(xyz, math.cos(half)))


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalise_quaternion(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass(frozen=True)
class MotionSemanticMap:
    """Map canonical semantic names to a physical 6-DoF delta plus gripper."""

    translation: tuple[str, str, str] = ("delta_x", "delta_y", "delta_z")
    rotation: tuple[str, str, str] = (
        "delta_roll", "delta_pitch", "delta_yaw",
    )
    gripper: str | None = "gripper"
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    translation_frame: str = "base"
    rotation_representation: str = "euler_xyz"
    rotation_frame: str = "eef"

    def validate(self) -> None:
        names = (*self.translation, *self.rotation)
        if len(set(names)) != 6:
            raise ValueError("translation and rotation semantics must be unique")
        if self.gripper is not None and self.gripper in names:
            raise ValueError("gripper semantic must not overlap motion semantics")
        if not math.isfinite(self.translation_scale) or self.translation_scale <= 0:
            raise ValueError("translation_scale must be finite and positive")
        if not math.isfinite(self.rotation_scale) or self.rotation_scale <= 0:
            raise ValueError("rotation_scale must be finite and positive")
        if self.translation_frame not in {"base", "eef"}:
            raise ValueError("translation_frame must be 'base' or 'eef'")
        if self.rotation_representation not in {"euler_xyz", "axis_angle"}:
            raise ValueError(
                "rotation_representation must be 'euler_xyz' or 'axis_angle'"
            )
        if self.rotation_frame not in {"base", "eef"}:
            raise ValueError("rotation_frame must be 'base' or 'eef'")


@dataclass(frozen=True)
class CanonicalMotionTrace:
    """Masked physical trace; zero-filled values are never valid without masks."""

    ee_delta: np.ndarray
    motion_dimension_mask: np.ndarray
    integrated_ee_pose_rel: np.ndarray
    pose_valid_mask: np.ndarray
    gripper_state: np.ndarray
    gripper_valid_mask: np.ndarray
    gripper_events: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    derivative_valid_order: np.ndarray
    path_length: float
    direction_reversal_count: int
    chunk_to_prev_consistency: float
    chunk_to_prev_consistency_valid: bool
    projected_uv: np.ndarray
    camera_roles: tuple[str, ...]
    camera_projection_valid: np.ndarray
    workspace_margin: np.ndarray
    workspace_margin_valid: np.ndarray
    valid_mask: np.ndarray
    kinematic_map_valid: bool
    coordinate_frame: str
    policy_id: str

    def __post_init__(self) -> None:
        arrays = {
            "ee_delta": (self.ee_delta, np.float32),
            "motion_dimension_mask": (self.motion_dimension_mask, np.bool_),
            "integrated_ee_pose_rel": (self.integrated_ee_pose_rel, np.float32),
            "pose_valid_mask": (self.pose_valid_mask, np.bool_),
            "gripper_state": (self.gripper_state, np.float32),
            "gripper_valid_mask": (self.gripper_valid_mask, np.bool_),
            "gripper_events": (self.gripper_events, np.int8),
            "velocity": (self.velocity, np.float32),
            "acceleration": (self.acceleration, np.float32),
            "jerk": (self.jerk, np.float32),
            "derivative_valid_order": (self.derivative_valid_order, np.int8),
            "projected_uv": (self.projected_uv, np.float32),
            "camera_projection_valid": (self.camera_projection_valid, np.bool_),
            "workspace_margin": (self.workspace_margin, np.float32),
            "workspace_margin_valid": (self.workspace_margin_valid, np.bool_),
            "valid_mask": (self.valid_mask, np.bool_),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly(value, dtype))
        self.validate()

    @property
    def horizon(self) -> int:
        return int(self.ee_delta.shape[0])

    def validate(self) -> None:
        if self.ee_delta.ndim != 2 or self.ee_delta.shape[1] != 6:
            raise ValueError("ee_delta must have shape [H,6]")
        horizon = self.ee_delta.shape[0]
        expected = {
            "motion_dimension_mask": (horizon, 6),
            "integrated_ee_pose_rel": (horizon, 7),
            "pose_valid_mask": (horizon,),
            "gripper_state": (horizon,),
            "gripper_valid_mask": (horizon,),
            "gripper_events": (horizon,),
            "velocity": (horizon, 6),
            "acceleration": (horizon, 6),
            "jerk": (horizon, 6),
            "derivative_valid_order": (horizon,),
            "workspace_margin": (horizon,),
            "workspace_margin_valid": (horizon,),
            "valid_mask": (horizon,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        cameras = len(self.camera_roles)
        if self.projected_uv.shape != (cameras, horizon, 2):
            raise ValueError("projected_uv must have shape [V,H,2]")
        if self.camera_projection_valid.shape != (cameras, horizon):
            raise ValueError("camera_projection_valid must have shape [V,H]")
        if len(set(self.camera_roles)) != cameras:
            raise ValueError("camera roles must be unique")
        if not np.isfinite(self.ee_delta).all():
            raise ValueError("ee_delta must be finite")
        if not np.isfinite(self.integrated_ee_pose_rel).all():
            raise ValueError("integrated pose must be finite")
        for value in (self.velocity, self.acceleration, self.jerk):
            if not np.isfinite(value).all():
                raise ValueError("derivatives must be finite")
        if not math.isfinite(self.path_length) or self.path_length < 0:
            raise ValueError("path_length must be finite and non-negative")
        if self.direction_reversal_count < 0:
            raise ValueError("direction_reversal_count cannot be negative")
        if self.chunk_to_prev_consistency_valid:
            if not math.isfinite(self.chunk_to_prev_consistency):
                raise ValueError("valid chunk consistency must be finite")
            if not -1.000001 <= self.chunk_to_prev_consistency <= 1.000001:
                raise ValueError("chunk consistency must be a cosine value")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame is required")
        if not self.policy_id:
            raise ValueError("policy_id is required")

    def summary(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "path_length": self.path_length,
            "direction_reversal_count": self.direction_reversal_count,
            "chunk_to_prev_consistency": (
                self.chunk_to_prev_consistency
                if self.chunk_to_prev_consistency_valid else None
            ),
            "valid_step_fraction": float(np.mean(self.valid_mask)) if self.horizon else 0.0,
            "gripper_event_count": int(np.count_nonzero(self.gripper_events)),
            "kinematic_map_valid": self.kinematic_map_valid,
            "camera_roles": list(self.camera_roles),
            "coordinate_frame": self.coordinate_frame,
            "policy_id": self.policy_id,
        }


def _semantic_indices(
    token: CanonicalActionToken, mapping: MotionSemanticMap,
) -> tuple[list[int] | None, int | None]:
    index = {name: position for position, name in enumerate(token.semantics)}
    motion_names = (*mapping.translation, *mapping.rotation)
    motion = [index[name] for name in motion_names] if all(
        name in index for name in motion_names
    ) else None
    gripper = index.get(mapping.gripper) if mapping.gripper is not None else None
    return motion, gripper


def _derivatives(
    delta: np.ndarray, valid: np.ndarray, dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = delta.shape[0]
    velocity = np.zeros_like(delta, dtype=np.float64)
    acceleration = np.zeros_like(delta, dtype=np.float64)
    jerk = np.zeros_like(delta, dtype=np.float64)
    order = np.zeros(horizon, dtype=np.int8)
    for step in range(horizon):
        if valid[step]:
            velocity[step] = delta[step] / float(dt[step])
            order[step] = 1
        if step >= 1 and valid[step] and valid[step - 1]:
            interval = max(float((dt[step] + dt[step - 1]) / 2.0), 1e-12)
            acceleration[step] = (velocity[step] - velocity[step - 1]) / interval
            order[step] = 2
        if step >= 2 and order[step] >= 2 and order[step - 1] >= 2:
            interval = max(float((dt[step] + dt[step - 1]) / 2.0), 1e-12)
            jerk[step] = (acceleration[step] - acceleration[step - 1]) / interval
            order[step] = 3
    return velocity, acceleration, jerk, order


def _direction_reversals(translation: np.ndarray, valid: np.ndarray) -> int:
    count = 0
    previous: np.ndarray | None = None
    for vector, is_valid in zip(translation, valid):
        norm = float(np.linalg.norm(vector))
        if not is_valid or norm <= 1e-9:
            continue
        if previous is not None and float(np.dot(previous, vector)) < 0:
            count += 1
        previous = vector
    return count


def _cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0, False
    return float(np.dot(left, right) / (left_norm * right_norm)), True


def action_to_motion_trace(
    token: CanonicalActionToken,
    *,
    semantic_map: MotionSemanticMap | None = None,
    initial_ee_pose: Sequence[float] | None = None,
    previous_action: CanonicalActionToken | None = None,
    workspace_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    projection_matrices: Mapping[str, np.ndarray] | None = None,
    gripper_event_epsilon: float = 1e-4,
) -> CanonicalMotionTrace:
    """Convert a canonical chunk into a conservative physical motion trace.

    ``initial_ee_pose`` is xyz+xyzw.  When omitted, integration is relative to
    origin and identity rotation.  Projection matrices are 3x4 matrices in the
    same frame as the integrated position.
    """
    token.validate()
    mapping = semantic_map or MotionSemanticMap()
    mapping.validate()
    if gripper_event_epsilon < 0 or not math.isfinite(gripper_event_epsilon):
        raise ValueError("gripper_event_epsilon must be finite and non-negative")
    horizon = token.horizon
    motion_indices, gripper_index = _semantic_indices(token, mapping)
    ee_delta = np.zeros((horizon, 6), dtype=np.float64)
    motion_mask = np.zeros((horizon, 6), dtype=np.bool_)
    if motion_indices is not None:
        ee_delta[:, :3] = token.values[:, motion_indices[:3]] * mapping.translation_scale
        ee_delta[:, 3:] = token.values[:, motion_indices[3:]] * mapping.rotation_scale
        motion_mask = token.dimension_mask[:, motion_indices].copy()
        motion_mask &= token.step_mask[:, None]
    valid = token.step_mask & motion_mask.all(axis=1)
    kinematic_map_valid = motion_indices is not None and bool(
        np.all(motion_mask[token.step_mask])
    )

    if initial_ee_pose is None:
        initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        initial = np.asarray(initial_ee_pose, dtype=np.float64)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("initial_ee_pose must be finite xyz+xyzw shape [7]")
        initial = initial.copy()
        initial[3:] = _normalise_quaternion(initial[3:])

    pose = np.zeros((horizon, 7), dtype=np.float64)
    pose_valid = np.zeros(horizon, dtype=np.bool_)
    current_position = initial[:3].copy()
    current_quaternion = initial[3:].copy()
    for step in range(horizon):
        if valid[step]:
            translation = ee_delta[step, :3]
            if mapping.translation_frame == "eef":
                translation = _rotation_matrix(current_quaternion) @ translation
            current_position = current_position + translation
            if mapping.rotation_representation == "axis_angle":
                delta_quaternion = _axis_angle_to_quaternion(ee_delta[step, 3:])
            else:
                delta_quaternion = _euler_xyz_to_quaternion(ee_delta[step, 3:])
            # robosuite OSC_POSE applies its axis-angle error in the base frame
            # (R_goal = R_delta @ R_current).  Local/intrinsic deltas use the
            # opposite multiplication order.
            if mapping.rotation_frame == "base":
                composed = _quat_multiply(delta_quaternion, current_quaternion)
            else:
                composed = _quat_multiply(current_quaternion, delta_quaternion)
            current_quaternion = _normalise_quaternion(composed)
            pose_valid[step] = True
        pose[step, :3] = current_position
        pose[step, 3:] = current_quaternion

    gripper_state = np.zeros(horizon, dtype=np.float64)
    gripper_valid = np.zeros(horizon, dtype=np.bool_)
    gripper_events = np.zeros(horizon, dtype=np.int8)
    if gripper_index is not None:
        gripper_state = token.values[:, gripper_index].astype(np.float64, copy=True)
        gripper_valid = token.step_mask & token.dimension_mask[:, gripper_index]
        previous_value: float | None = None
        for step in range(horizon):
            if not gripper_valid[step]:
                continue
            value = float(gripper_state[step])
            if previous_value is not None:
                difference = value - previous_value
                if abs(difference) > gripper_event_epsilon:
                    gripper_events[step] = 1 if difference > 0 else -1
            previous_value = value

    velocity, acceleration, jerk, derivative_order = _derivatives(
        ee_delta, valid, token.delta_time_s,
    )
    path_length = float(np.sum(np.linalg.norm(ee_delta[valid, :3], axis=1)))
    reversals = _direction_reversals(ee_delta[:, :3], valid)

    consistency = 0.0
    consistency_valid = False
    if previous_action is not None:
        previous_action.validate()
        previous_indices, _ = _semantic_indices(previous_action, mapping)
        current_valid_steps = np.flatnonzero(valid)
        if previous_indices is not None and current_valid_steps.size:
            previous_mask = (
                previous_action.step_mask
                & previous_action.dimension_mask[:, previous_indices].all(axis=1)
            )
            previous_valid_steps = np.flatnonzero(previous_mask)
            if previous_valid_steps.size:
                previous_delta = previous_action.values[
                    previous_valid_steps[-1], previous_indices
                ].astype(np.float64)
                previous_delta[:3] *= mapping.translation_scale
                previous_delta[3:] *= mapping.rotation_scale
                consistency, consistency_valid = _cosine(
                    previous_delta, ee_delta[current_valid_steps[0]],
                )

    workspace_margin = np.zeros(horizon, dtype=np.float64)
    workspace_valid = np.zeros(horizon, dtype=np.bool_)
    if workspace_bounds is not None:
        lower = np.asarray(workspace_bounds[0], dtype=np.float64)
        upper = np.asarray(workspace_bounds[1], dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("workspace bounds must each have shape [3]")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("workspace bounds must be finite and ordered")
        for step in range(horizon):
            if pose_valid[step]:
                distances = np.concatenate((pose[step, :3] - lower, upper - pose[step, :3]))
                workspace_margin[step] = float(np.min(distances))
                workspace_valid[step] = True

    roles = tuple(sorted((projection_matrices or {}).keys()))
    projected = np.zeros((len(roles), horizon, 2), dtype=np.float64)
    projection_valid = np.zeros((len(roles), horizon), dtype=np.bool_)
    for camera_index, role in enumerate(roles):
        matrix = np.asarray(projection_matrices[role], dtype=np.float64)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"projection matrix for {role!r} must be finite [3,4]")
        for step in range(horizon):
            if not pose_valid[step]:
                continue
            homogeneous = np.append(pose[step, :3], 1.0)
            image = matrix @ homogeneous
            if image[2] > 1e-9 and np.isfinite(image).all():
                projected[camera_index, step] = image[:2] / image[2]
                projection_valid[camera_index, step] = True

    return CanonicalMotionTrace(
        ee_delta=ee_delta,
        motion_dimension_mask=motion_mask,
        integrated_ee_pose_rel=pose,
        pose_valid_mask=pose_valid,
        gripper_state=gripper_state,
        gripper_valid_mask=gripper_valid,
        gripper_events=gripper_events,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        derivative_valid_order=derivative_order,
        path_length=path_length,
        direction_reversal_count=reversals,
        chunk_to_prev_consistency=consistency,
        chunk_to_prev_consistency_valid=consistency_valid,
        projected_uv=projected,
        camera_roles=roles,
        camera_projection_valid=projection_valid,
        workspace_margin=workspace_margin,
        workspace_margin_valid=workspace_valid,
        valid_mask=valid,
        kinematic_map_valid=kinematic_map_valid,
        coordinate_frame=token.coordinate_frame,
        policy_id=token.policy_id,
    )

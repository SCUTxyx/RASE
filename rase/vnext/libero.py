"""Minimal lossless LIBERO implementation of the vNext canonical interfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .schema import (
    CanonicalActionToken,
    CanonicalObservation,
    CanonicalRobotSpec,
    CorrectionProfile,
    PolicyDescriptor,
    SeedLedger,
)
from .motion_trace import MotionSemanticMap


LIBERO_ACTION_SEMANTICS = (
    "delta_x", "delta_y", "delta_z",
    "delta_axisangle_x", "delta_axisangle_y", "delta_axisangle_z",
    "gripper_close",
)
LIBERO_PROPRIO_SEMANTICS = (
    "eef_x", "eef_y", "eef_z", "eef_axisangle_x", "eef_axisangle_y",
    "eef_axisangle_z", "gripper_left", "gripper_right",
)
LIBERO_ROBOT_SPEC = CanonicalRobotSpec(
    spec_id="libero.panda.relative-7d",
    embodiment_id="franka.panda",
    action_semantics=LIBERO_ACTION_SEMANTICS,
    proprio_semantics=LIBERO_PROPRIO_SEMANTICS,
    control_hz=10.0,
    coordinate_frame="robot.base.relative",
    rotation_representation="incremental.scaled_axis_angle",
    gripper_range=(-1.0, 1.0),
    camera_roles=("agentview", "wrist"),
)
LIBERO_ROBOT_SPEC.validate()

# Verified against the installed robosuite OSC_POSE controller: normalized
# [-1, 1] commands map to +/-0.05 m translation and +/-0.5 rad rotation
# vectors.  set_goal_orientation left-multiplies the rotation error, so both
# translation and rotation deltas are expressed in the robot base frame.
LIBERO_MOTION_SEMANTIC_MAP = MotionSemanticMap(
    translation=("delta_x", "delta_y", "delta_z"),
    rotation=(
        "delta_axisangle_x", "delta_axisangle_y", "delta_axisangle_z",
    ),
    gripper="gripper_close",
    translation_scale=0.05,
    rotation_scale=0.5,
    translation_frame="base",
    rotation_representation="axis_angle",
    rotation_frame="base",
)
LIBERO_MOTION_SEMANTIC_MAP.validate()


def _unbatch_image(value: Any) -> np.ndarray:
    image = np.asarray(value, dtype=np.uint8)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC or 1xHWC RGB image, got {image.shape}")
    return image


def _unbatch_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},), got {array.shape}")
    return array


def _quat_xyzw_to_axis_angle(value: Any) -> np.ndarray:
    """Match LeRobot LiberoProcessorStep._quat2axisangle in NumPy."""
    quaternion = _unbatch_vector(value, size=4, name="eef quaternion").astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("eef quaternion must have non-zero norm")
    quaternion /= norm
    # q and -q represent the same pose; select the shortest [0, pi] rotation.
    if quaternion[3] < 0:
        quaternion *= -1
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return np.asarray(quaternion[:3] / denominator * angle, dtype=np.float32)


class LiberoPolicyAdapter:
    """Lossless raw-action codec plus optional policy proposal callback."""

    def __init__(
        self,
        *,
        policy_id: str,
        family: str,
        proposer: Callable[[CanonicalObservation], np.ndarray] | None = None,
        supports_requery: bool = True,
        supports_resample: bool = False,
        supports_short_chunk: bool = True,
        stochastic_sampling: bool = False,
    ) -> None:
        self._descriptor = PolicyDescriptor(
            policy_id=policy_id,
            family=family,
            action_semantics=LIBERO_ACTION_SEMANTICS,
            supports_requery=supports_requery,
            supports_resample=supports_resample,
            supports_short_chunk=supports_short_chunk,
            stochastic_sampling=stochastic_sampling,
        )
        self._descriptor.validate()
        self._proposer = proposer

    @property
    def descriptor(self) -> PolicyDescriptor:
        return self._descriptor

    def raw_to_canonical(self, value: np.ndarray) -> CanonicalActionToken:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(LIBERO_ACTION_SEMANTICS):
            raise ValueError(f"LIBERO action must have shape [H,7], got {array.shape}")
        return CanonicalActionToken.from_array(
            array,
            semantics=LIBERO_ACTION_SEMANTICS,
            control_hz=LIBERO_ROBOT_SPEC.control_hz,
            coordinate_frame=LIBERO_ROBOT_SPEC.coordinate_frame,
            policy_id=self.descriptor.policy_id,
        )

    def canonical_to_raw(self, token: CanonicalActionToken) -> np.ndarray:
        token.validate()
        if token.semantics != LIBERO_ACTION_SEMANTICS:
            raise ValueError("canonical token semantics do not match LIBERO")
        if token.coordinate_frame != LIBERO_ROBOT_SPEC.coordinate_frame:
            raise ValueError("canonical token coordinate frame does not match LIBERO")
        if not np.all(token.dimension_mask[token.step_mask]):
            raise ValueError("LIBERO execution requires all seven dimensions on valid steps")
        return np.asarray(token.values[token.step_mask], dtype=np.float32).copy()

    def propose(self, observation: CanonicalObservation) -> CanonicalActionToken:
        if self._proposer is None:
            raise RuntimeError("LiberoPolicyAdapter has no bound proposer")
        return self.raw_to_canonical(self._proposer(observation))


class LiberoCorrectionOperator:
    """Minimal LIBERO correction wrapper with an explicit semantic profile."""

    def __init__(
        self,
        *,
        profile: CorrectionProfile,
        proposer: Callable[[CanonicalObservation, SeedLedger], np.ndarray | None],
        policy_adapter: LiberoPolicyAdapter,
    ) -> None:
        profile.validate()
        self._profile = profile
        self._proposer = proposer
        self._policy_adapter = policy_adapter

    @property
    def profile(self) -> CorrectionProfile:
        return self._profile

    def propose(
        self, observation: CanonicalObservation, *, seed_ledger: SeedLedger
    ) -> CanonicalActionToken | None:
        seed_ledger.validate()
        raw = self._proposer(observation, seed_ledger)
        return None if raw is None else self._policy_adapter.raw_to_canonical(raw)


class LiberoBenchmarkAdapter:
    """Thin environment wrapper; imports no LIBERO/GPU modules at import time."""

    def __init__(self, *, vector_env: Any, forkable: Any) -> None:
        self.vector_env = vector_env
        self.forkable = forkable

    @property
    def robot_spec(self) -> CanonicalRobotSpec:
        return LIBERO_ROBOT_SPEC

    def reset(self) -> Mapping[str, Any]:
        value = self.vector_env.reset()
        return value[0] if isinstance(value, tuple) else value

    def observation_to_canonical(
        self, observation: Mapping[str, Any], *, task_text: str, timestamp_s: float
    ) -> CanonicalObservation:
        pixels = observation.get("pixels")
        if not isinstance(pixels, Mapping):
            raise ValueError("LIBERO observation is missing pixels")
        image_keys = {"image": "agentview", "image2": "wrist"}
        images = {
            image_keys.get(str(name), str(name)): _unbatch_image(value)
            for name, value in pixels.items()
            if str(name) in image_keys
        }
        proprio_value = observation.get("proprio", observation.get("observation.state"))
        if proprio_value is not None:
            proprio = _unbatch_vector(
                proprio_value, size=LIBERO_ROBOT_SPEC.proprio_dim, name="LIBERO proprio",
            )
        else:
            robot_state = observation.get("robot_state")
            if not isinstance(robot_state, Mapping):
                raise ValueError(
                    "LIBERO canonical observation requires explicit proprio or robot_state"
                )
            eef = robot_state.get("eef")
            gripper = robot_state.get("gripper")
            if not isinstance(eef, Mapping) or not isinstance(gripper, Mapping):
                raise ValueError("LIBERO robot_state requires eef and gripper mappings")
            position = _unbatch_vector(eef.get("pos"), size=3, name="eef position")
            axis_angle = _quat_xyzw_to_axis_angle(eef.get("quat"))
            gripper_qpos = _unbatch_vector(
                gripper.get("qpos"), size=2, name="gripper qpos",
            )
            proprio = np.concatenate((position, axis_angle, gripper_qpos)).astype(np.float32)
        return CanonicalObservation(
            images=images,
            proprio=proprio,
            proprio_semantics=LIBERO_PROPRIO_SEMANTICS,
            proprio_mask=np.ones_like(proprio, dtype=np.bool_),
            task_text=task_text,
            timestamp_s=timestamp_s,
        )

    @staticmethod
    def success(info: Mapping[str, Any]) -> bool:
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return bool(np.asarray(info[key]).reshape(-1)[0])
        return False

    def snapshot(self) -> Any:
        return self.forkable.snapshot()

    def restore(self, snapshot: Any) -> None:
        self.forkable.restore(snapshot)

    def execute(self, action: CanonicalActionToken) -> list[tuple[Any, ...]]:
        raw = LiberoPolicyAdapter(
            policy_id=action.policy_id, family="canonical.transport",
        ).canonical_to_raw(action)
        transitions = []
        for step in raw:
            transition = self.vector_env.step(step[None, :])
            transitions.append(transition)
            if len(transition) >= 5:
                terminated = bool(np.asarray(transition[2]).reshape(-1)[0])
                truncated = bool(np.asarray(transition[3]).reshape(-1)[0])
                if terminated or truncated:
                    break
            elif len(transition) == 4:
                done = bool(np.asarray(transition[2]).reshape(-1)[0])
                if done:
                    break
        return transitions

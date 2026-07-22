"""Continuation policies that query a remote OFT oracle over ZeroMQ."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from rase.collect.pool_candidates import raw_observations_from_control_env
from rase.oracle.client import OracleClient


def quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite-compatible quaternion (x,y,z,w) → axis-angle."""
    import math

    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def raw_libero_to_oracle_arrays(
    control_env: Any,
    *,
    force_update: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build single-env wire arrays from a live ControlEnv."""
    obs = raw_observations_from_control_env(control_env, force_update=force_update)
    agentview = np.asarray(obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if agentview.ndim == 4:
        agentview = agentview[0]
    if wrist.ndim == 4:
        wrist = wrist[0]
    # Official path flips inside the adapter unless images_already_flipped.
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if grip.size == 1:
        grip = np.array([grip[0], grip[0]], dtype=np.float32)
    state = np.concatenate(
        [pos, quat_xyzw_to_axisangle(quat), grip[:2]]
    )
    return agentview, wrist, state.astype(np.float32)


class OracleChunkContinuation:
    """Open-loop OFT chunk continuation with per-env action queue."""

    def __init__(
        self,
        client: OracleClient,
        *,
        instruction: str,
        env_id: int = 0,
        control_env: Any | None = None,
    ) -> None:
        self.client = client
        self.control_env = control_env
        self.instruction = instruction
        self.env_id = int(env_id)
        self._queue: deque[np.ndarray] = deque()

    def bind_control_env(self, control_env: Any) -> None:
        self.control_env = control_env

    def reset(self) -> None:
        self._queue.clear()

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray:
        del observation
        if self.control_env is None:
            raise RuntimeError("OracleChunkContinuation has no bound control_env")
        if not self._queue:
            agentview, wrist, proprio = raw_libero_to_oracle_arrays(self.control_env)
            outputs = self.client.predict(
                {
                    "agentview": agentview[None, ...],
                    "wrist": wrist[None, ...],
                    "proprio": proprio[None, ...],
                },
                payload={
                    "instructions": [task or self.instruction],
                    "return_mode": "chunk",
                    "proprio_format": "policy_state",
                    "images_already_flipped": False,
                    "env_id": [self.env_id],
                },
            )
            actions = np.asarray(outputs["actions"], dtype=np.float32)
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 7:
                raise ValueError(f"unexpected oracle actions shape {actions.shape}")
            for step in actions[0]:
                self._queue.append(step)
        return self._queue.popleft()

"""Oracle (OFT) policy via ZeroMQ client for Route C pipeline.

Provides `make_oft_oracle_policy` as a simple factory that returns an OFTOraclePolicy
object wrapping the existing `OracleClient` with a `.predict()` API.

Exposes `.predict_from_env(control_env, instruction)` which extracts images
from the raw LIBERO env and queries the OFT server.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
_OVLA_OFT_SRC = "/root/autodl-tmp/src/openvla-oft"
if _OVLA_OFT_SRC not in sys.path:
    sys.path.insert(0, _OVLA_OFT_SRC)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class OFTOraclePolicy:
    """Single-step OFT oracle policy backed by a running ZeroMQ server.

    Client-side wrapper around `OracleClient` without chunk queuing.
    """

    def __init__(self, client, task_id: str | None = None):
        self._client = client
        self._task_id = task_id

    def predict_from_env(self, control_env, *, instruction: str | None = None,
                         return_mode: str = "first") -> np.ndarray:
        """Extract RGB images + proprio from the raw LIBERO env and predict.

        Args:
            control_env: Raw LIBERO Mujoco env object (e.g., handle.control_env)
            instruction: Task instruction string (optional, uses init if None)
            return_mode: 'first' for single action, 'chunk' for full chunk

        Returns:
            7-DoF action array (shape (7,) for first mode, (H,7) for chunk mode)
        """
        from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays

        task = instruction or self._task_id
        agentview, wrist, proprio = raw_libero_to_oracle_arrays(control_env)

        outputs = self._client.predict(
            {
                "agentview": agentview[None, ...],
                "wrist": wrist[None, ...],
                "proprio": proprio[None, ...],
            },
            payload={
                "instructions": [task or ""],
                "return_mode": return_mode,
                "proprio_format": "policy_state",
                "images_already_flipped": False,
                "env_id": [0],
            },
        )
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        if return_mode == "first" or actions.ndim <= 1:
            return actions.reshape(-1)[:7]
        return actions[0]

    def model_info(self) -> dict:
        try:
            return self._client.model_info()
        except Exception:
            return {}

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


def make_oft_oracle_policy(checkpoint_path: str, device: str = "cuda:0",
                           vllm_port: int = 5555) -> OFTOraclePolicy:
    """Factory: create OFTOraclePolicy connected to an already-running ZeroMQ OFT server.

    The OFT server must be started separately with:
        python -m rase.oracle.server --endpoint tcp://127.0.0.1:PORT --adapter ...

    Args:
        checkpoint_path: Path to OFT checkpoint dir (used for instruction resolution only)
        device: Ignored; OFT policy runs in its own process
        vllm_port: TCP port of the ZeroMQ OFT server

    Returns:
        OFTOraclePolicy wrapping an OracleClient
    """
    from rase.oracle.client import OracleClient

    endpoint = f"tcp://127.0.0.1:{vllm_port}"
    client = OracleClient(endpoint, timeout_ms=30000)
    return OFTOraclePolicy(client)

"""Shared RecoveryHistoryBuffer for Route C.

Maintains a rolling window of (proprio, student_action, progress,
executed_action) tuples.  Used identically by collector, trainer, and
evaluator to guarantee the same history representation.

Format (per timestep, 23 dimensions):
  [0:8]   proprio (8 dims, padded)
  [8:15]  student_action (7 dims)
  [15]    progress (scalar)
  [16:23] executed_action (7 dims)

The ``executed_action`` field during *training* is the student action
itself (there is no plugin to modify it).  During *deployment* it is the
mix of student + plugin residual.

Target leakage protection:
  When constructing the history tensor for training step *t*, only steps
  < t are included.  The current student_action / teacher_action /
  delta_target are never placed into the history window for that step.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class RecoveryHistoryBuffer:
    """Rolling-window history buffer for the recovery plugin.

    Parameters
    ----------
    window : int
        Maximum number of timesteps to retain (default 8).
    proprio_dim : int
        Padded proprio dimension (default 8).
    action_dim : int
        Action dimension (default 7).
    """

    def __init__(
        self,
        window: int = 8,
        proprio_dim: int = 8,
        action_dim: int = 7,
    ):
        if window < 1:
            raise ValueError("history window must be at least 1")
        self._window = window
        self._proprio_dim = proprio_dim
        self._action_dim = action_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        proprio: np.ndarray,
        student_action: np.ndarray,
        progress: float,
        executed_action: np.ndarray | None = None,
    ) -> None:
        """Record one timestep.

        ``executed_action`` defaults to ``student_action`` (training mode).
        In deployment mode pass the mixed student+plugin action.
        """
        p = self._pad_to(np.asarray(proprio, dtype=np.float32).flatten(),
                          self._proprio_dim)
        a_stu = self._pad_to(np.asarray(student_action, dtype=np.float32).flatten(),
                              self._action_dim)
        if executed_action is not None:
            a_exe = self._pad_to(np.asarray(executed_action, dtype=np.float32).flatten(),
                                  self._action_dim)
        else:
            a_exe = a_stu.copy()
        step_vec = np.concatenate([p, a_stu, [float(progress)], a_exe]).astype(
            np.float32
        )
        self._buffer.append(step_vec)

    def get_history_tensor(self) -> dict[str, np.ndarray]:
        """Return the current window as a padded tensor.

        Returns
        -------
        dict
            ``data``   – float32 array of shape ``(window, dim)``,
                         zero-padded on the left.
            ``mask``   – bool array of shape ``(window,)``: True where
                         a real step exists (False for padding).
            ``length`` – int, number of real steps currently in the buffer
                         (always ≤ window).
            ``dim``    – int, per-step feature dimension.
        """
        entries = list(self._buffer)
        n = len(entries)
        dim = self.per_step_dim
        data = np.zeros((self._window, dim), dtype=np.float32)
        mask = np.zeros(self._window, dtype=bool)

        if n > 0:
            # Right-align: most recent step at position window-1.
            offset = self._window - n
            for i, vec in enumerate(entries):
                data[offset + i] = vec
                mask[offset + i] = True

        return {"data": data, "mask": mask, "length": n, "dim": dim}

    def reset(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def per_step_dim(self) -> int:
        return self._proprio_dim + self._action_dim + 1 + self._action_dim

    @property
    def window(self) -> int:
        return self._window

    @property
    def length(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._window

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_list(self) -> list[list[float]]:
        """Return the buffer as a list of lists (JSON-serializable)."""
        return [vec.tolist() for vec in self._buffer]

    @classmethod
    def from_list(
        cls,
        data: list[list[float]],
        window: int = 8,
        proprio_dim: int = 8,
        action_dim: int = 7,
    ) -> RecoveryHistoryBuffer:
        buf = cls(window=window, proprio_dim=proprio_dim, action_dim=action_dim)
        for row in data:
            arr = np.asarray(row, dtype=np.float32).flatten()
            if len(arr) >= buf.per_step_dim:
                arr = arr[: buf.per_step_dim]
            else:
                padded = np.zeros(buf.per_step_dim, dtype=np.float32)
                padded[: len(arr)] = arr
                arr = padded
            buf._buffer.append(arr)
        return buf

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to(arr: np.ndarray, target_dim: int) -> np.ndarray:
        if len(arr) >= target_dim:
            return arr[:target_dim]
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded

    def __repr__(self) -> str:
        return (
            f"RecoveryHistoryBuffer(window={self._window}, "
            f"length={self.length}, dim={self.per_step_dim})"
        )

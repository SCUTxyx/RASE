"""Shared StagnationDetector for Route C recovery pipeline.

Replaces the duplicated inline stagnation logic in:
  - plugin_executor.py
  - eval_route_c_plugin.py
  - collect_route_c_demos.py
  - pilot_route_c_headroom.py
"""

from __future__ import annotations

import numpy as np


class StagnationDetector:
    """Windowed progress-variance stagnation detector.

    A rollout step is flagged as stagnant when the standard deviation of the
    last ``window`` progress values falls below ``eps`` AND the maximum progress
    in the window is above ``min_progress`` (to avoid firing when the robot has
    not moved at all).
    """

    def __init__(
        self,
        window: int = 20,
        eps: float = 1e-4,
        min_progress: float = 1e-8,
    ):
        if window < 2:
            raise ValueError("stagnation window must be at least 2")
        self.window = window
        self.eps = eps
        self.min_progress = min_progress
        self._values: list[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, progress: float) -> None:
        """Record a new progress value."""
        self._values.append(float(progress))

    def is_stagnant(self) -> bool:
        """Return True if the trailing window indicates stagnation."""
        if len(self._values) < self.window:
            return False
        window = self._values[-self.window:]
        if np.max(window) <= self.min_progress:
            return False
        return bool(np.std(window) < self.eps)

    def reset(self) -> None:
        """Clear all recorded progress values."""
        self._values.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def progress_values(self) -> list[float]:
        return list(self._values)

    @property
    def last_window(self) -> list[float] | None:
        if len(self._values) < self.window:
            return None
        return self._values[-self.window:]

    @property
    def n_steps(self) -> int:
        return len(self._values)

    @property
    def latest_progress(self) -> float:
        return self._values[-1] if self._values else 0.0

    # ------------------------------------------------------------------
    # Serialization (for event logging and reproducibility)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "window": self.window,
            "eps": self.eps,
            "min_progress": self.min_progress,
            "values": list(self._values),
        }

    def load_state_dict(self, d: dict) -> None:
        self.window = int(d.get("window", self.window))
        self.eps = float(d.get("eps", self.eps))
        self.min_progress = float(d.get("min_progress", self.min_progress))
        self._values = [float(v) for v in d.get("values", [])]

    def __repr__(self) -> str:
        return (f"StagnationDetector(window={self.window}, eps={self.eps}, "
                f"n_steps={self.n_steps}, stagnant={self.is_stagnant()})")

#!/usr/bin/env python3
"""Route C recovery plugin executor.

Bounded takeover, safe action mixing, and handback logic.
Now uses shared StagnationDetector from rase.recovery.stagnation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from rase.recovery.residual_plugin import ResidualRecoveryPlugin
from rase.recovery.stagnation import StagnationDetector


class RecoveryPluginExecutor:
    """Wraps the plugin with takeover, mixing, and handback logic."""

    def __init__(
        self,
        plugin: ResidualRecoveryPlugin,
        smolvla_bundle: dict,
        *,
        history_window: int = 8,
        stagnation_window: int = 20,
        stagnation_eps: float = 1e-4,
        max_takeover_steps: int = 16,
        delta_clip: float = 0.5,
        action_rate_limit: float = 0.1,
        handback_consecutive_progress: int = 3,
        takeover_ramp: list[float] | None = None,
        force_off: bool = False,
    ):
        self.plugin = plugin
        self.smolvla_bundle = smolvla_bundle
        self.history_window = history_window
        self.max_takeover_steps = max_takeover_steps
        self.delta_clip = delta_clip
        self.action_rate_limit = action_rate_limit
        self.handback_consecutive_progress = handback_consecutive_progress
        self.takeover_ramp = takeover_ramp or [0.0, 0.3, 0.6, 1.0]
        self._force_off = force_off

        # Shared stagnation detector
        self._stagnation: StagnationDetector = StagnationDetector(
            window=stagnation_window,
            eps=stagnation_eps,
            min_progress=1e-8,
        )

        self._history: deque = deque(maxlen=history_window)
        self._taking_over = False
        self._takeover_t: int = 0
        self._consecutive_progress: int = 0
        self._last_progress: float = 0.0
        self._last_action: np.ndarray | None = None
        self._start_proprio: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties (compat with old API)
    # ------------------------------------------------------------------

    @property
    def stagnation_window(self) -> int:
        return self._stagnation.window

    @property
    def stagnation_eps(self) -> float:
        return self._stagnation.eps

    @property
    def _progress_values(self) -> list[float]:
        return self._stagnation.progress_values

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        self._history.clear()
        self._taking_over = False
        self._takeover_t = 0
        self._consecutive_progress = 0
        self._last_progress = 0.0
        self._last_action = None
        self._start_proprio = None
        self._stagnation.reset()

    def _g_mix(self, t: int) -> float:
        ramp = self.takeover_ramp
        if t < len(ramp):
            return ramp[t]
        return 1.0

    def should_takeover(self, progress: float) -> bool:
        if self._force_off:
            return False
        self._stagnation.update(progress)
        if self._stagnation.is_stagnant() and not self._taking_over:
            self._taking_over = True
            self._takeover_t = 0
            self._consecutive_progress = 0
            return True
        return False

    def check_takeover_after_update(self) -> bool:
        """Check stagnation AFTER an external update (e.g. from step()).
        Does NOT call _stagnation.update() itself — assumes caller already did."""
        if self._force_off:
            return False
        if self._stagnation.is_stagnant() and not self._taking_over:
            self._taking_over = True
            self._takeover_t = 0
            self._consecutive_progress = 0
            return True
        return False

    def step(self, obs: Any, student_action: np.ndarray, progress: float,
             obs_features: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        info = {"phase": "student", "takeover": False, "mixed": False,
                "delta": np.zeros(student_action.shape)}
        self._stagnation.update(progress)

        if self._taking_over and self._takeover_t < self.max_takeover_steps:
            if self._consecutive_progress >= self.handback_consecutive_progress:
                self._taking_over = False
                info["phase"] = "handback"
                return student_action, info

            history_arr = np.zeros((self.history_window, 8 + 7 + 1 + 7), dtype=np.float32)
            for i, h in enumerate(self._history):
                p = np.asarray(h.get("proprio", np.zeros(8)), dtype=np.float32).flatten()
                a = np.asarray(h.get("student_action", np.zeros(7)), dtype=np.float32).flatten()
                # Pad to expected dimensions
                p_pad = np.zeros(8, dtype=np.float32); p_pad[:min(len(p), 8)] = p[:8]
                a_pad = np.zeros(7, dtype=np.float32); a_pad[:min(len(a), 7)] = a[:7]
                history_arr[i] = np.concatenate([p_pad, a_pad,
                                                  [float(h.get("progress", 0.0))],
                                                  a_pad])

            # Use provided obs_features, or fall back to zeros (backward compat)
            if obs_features is not None and obs_features.size > 0:
                obs_feat = np.asarray(obs_features, dtype=np.float32).flatten()
                if obs_feat.size < self.plugin.obs_feature_dim:
                    padded = np.zeros(self.plugin.obs_feature_dim, dtype=np.float32)
                    padded[:obs_feat.size] = obs_feat
                    obs_feat = padded
                else:
                    obs_feat = obs_feat[:self.plugin.obs_feature_dim]
            else:
                obs_feat = np.zeros(self.plugin.obs_feature_dim, dtype=np.float32)

            delta = self.plugin.predict_delta(history_arr, obs_feat, student_action)
            g = self._g_mix(self._takeover_t)
            mixed_action = np.clip(student_action + g * delta, -1.0, 1.0)
            if self._last_action is not None:
                mixed_action = np.clip(mixed_action,
                                       self._last_action - self.action_rate_limit,
                                       self._last_action + self.action_rate_limit)
            self._takeover_t += 1
            self._last_action = mixed_action
            info["phase"] = "plugin_takeover"
            info["takeover"] = True
            info["mixed"] = True
            info["delta"] = delta
            info["g_mix"] = g
            if progress > self._last_progress + self._stagnation.eps:
                self._consecutive_progress += 1
            else:
                self._consecutive_progress = 0
            self._last_progress = progress
            return mixed_action, info

        if self._taking_over and self._takeover_t >= self.max_takeover_steps:
            self._taking_over = False
            info["phase"] = "max_steps_handback"

        return student_action, info

    def record_history(self, proprio: np.ndarray, student_action: np.ndarray,
                       progress: float, obs_feat: np.ndarray):
        p = np.asarray(proprio, dtype=np.float32).flatten()
        a = np.asarray(student_action, dtype=np.float32).flatten()
        # Pad to expected dimensions
        p_pad = np.zeros(8, dtype=np.float32)
        a_pad = np.zeros(7, dtype=np.float32)
        p_pad[:min(len(p), 8)] = p[:8]
        a_pad[:min(len(a), 7)] = a[:7]
        self._history.append({
            "proprio": p_pad,
            "student_action": a_pad,
            "progress": float(progress),
            "obs_feat": obs_feat.astype(np.float32).flatten()[:7],
        })

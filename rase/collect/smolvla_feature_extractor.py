"""Frozen SmolVLA feature extractor for Route C plugin conditioning.

Extracts a 128-D latent vector from the SmolVLA VLAFlowMatching model's
expert hidden state (before the action_out_proj layer).

Feature levels:
  F0: zeros only (baseline, current behavior)
  F1: proprio + student_action + stagnation_stats
  F2: frozen SmolVLA latent (128-D) + proprio + action + stats

The extractor uses a forward hook on model.action_out_proj to capture the
expert hidden states without modifying the LeRobot source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ── Feature dimension constants ──────────────────────────────────────

SMOLVLA_LATENT_DIM = 128     # projected from expert hidden (864) to 128
PROPRIO_DIM = 7              # eef pos (3) + eef quat (4) or similar
ACTION_DIM = 7               # translation (3) + rotation (3) + gripper (1)
STAGNATION_STATS_DIM = 2     # stagnation_length, normalized_progress_delta

F2_FEATURE_DIM = SMOLVLA_LATENT_DIM + PROPRIO_DIM + ACTION_DIM + STAGNATION_STATS_DIM  # 144


# ── Feature level enum ──────────────────────────────────────────────

FEATURE_LEVELS = {"F0", "F1", "F2"}


# ── SmolVLA latent extractor ─────────────────────────────────────────

class SmolVLAFeatureExtractor:
    """Extract frozen SmolVLA visual-language embedding via forward hook.

    Captures the input tensor to ``model.action_out_proj`` (expert hidden
    state of shape [B, chunk_size, expert_hidden]) and projects it to a
    fixed-size latent vector.

    The extractor registers a temporary hook during each inference call
    and removes it afterward, so it has no persistent side effects on
    the policy.
    """

    def __init__(
        self,
        bundle: dict[str, Any],
        *,
        latent_dim: int = SMOLVLA_LATENT_DIM,
        pooling: str = "mean",
        device: str = "cuda",
    ):
        policy = bundle["policy"]
        model = policy.model  # VLAFlowMatching

        # Determine expert hidden dim from action_out_proj
        action_out = model.action_out_proj
        if hasattr(action_out, "in_features"):
            expert_hidden_dim = action_out.in_features
        else:
            # Fallback: check first linear in the expert
            expert_hidden_dim = 864  # default for expert_width_multiplier=0.75

        self._policy = policy
        self._model = model
        self._expert_hidden_dim = expert_hidden_dim
        self._pooling = pooling
        self._device = device
        self._bundle = bundle  # retain full bundle for extract() preprocessors

        # Projection: expert_hidden_dim → latent_dim (frozen)
        self._proj = nn.Linear(expert_hidden_dim, latent_dim, bias=False)
        self._proj.to(device)
        self._proj.requires_grad_(False)  # frozen

        self._captured: torch.Tensor | None = None
        self._hook_handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, obs: dict[str, Any], *, seed: int | None = None) -> np.ndarray:
        """Extract frozen SmolVLA latent for an observation dict.

        Returns a 1-D numpy array of shape (latent_dim,).

        Args:
            obs: observation dict from the environment
            seed: if provided, seed torch + numpy RNG before forward
                  for deterministic extraction
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reset policy action queue to force a fresh forward pass
        self._policy.reset()

        # Register hook
        action_out = self._model.action_out_proj
        self._hook_handle = action_out.register_forward_pre_hook(self._hook_fn)
        self._captured = None

        try:
            # Run policy forward (we need select_action to trigger the
            # full forward pass through the flow-matching decoder).
            from lerobot.envs.utils import preprocess_observation
            from lerobot.utils.constants import ACTION

            policy_observation = preprocess_observation(
                {key: value for key, value in obs.items() if key != "task"}
            )
            policy_observation["task"] = [obs.get("task", "")]

            bundle = self._get_bundle()
            obs_tensor = bundle["env_preprocessor"](policy_observation)
            processed = bundle["preprocessor"](obs_tensor)

            # Use eval mode and no_grad to avoid side effects
            self._policy.eval()
            with torch.no_grad():
                _ = self._policy.select_action(processed)

        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

        if self._captured is None:
            return np.zeros(self._proj.out_features, dtype=np.float32)

        # Pool and project
        return self._project(self._captured)

    def extract_from_processed(self, processed: dict[str, Any]) -> np.ndarray:
        """Extract latent from already-processed observation dict.

        This is the preferred path during rollout: the observation has
        already been through preprocessor → env_preprocessor → preprocessor,
        so we don't re-process it.
        """
        action_out = self._model.action_out_proj
        self._hook_handle = action_out.register_forward_pre_hook(self._hook_fn)
        self._captured = None

        try:
            self._policy.eval()
            with torch.no_grad():
                _ = self._policy.select_action(processed)
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

        if self._captured is None:
            return np.zeros(self._proj.out_features, dtype=np.float32)

        return self._project(self._captured)

    # ------------------------------------------------------------------
    # Capture-during-select API (no extra forward pass, no reset)
    # ------------------------------------------------------------------

    def start_capture(self):
        """Register hook on action_out_proj so the next select_action
        call captures the expert hidden state. Call finish_capture() after
        the select_action to get the projected latent."""
        action_out = self._model.action_out_proj
        self._hook_handle = action_out.register_forward_pre_hook(self._hook_fn)
        self._captured = None

    def finish_capture(self) -> np.ndarray:
        """Remove hook and return the projected latent captured during the
        most recent select_action call."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        if self._captured is None:
            return np.zeros(self._proj.out_features, dtype=np.float32)

        return self._project(self._captured)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hook_fn(self, module, args):
        # args[0] should be the expert hidden state [B, chunk, dim]
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            self._captured = args[0].detach()

    def _project(self, hidden: torch.Tensor) -> np.ndarray:
        """Pool over chunk dimension and project to latent_dim."""
        # hidden: [B, chunk, expert_hidden_dim] or [B, expert_hidden_dim]
        if hidden.dim() == 3:
            if self._pooling == "mean":
                pooled = hidden.mean(dim=1)
            elif self._pooling == "last":
                pooled = hidden[:, -1, :]
            else:
                pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        projected = self._proj(pooled.to(self._device))
        return projected.squeeze(0).cpu().numpy().astype(np.float32)

    def _get_bundle(self) -> dict[str, Any]:
        """Return the stored bundle (set during init)."""
        return self._bundle


# ── Standalone feature construction (without SmolVLA) ────────────────

def build_proprio_features(
    proprio: np.ndarray,
    *,
    proprio_dim: int = PROPRIO_DIM,
) -> np.ndarray:
    """Normalize proprioceptive data to a fixed-size feature vector."""
    arr = np.asarray(proprio, dtype=np.float32).flatten()
    if len(arr) < proprio_dim:
        padded = np.zeros(proprio_dim, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded
    return arr[:proprio_dim]


def build_stagnation_features(
    stagnation_length: int,
    progress_delta: float,
    *,
    max_stagnation: int = 20,
) -> np.ndarray:
    """Build stagnation statistics features:
    - stagnation_length / max_stagnation (normalized)
    - progress_delta (clipped to [-1, 1])
    """
    norm_len = min(float(stagnation_length) / max(max_stagnation, 1), 1.0)
    norm_delta = float(np.clip(progress_delta, -1.0, 1.0))
    return np.array([norm_len, norm_delta], dtype=np.float32)


def build_feature_vector(
    smolvla_latent: np.ndarray | None,
    proprio: np.ndarray,
    student_action: np.ndarray,
    stagnation_length: int = 0,
    progress_delta: float = 0.0,
    *,
    feature_level: str = "F2",
) -> np.ndarray:
    """Build the full observation feature vector for the plugin.

    Args:
        smolvla_latent: 128-D frozen SmolVLA embedding (or None for F0/F1)
        proprio: raw proprio data
        student_action: current SmolVLA action
        stagnation_length: consecutive stagnation steps
        progress_delta: change in progress
        feature_level: "F0", "F1", or "F2"

    Returns:
        1-D float32 numpy array of shape (144,) for F2, same size for F0/F1 but
        with relevant components zeroed out.
    """
    if feature_level == "F2" and smolvla_latent is not None:
        latent = np.asarray(smolvla_latent, dtype=np.float32).flatten()
        if len(latent) < SMOLVLA_LATENT_DIM:
            padded = np.zeros(SMOLVLA_LATENT_DIM, dtype=np.float32)
            padded[: len(latent)] = latent[:SMOLVLA_LATENT_DIM]
            latent = padded
        else:
            latent = latent[:SMOLVLA_LATENT_DIM]
    else:
        latent = np.zeros(SMOLVLA_LATENT_DIM, dtype=np.float32)

    prop = build_proprio_features(proprio)
    act = np.asarray(student_action, dtype=np.float32).flatten()[:ACTION_DIM]
    if len(act) < ACTION_DIM:
        padded = np.zeros(ACTION_DIM, dtype=np.float32)
        padded[: len(act)] = act
        act = padded

    stag = build_stagnation_features(stagnation_length, progress_delta)

    return np.concatenate([latent, prop, act, stag]).astype(np.float32)

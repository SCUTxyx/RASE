"""SmolVLA adapter implementing the NGC ``CandidatePolicy`` protocol.

Diversity comes from flow-matching initial noise. Protocol temperature scales
that noise (``temperature=1.0`` matches LeRobot's default ``sample_noise``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rase.backends.lerobot_libero_plus import _get_or_load_policy


def checkpoint_sha256(path: str | Path) -> str:
    """Immutable provenance hash of the frozen weight file."""
    target = Path(path).expanduser().resolve()
    weight = target / "model.safetensors" if target.is_dir() else target
    if not weight.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weight}")
    digest = hashlib.sha256()
    with weight.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flow_matching_noise(
    shape: tuple[int, ...],
    *,
    device: Any,
    temperature: float,
) -> Any:
    """Sample N(0, temperature^2) noise for SmolVLA flow matching."""
    import torch

    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    noise = torch.randn(shape, dtype=torch.float32, device=device)
    if temperature != 1.0:
        noise = noise * float(temperature)
    return noise


@dataclass
class SmolVLACandidatePolicy:
    """Duck-typed ``CandidatePolicy`` over a loaded LeRobot SmolVLA bundle."""

    policy: Any
    preprocessor: Any
    env_preprocessor: Any
    postprocessor: Any
    env_postprocessor: Any
    chunk_length: int = 10
    checkpoint: str = ""
    revision: str | None = None

    def reset(self) -> None:
        self.policy.reset()

    def sample_chunk(self, observation: Mapping[str, Any], *, temperature: float) -> np.ndarray:
        """Return one env-space action chunk of shape ``[T, 7]``."""
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        if self.chunk_length < 1:
            raise ValueError("chunk_length must be positive")

        task = observation.get("task")
        if task is None:
            raise ValueError("observation must include task (language instruction)")
        gym_obs = {key: value for key, value in observation.items() if key != "task"}
        policy_observation = preprocess_observation(gym_obs)
        if isinstance(task, str):
            policy_observation["task"] = [task]
        else:
            policy_observation["task"] = list(task)

        env_observation = self.env_preprocessor(policy_observation)
        processed = self.preprocessor(env_observation)

        device = next(self.policy.parameters()).device
        chunk_size = int(self.policy.config.chunk_size)
        max_action_dim = int(self.policy.config.max_action_dim)
        noise = flow_matching_noise(
            (1, chunk_size, max_action_dim),
            device=device,
            temperature=temperature,
        )

        device_type = device.type
        amp_context = (
            torch.autocast(device_type=device_type)
            if bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        with torch.no_grad(), amp_context:
            actions = self.policy.predict_action_chunk(processed, noise=noise)

        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"unexpected action chunk shape {tuple(actions.shape)}")
        if actions.shape[1] < self.chunk_length:
            raise ValueError(
                f"policy chunk_size {actions.shape[1]} < requested T={self.chunk_length}"
            )

        steps: list[np.ndarray] = []
        for index in range(self.chunk_length):
            step = actions[:, index, :]
            step = self.postprocessor(step)
            transition = self.env_postprocessor({ACTION: step})
            array = transition[ACTION].detach().cpu().numpy()
            if array.ndim == 2:
                array = array[0]
            if array.shape[-1] != 7:
                raise ValueError(f"expected action dim 7, got {array.shape}")
            steps.append(np.asarray(array, dtype=np.float32))
        return np.stack(steps, axis=0)


def load_smolvla_candidate_policy(
    policy_path: str | Path,
    *,
    device: str = "cuda",
    num_steps: int = 10,
    n_action_steps: int = 10,
    chunk_length: int = 10,
    tokenizer_path: str | Path | None = None,
    observation_height: int = 360,
    observation_width: int = 360,
) -> SmolVLACandidatePolicy:
    """Load frozen SmolVLA processors/policy for candidate generation."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    path = Path(policy_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"policy path does not exist: {path}")
    tok = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else None
    # Feature geometry only; task id is unused for inference processors.
    env_cfg = LiberoEnvConfig(
        task="libero_spatial",
        task_ids=[0],
        obs_type="pixels_agent_pos",
        init_states=True,
        observation_height=observation_height,
        observation_width=observation_width,
    )
    bundle = _get_or_load_policy(
        path,
        device=device,
        num_steps=num_steps,
        n_action_steps=n_action_steps,
        env_cfg=env_cfg,
        tokenizer_path=tok,
    )
    policy = bundle["policy"]
    revision = getattr(getattr(policy, "config", None), "repo_id", None)
    return SmolVLACandidatePolicy(
        policy=policy,
        preprocessor=bundle["preprocessor"],
        env_preprocessor=bundle["env_preprocessor"],
        postprocessor=bundle["postprocessor"],
        env_postprocessor=bundle["env_postprocessor"],
        chunk_length=int(chunk_length),
        checkpoint=str(path),
        revision=str(revision) if revision else None,
    )

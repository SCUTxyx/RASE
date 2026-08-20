"""Continuation must pass temperature-scaled noise into select_action."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rase.collect.policy_step import select_env_action, select_env_action_with_native_chunk


class _FakePolicy:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(
            chunk_size=10, max_action_dim=32, use_amp=False, n_action_steps=10
        )

    def parameters(self):
        yield torch.zeros(1, device="cpu")

    def select_action(self, processed, noise=None):
        del processed
        self.calls.append(noise)
        return self.predict_action_chunk({}, noise=noise)[:, 0, :7]

    def predict_action_chunk(self, processed, noise=None):
        del processed, noise
        return torch.arange(10 * 32, dtype=torch.float32).reshape(1, 10, 32)


def test_select_env_action_scales_noise(monkeypatch):
    from lerobot.utils.constants import ACTION

    monkeypatch.setattr(
        "lerobot.envs.utils.preprocess_observation",
        lambda obs: dict(obs),
    )
    fake = _FakePolicy()
    bundle = {
        "policy": fake,
        "preprocessor": lambda x: x,
        "postprocessor": lambda x: x,
        "env_preprocessor": lambda x: x,
        "env_postprocessor": lambda d: d,
    }
    obs = {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}}
    out = select_env_action(bundle, obs, task="do it", temperature=0.5)
    assert ACTION in {"action", ACTION}  # sanity
    assert np.asarray(out).shape[-1] == 7
    assert len(fake.calls) == 1
    noise = fake.calls[0]
    assert noise is not None
    assert tuple(noise.shape) == (1, 10, 32)
    assert float(noise.std()) < 0.9


def test_select_env_action_without_temperature_omits_noise(monkeypatch):
    monkeypatch.setattr(
        "lerobot.envs.utils.preprocess_observation",
        lambda obs: dict(obs),
    )
    fake = _FakePolicy()
    bundle = {
        "policy": fake,
        "preprocessor": lambda x: x,
        "postprocessor": lambda x: x,
        "env_preprocessor": lambda x: x,
        "env_postprocessor": lambda d: d,
    }
    select_env_action(
        bundle,
        {"pixels": {"image": np.zeros((1, 8, 8, 3), dtype=np.uint8)}},
        task="do it",
    )
    assert fake.calls == [None]


def test_native_chunk_is_captured_during_executed_inference(monkeypatch):
    from lerobot.utils.constants import ACTION

    monkeypatch.setattr(
        "lerobot.envs.utils.preprocess_observation", lambda obs: dict(obs),
    )
    fake = _FakePolicy()
    bundle = {
        "policy": fake,
        "preprocessor": lambda x: x,
        "env_preprocessor": lambda x: x,
        "postprocessor": lambda x: x[..., :7],
        "env_postprocessor": lambda d: d,
    }
    first, chunk = select_env_action_with_native_chunk(
        bundle, {"observation.state": 0}, task="test", horizon=10,
    )
    assert chunk.shape == (10, 7)
    assert chunk.dtype.name == "float32"
    assert (first == chunk[0]).all()
    assert chunk[-1, -1] == 294.0

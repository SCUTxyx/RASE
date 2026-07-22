"""Continuation must pass temperature-scaled noise into select_action."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rase.collect.policy_step import select_env_action


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
        return torch.zeros(1, 7)


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

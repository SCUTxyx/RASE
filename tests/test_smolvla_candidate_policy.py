"""Unit tests for SmolVLA candidate helpers (no GPU / no LIBERO)."""

from __future__ import annotations

import numpy as np
import pytest

from rase.collect.candidates import generate_candidates
from rase.collect.pool_candidates import candidate_base_seed, diversity_summary
from rase.collect.smolvla_candidate_policy import flow_matching_noise


def test_flow_matching_noise_scales_with_temperature():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    unit = flow_matching_noise((8, 4, 7), device="cpu", temperature=1.0)
    torch.manual_seed(0)
    scaled = flow_matching_noise((8, 4, 7), device="cpu", temperature=0.5)
    assert unit.shape == (8, 4, 7)
    np.testing.assert_allclose(scaled.numpy(), unit.numpy() * 0.5, rtol=0, atol=0)


def test_flow_matching_noise_zero_temperature_is_deterministic():
    pytest.importorskip("torch")
    first = flow_matching_noise((2, 3, 7), device="cpu", temperature=0.0)
    second = flow_matching_noise((2, 3, 7), device="cpu", temperature=0.0)
    assert float(first.abs().max()) == 0.0
    assert float(second.abs().max()) == 0.0


def test_candidate_base_seed_stable():
    a = candidate_base_seed(10, "sp1_abcdef0123456789deadbeefcafebabe", 100)
    b = candidate_base_seed(10, "sp1_abcdef0123456789deadbeefcafebabe", 100)
    c = candidate_base_seed(11, "sp1_abcdef0123456789deadbeefcafebabe", 100)
    assert a == b
    assert a != c
    assert 0 <= a <= 2**32 - 8
    # Large mix must stay in NumPy's legacy seed range even after +7.
    huge = candidate_base_seed(2**31, "sp1_ffffffff00000000deadbeefcafebabe", 17072026)
    assert 0 <= huge + 7 <= 2**32 - 1


def test_diversity_summary_empty():
    summary = diversity_summary([])
    assert summary["n"] == 0
    assert summary["mean_endpoint_l2"] == 0.0


def test_raw_observations_from_control_env_uses_inner_task():
    from rase.collect.pool_candidates import raw_observations_from_control_env

    class _Task:
        def __init__(self):
            self.calls = []

        def _get_observations(self, force_update=False):
            self.calls.append(force_update)
            return {"agentview_image": "ok"}

    class _Wrapper:
        def __init__(self):
            self.env = _Task()

    wrapper = _Wrapper()
    obs = raw_observations_from_control_env(wrapper, force_update=True)
    assert obs["agentview_image"] == "ok"
    assert wrapper.env.calls == [True]


def test_batch_single_gym_observation_adds_batch_dim():
    from rase.collect.pool_candidates import batch_single_gym_observation

    obs = batch_single_gym_observation(
        {
            "pixels": {"image": np.zeros((64, 64, 3), dtype=np.uint8)},
            "robot_state": {
                "eef": {
                    "pos": np.zeros(3, dtype=np.float64),
                    "quat": np.array([0, 0, 0, 1], dtype=np.float64),
                    "mat": np.eye(3, dtype=np.float64),
                },
                "gripper": {"qpos": np.zeros(2, dtype=np.float64)},
            },
            "task": "pick up the bowl",
        }
    )
    assert obs["pixels"]["image"].shape == (1, 64, 64, 3)
    assert obs["robot_state"]["eef"]["quat"].shape == (1, 4)
    assert obs["robot_state"]["eef"]["mat"].shape == (1, 3, 3)
    assert obs["task"] == "pick up the bowl"


class _TempPolicy:
    revision = "unit"
    checkpoint = "fake"

    def reset(self):
        return None

    def sample_chunk(self, observation, *, temperature):
        del observation
        return np.random.normal(size=(10, 7)).astype(np.float32) * (0.1 + temperature)


def test_generate_candidates_chunk_length_ten():
    artifact = generate_candidates(
        _TempPolicy(),
        {"task": "demo"},
        temperature=0.7,
        base_seed=3,
        policy_hash="unit-hash",
    )
    assert artifact.actions.shape == (8, 10, 7)
    assert artifact.metadata.diversity.mean_pairwise_endpoint_l2 > 0

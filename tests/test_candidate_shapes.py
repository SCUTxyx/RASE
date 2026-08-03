import numpy as np
import pytest

from rase.collect.candidates import (
    K_DEFAULT,
    generate_candidates,
    load_artifact,
    make_artifact,
    save_artifact,
)


class FakePolicy:
    revision = "fake-v1"

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def sample_chunk(self, observation, *, temperature):
        return np.random.normal(size=(4, 7)).astype(np.float32) * temperature


def test_generate_fixed_shape_and_provenance(tmp_path):
    policy = FakePolicy()
    artifact = generate_candidates(
        policy, {}, base_seed=10, temperature=0.4, policy_hash="abc123"
    )
    assert artifact.actions.shape == (8, 4, 7)
    assert artifact.metadata.shape == (8, 4, 7)
    assert artifact.metadata.seeds == tuple(range(10, 18))
    assert artifact.metadata.policy_hash == "abc123"
    assert artifact.metadata.diversity.mean_pairwise_endpoint_l2 > 0
    assert policy.resets == K_DEFAULT

    path = tmp_path / "candidate.npz"
    save_artifact(path, artifact)
    with np.load(path, allow_pickle=False) as stored:
        assert set(
            [
                "actions",
                "seeds",
                "temperature",
                "policy_hash",
                "diversity",
                "format_version",
                "metadata",
            ]
        ) <= set(stored.files)
    restored = load_artifact(path)
    np.testing.assert_array_equal(restored.actions, artifact.actions)
    assert restored.metadata == artifact.metadata


def test_generate_protocol_k4_is_supported_and_roundtrips(tmp_path):
    policy = FakePolicy()
    artifact = generate_candidates(
        policy, {}, k=4, base_seed=20, temperature=0.7, policy_hash="k4"
    )
    assert artifact.actions.shape == (4, 4, 7)
    assert artifact.metadata.seeds == (20, 21, 22, 23)
    assert policy.resets == 4
    path = tmp_path / "candidate-k4.npz"
    save_artifact(path, artifact)
    restored = load_artifact(path)
    np.testing.assert_array_equal(restored.actions, artifact.actions)
    assert restored.metadata == artifact.metadata


@pytest.mark.parametrize(
    "shape",
    [
        (8, 4, 6),
        (8, 0, 7),
        (8, 7),
    ],
)
def test_rejects_non_protocol_shapes(shape):
    with pytest.raises(ValueError):
        make_artifact(
            np.zeros(shape, dtype=np.float32),
            seeds=range(8),
            temperature=0.7,
            policy_hash="hash",
        )


def test_rejects_seed_count_mismatch_and_duplicates():
    actions = np.zeros((4, 4, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="seed count"):
        make_artifact(actions, seeds=range(3), temperature=0.7, policy_hash="hash")
    with pytest.raises(ValueError, match="seeds must be unique"):
        make_artifact(actions, seeds=[1, 1, 2, 3], temperature=0.7, policy_hash="hash")


def test_generation_is_seed_reproducible():
    first = generate_candidates(FakePolicy(), {}, base_seed=42)
    second = generate_candidates(FakePolicy(), {}, base_seed=42)
    np.testing.assert_array_equal(first.actions, second.actions)

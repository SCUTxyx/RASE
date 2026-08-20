from __future__ import annotations

import numpy as np
import pytest

from rase.vnext.libero import (
    LIBERO_ACTION_SEMANTICS,
    LIBERO_MOTION_SEMANTIC_MAP,
    LIBERO_ROBOT_SPEC,
    LiberoBenchmarkAdapter,
    LiberoCorrectionOperator,
    LiberoPolicyAdapter,
)
from rase.vnext.schema import (
    CanonicalActionToken,
    CorrectionKind,
    CorrectionProfile,
    PolicyDescriptor,
    SeedLedger,
    operator_mask,
)


def test_libero_action_roundtrip_is_exact() -> None:
    raw = np.linspace(-0.9, 0.9, 21, dtype=np.float32).reshape(3, 7)
    adapter = LiberoPolicyAdapter(policy_id="pi05.libero", family="pi05")
    token = adapter.raw_to_canonical(raw)
    np.testing.assert_array_equal(adapter.canonical_to_raw(token), raw)
    assert token.semantics == LIBERO_ACTION_SEMANTICS
    np.testing.assert_allclose(token.delta_time_s, 0.1)
    assert token.coordinate_frame == "robot.base.relative"
    assert LIBERO_ROBOT_SPEC.rotation_representation == "incremental.scaled_axis_angle"
    assert LIBERO_MOTION_SEMANTIC_MAP.translation_scale == 0.05
    assert LIBERO_MOTION_SEMANTIC_MAP.rotation_scale == 0.5
    assert LIBERO_MOTION_SEMANTIC_MAP.translation_frame == "base"
    assert LIBERO_MOTION_SEMANTIC_MAP.rotation_frame == "base"
    assert LIBERO_ROBOT_SPEC.gripper_range == (-1.0, 1.0)


def test_variable_dimension_action_masks_are_not_hard_coded_to_seven() -> None:
    values = np.arange(20, dtype=np.float32).reshape(2, 10)
    token = CanonicalActionToken.from_array(
        values,
        semantics=tuple(f"joint_{index}" for index in range(10)),
        control_hz=20.0,
        coordinate_frame="robot.joint.relative",
        policy_id="future.policy",
        step_mask=np.array([True, False]),
    )
    assert token.action_dim == 10
    assert token.dimension_mask[0].all()
    assert not token.dimension_mask[1].any()


def test_libero_nested_robot_state_matches_lerobot_proprio_contract() -> None:
    adapter = LiberoBenchmarkAdapter(vector_env=_VectorEnv(), forkable=_Forkable())
    observation = adapter.observation_to_canonical(
        {
            "pixels": {"image": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            "robot_state": {
                "eef": {
                    "pos": np.array([[0.1, 0.2, 0.3]]),
                    "quat": np.array([[0.0, 0.0, np.sin(0.25), np.cos(0.25)]]),
                },
                "gripper": {"qpos": np.array([[0.01, 0.02]])},
            },
        },
        task_text="nested", timestamp_s=0.0,
    )
    np.testing.assert_allclose(
        observation.proprio,
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.5, 0.01, 0.02],
        atol=1e-6,
    )


def test_libero_rejects_non_libero_semantics() -> None:
    token = CanonicalActionToken.from_array(
        np.zeros((1, 7), dtype=np.float32),
        semantics=tuple(f"wrong_{index}" for index in range(7)),
        control_hz=10.0,
        coordinate_frame="robot.base.relative",
        policy_id="pi05.libero",
    )
    with pytest.raises(ValueError, match="semantics"):
        LiberoPolicyAdapter(policy_id="pi05.libero", family="pi05").canonical_to_raw(token)


def test_operator_mask_counts_resample_candidates_as_one_operator() -> None:
    profiles = (
        CorrectionProfile("continue.source", CorrectionKind.CONTINUE),
        CorrectionProfile("requery.source", CorrectionKind.REQUERY),
        CorrectionProfile(
            "resample.source", CorrectionKind.RESAMPLE,
            candidate_ids=("candidate.0", "candidate.1"),
        ),
        CorrectionProfile("fallback.persistent", CorrectionKind.FALLBACK),
        CorrectionProfile("abort.safe", CorrectionKind.ABORT),
    )
    policy = PolicyDescriptor(
        policy_id="pi0fast.libero", family="pi0fast",
        action_semantics=LIBERO_ACTION_SEMANTICS,
        supports_requery=True, supports_resample=False,
        supports_short_chunk=True, stochastic_sampling=False,
    )
    mask = operator_mask(profiles, policy, fallback_available=True)
    assert set(mask) == {profile.operator_id for profile in profiles}
    assert mask["resample.source"] is False
    assert mask["fallback.persistent"] is True


def test_seed_ledger_keeps_all_randomness_layers_distinct() -> None:
    ledger = SeedLedger(1, 2, 3, 4, 0)
    assert ledger.to_dict() == {
        "init_state_id": 1,
        "environment_seed": 2,
        "source_sampling_seed": 3,
        "operator_seed": 4,
        "exact_repeat_replica": 0,
    }


def test_libero_correction_operator_keeps_candidate_seed_explicit() -> None:
    adapter = LiberoPolicyAdapter(policy_id="pi05.libero", family="pi05")
    profile = CorrectionProfile(
        "resample.source", CorrectionKind.RESAMPLE,
        candidate_ids=("candidate.0", "candidate.1"),
    )
    seen = []

    def propose(observation, ledger):
        seen.append(ledger.operator_seed)
        return np.zeros((1, 7), dtype=np.float32)

    operator = LiberoCorrectionOperator(
        profile=profile, proposer=propose, policy_adapter=adapter,
    )
    observation = LiberoBenchmarkAdapter(
        vector_env=_VectorEnv(), forkable=_Forkable(),
    ).observation_to_canonical(
        {"pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)},
         "proprio": np.zeros(8, dtype=np.float32)},
        task_text="task", timestamp_s=0.0,
    )
    result = operator.propose(observation, seed_ledger=SeedLedger(1, 2, 3, 4, 0))
    assert result is not None and result.action_dim == 7
    assert seen == [4]


class _Forkable:
    def __init__(self) -> None:
        self.value = "snapshot"
        self.restored = None

    def snapshot(self):
        return self.value

    def restore(self, value):
        self.restored = value


class _VectorEnv:
    def __init__(self) -> None:
        self.actions = []

    def reset(self):
        return {"reset": True}

    def step(self, action):
        self.actions.append(np.asarray(action))
        return ({}, np.array([0.0]), np.array([False]), np.array([False]), {})


def test_libero_benchmark_adapter_observation_snapshot_and_execute() -> None:
    env, forkable = _VectorEnv(), _Forkable()
    adapter = LiberoBenchmarkAdapter(vector_env=env, forkable=forkable)
    observation = adapter.observation_to_canonical(
        {
            "pixels": {
                "image": np.zeros((1, 4, 5, 3), dtype=np.uint8),
                "image2": np.ones((1, 4, 5, 3), dtype=np.uint8),
            },
            "proprio": np.arange(8, dtype=np.float32)[None, :],
        },
        task_text="move object",
        timestamp_s=0.0,
    )
    assert set(observation.images) == {"agentview", "wrist"}
    snapshot = adapter.snapshot()
    adapter.restore(snapshot)
    assert forkable.restored == "snapshot"
    token = LiberoPolicyAdapter(
        policy_id="pi05.libero", family="pi05",
    ).raw_to_canonical(np.zeros((2, 7), dtype=np.float32))
    assert len(adapter.execute(token)) == 2
    assert all(action.shape == (1, 7) for action in env.actions)

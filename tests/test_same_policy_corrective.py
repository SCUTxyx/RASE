from __future__ import annotations

import numpy as np
import pytest

from rase.collect.same_policy_corrective import (
    RECEDING_HORIZONS,
    CorrectiveArmSpec,
    RecedingHorizonSmolVLAContinuation,
    action_tensor_sha256,
    build_pre_c0_arm_specs,
    deterministic_sha256,
    nested_oracle_metrics,
    validate_action_tensor,
)

PROVENANCE = {
    "snapshot_sha256": "snapshot",
    "checkpoint_sha256": "checkpoint",
    "history_fingerprint": "history",
}


def test_arm_specs_freeze_protocol_families_seeds_horizons_and_provenance():
    specs = build_pre_c0_arm_specs(
        strict_resample_seeds=range(8),
        fresh_replan_seeds=range(100, 104),
        provenance=PROVENANCE,
    )

    assert [spec.name for spec in specs] == [
        "current_suffix",
        "strict_resample",
        "fresh_replan",
        "receding_horizon@1",
        "receding_horizon@2",
        "receding_horizon@4",
    ]
    assert specs[0].uses_active_suffix
    assert specs[1].seeds == tuple(range(8))
    assert specs[2].fresh_cache
    assert tuple(spec.execution_horizon for spec in specs[3:]) == RECEDING_HORIZONS
    assert all(spec.provenance == PROVENANCE for spec in specs)
    assert all(len(spec.fingerprint) == 64 for spec in specs)


def test_arm_specs_require_provenance_unique_seeds_and_frozen_horizons():
    with pytest.raises(ValueError, match="provenance"):
        build_pre_c0_arm_specs(
            strict_resample_seeds=[1],
            fresh_replan_seeds=[2],
            provenance={},
        )
    with pytest.raises(ValueError, match="unique"):
        build_pre_c0_arm_specs(
            strict_resample_seeds=[1, 1],
            fresh_replan_seeds=[2],
            provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="exactly"):
        build_pre_c0_arm_specs(
            strict_resample_seeds=[1],
            fresh_replan_seeds=[2],
            receding_horizons=[1, 2],
            provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="positive"):
        CorrectiveArmSpec(
            "bad", "receding_horizon", (1,), 0, False, True, PROVENANCE
        )


def test_hashes_are_deterministic_order_independent_and_action_sensitive():
    first = deterministic_sha256({"b": [2, 3], "a": 1})
    second = deterministic_sha256({"a": 1, "b": [2, 3]})
    assert first == second

    actions = np.arange(14, dtype=np.float64).reshape(2, 7)
    same_values = actions.astype(np.float32)
    changed = same_values.copy()
    changed[1, 6] += 1
    assert action_tensor_sha256(actions) == action_tensor_sha256(same_values)
    assert action_tensor_sha256(changed) != action_tensor_sha256(same_values)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros(7),
        np.zeros((0, 7)),
        np.zeros((2, 6)),
        np.full((1, 7), np.nan),
        np.full((1, 7), np.inf),
    ],
)
def test_action_validation_requires_finite_nonempty_t_by_7(actions):
    with pytest.raises(ValueError):
        validate_action_tensor(actions)


class _FakeContinuation:
    def __init__(self):
        self.resets = 0
        self.calls: list[int] = []

    def reset(self):
        self.resets += 1

    def act(self, observation, *, task):
        del task
        self.calls.append(observation["step"])
        return np.full(7, observation["step"], dtype=np.float64)


def test_receding_continuation_resets_cache_every_execution_horizon():
    fake = _FakeContinuation()
    policy = RecedingHorizonSmolVLAContinuation(
        execution_horizon=2,
        continuation=fake,
    )
    policy.reset()
    actions = [
        policy.act({"step": step}, task="task")
        for step in range(5)
    ]

    assert fake.resets == 3  # rollout reset, then before actions 3 and 5
    assert fake.calls == [0, 1, 2, 3, 4]
    assert [float(action[0]) for action in actions] == list(map(float, range(5)))
    assert policy.metrics()["cache_resets"] == 3
    assert policy.metrics()["actions"] == 5


def test_receding_continuation_rejects_bad_horizon_and_nonfinite_action():
    with pytest.raises(ValueError, match="positive"):
        RecedingHorizonSmolVLAContinuation(
            execution_horizon=0,
            continuation=_FakeContinuation(),
        )

    fake = _FakeContinuation()
    fake.act = lambda observation, *, task: np.full(7, np.nan)
    policy = RecedingHorizonSmolVLAContinuation(
        execution_horizon=1,
        continuation=fake,
    )
    policy.reset()
    with pytest.raises(ValueError, match="finite"):
        policy.act({}, task="task")


def test_nested_oracle_metrics_reports_incremental_family_headroom():
    outcomes = {
        "s0": {
            "current_suffix": True,
            "strict_resample": [False, False],
            "fresh_replan": [False],
            "receding_horizon@1": False,
            "receding_horizon@2": False,
            "receding_horizon@4": False,
        },
        "s1": {
            "current_suffix": False,
            "strict_resample": [{"success": False}, {"success": True}],
            "fresh_replan": False,
            "receding_horizon@1": False,
            "receding_horizon@2": False,
            "receding_horizon@4": False,
        },
        "s2": {
            "current_suffix": False,
            "strict_resample": False,
            "fresh_replan": True,
            "receding_horizon@1": False,
            "receding_horizon@2": False,
            "receding_horizon@4": False,
        },
        "s3": {
            "current_suffix": False,
            "strict_resample": False,
            "fresh_replan": False,
            "receding_horizon@1": False,
            "receding_horizon@2": True,
            "receding_horizon@4": False,
        },
    }

    result = nested_oracle_metrics(outcomes)

    assert result["successes"] == {"S0": 1, "S1": 2, "S2": 3, "S3": 4}
    assert result["rates"] == {"S0": 0.25, "S1": 0.5, "S2": 0.75, "S3": 1.0}
    assert result["headroom"] == {
        "H_sampling": 0.25,
        "H_reconditioning": 0.25,
        "H_closed_loop": 0.25,
        "H_total": 0.75,
    }
    assert result["per_state"]["s3"] == {
        "S0": False,
        "S1": False,
        "S2": False,
        "S3": True,
    }


def test_nested_oracle_requires_all_families_for_every_state():
    with pytest.raises(ValueError, match="missing arm families"):
        nested_oracle_metrics(
            {
                "s0": {
                    "current_suffix": False,
                    "strict_resample": False,
                    "fresh_replan": False,
                }
            }
        )

"""Tests for the formal one-sided alpha-spending protocol."""

from __future__ import annotations

from rase.collect.adaptive import (
    PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    sequential_adaptive_sample,
    z_from_alpha,
)


def test_z_one_sided_less_than_two_sided():
    assert z_from_alpha(0.05, sidedness="one-sided") < z_from_alpha(
        0.05, sidedness="two-sided"
    )


def test_all_fail_can_stop_at_stage_one_with_n6():
    seen = []

    def rollout(index: int) -> bool:
        seen.append(index)
        return False

    result = sequential_adaptive_sample(
        rollout,
        threshold=0.5,
        n_first=6,
        n_total=20,
        alpha_first=0.01,
        alpha_final=0.04,
        sidedness="one-sided",
    )
    assert result.trials == 6
    assert result.stopped_early is True
    assert result.upper < 0.5
    assert result.protocol_version == PROTOCOL_SEQUENTIAL_ONESIDED_V1
    assert seen == list(range(6))


def test_boundary_continues_to_n_total():
    outcomes = [True, False] * 10
    iterator = iter(outcomes)
    seen = []

    def rollout(index: int) -> bool:
        seen.append(index)
        return next(iterator)

    result = sequential_adaptive_sample(
        rollout,
        threshold=0.5,
        n_first=6,
        n_total=20,
        alpha_first=0.01,
        alpha_final=0.04,
        sidedness="one-sided",
    )
    assert result.trials == 20
    assert result.stopped_early is False
    assert len(seen) == 20


def test_all_success_can_stop_early():
    result = sequential_adaptive_sample(
        lambda _: True,
        threshold=0.5,
        n_first=6,
        n_total=20,
        alpha_first=0.01,
        alpha_final=0.04,
        sidedness="one-sided",
    )
    assert result.trials == 6
    assert result.lower > 0.5
    assert result.stopped_early is True

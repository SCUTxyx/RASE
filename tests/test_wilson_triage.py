import pytest

from rase.collect.adaptive import (
    SetLabel,
    adaptive_sample,
    estimate,
    triage,
    wilson_interval,
)


def test_wilson_known_values_without_stats_packages():
    lower, upper = wilson_interval(5, 10)
    assert lower == pytest.approx(0.236593, abs=1e-6)
    assert upper == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_two_stage_uses_ten_when_stage_one_crosses_threshold():
    outcomes = iter([True, False, False, True, True, True, True, True, True, True])
    seen = []

    def rollout(index):
        seen.append(index)
        return next(outcomes)

    result = adaptive_sample(rollout)
    assert result.trials == 10
    assert result.successes == 8
    assert seen == list(range(10))


@pytest.mark.parametrize("stage_one_successes", range(4))
def test_default_n3_wilson_stage_cannot_stop_early(stage_one_successes):
    stage_one = [True] * stage_one_successes + [False] * (3 - stage_one_successes)
    later = [False] * 7
    outcomes = iter(stage_one + later)
    result = adaptive_sample(lambda _: next(outcomes))
    assert result.trials == 10


def test_two_stage_can_stop_after_three_when_interval_excludes_threshold():
    result = adaptive_sample(lambda _: False, threshold=0.9)
    assert result.trials == 3


def test_exact_set_rules():
    bad = estimate(0, 10)
    good = estimate(10, 10)
    crossing = estimate(5, 10)

    assert triage([bad] * 8) is SetLabel.C
    assert triage([good, good, good, *([crossing] * 5)]) is SetLabel.A
    assert triage([good, *([crossing] * 7)]) is SetLabel.B
    assert triage([crossing] * 8) is SetLabel.UNCERTAIN


def test_boundaries_are_strict():
    # Synthetic values make equality behavior explicit.
    equal_upper = estimate(0, 10)
    equal_upper = equal_upper.__class__(
        equal_upper.successes, equal_upper.trials, equal_upper.rate, 0.0, 0.5
    )
    assert triage([equal_upper] * 8) is SetLabel.UNCERTAIN

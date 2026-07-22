from collections import Counter

from rase.collect.perturb_sampler import (
    DIMENSION_QUOTAS,
    SUITE_QUOTAS,
    quota_counts,
    sample_perturbations,
)


def test_exact_protocol_quotas_for_100_requests():
    requests = sample_perturbations(100, seed=9)
    assert Counter(item.dimension for item in requests) == DIMENSION_QUOTAS
    assert Counter(item.suite for item in requests) == SUITE_QUOTAS


def test_largest_remainder_preserves_small_batch_total():
    dimensions, suites = quota_counts(7)
    assert sum(dimensions.values()) == 7
    assert sum(suites.values()) == 7


def test_sampler_is_deterministic_and_levels_follow_protocol():
    first = sample_perturbations(53, seed=123)
    assert first == sample_perturbations(53, seed=123)
    assert first != sample_perturbations(53, seed=124)
    for item in first:
        minimum = 4 if item.dimension == "other" else 3
        assert minimum <= item.level <= 5
        if item.dimension == "combination":
            assert item.subdimension == "camera+robot"

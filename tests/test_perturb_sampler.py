from collections import Counter

import pytest

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


def test_levels_by_dimension_override_preserves_other_defaults():
    requests = sample_perturbations(
        40,
        seed=3,
        dimension_quotas={"camera": 1, "robot": 1},
        levels_by_dimension={"camera": [1, 2]},
    )
    assert {item.level for item in requests if item.dimension == "camera"} <= {1, 2}
    assert {item.level for item in requests if item.dimension == "robot"} <= {3, 4, 5}


def test_clean_control_requests_are_explicit_l0_without_perturbation():
    requests = sample_perturbations(
        16,
        seed=11,
        dimension_quotas={"clean": 1},
        suite_quotas={"Spatial": 1, "Goal": 1},
        levels_by_dimension={"clean": [0]},
    )
    assert {item.dimension for item in requests} == {"clean"}
    assert {item.subdimension for item in requests} == {"none"}
    assert {item.level for item in requests} == {0}


def test_factorial_cells_balance_every_suite_cell_pair():
    cells = [
        {"dimension": "clean", "level": 0},
        {"dimension": "camera", "level": 1},
        {"dimension": "robot", "level": 1},
    ]
    requests = sample_perturbations(
        24,
        seed=17,
        suite_quotas={"Spatial": 1, "Object": 1, "Goal": 1, "Long": 1},
        factorial_cells=cells,
    )

    assert Counter(
        (item.suite, item.dimension, item.level) for item in requests
    ) == Counter(
        {
            (suite, dimension, level): 2
            for suite in ("Spatial", "Object", "Goal", "Long")
            for dimension, level in (("clean", 0), ("camera", 1), ("robot", 1))
        }
    )
    assert requests == sample_perturbations(
        24,
        seed=17,
        suite_quotas={"Spatial": 1, "Object": 1, "Goal": 1, "Long": 1},
        factorial_cells=cells,
    )


def test_factorial_cells_reject_incomplete_design_and_ambiguous_overrides():
    cells = [{"dimension": "clean", "level": 0}]
    with pytest.raises(ValueError, match="must be divisible"):
        sample_perturbations(
            5,
            suite_quotas={"Spatial": 1, "Object": 1},
            factorial_cells=cells,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        sample_perturbations(
            4,
            dimension_quotas={"clean": 1},
            factorial_cells=cells,
        )


def test_factorial_cell_weights_oversample_boundary_control():
    requests = sample_perturbations(
        8,
        seed=19,
        suite_quotas={"Spatial": 1, "Goal": 1},
        factorial_cells=[
            {"dimension": "clean", "level": 0, "weight": 2},
            {"dimension": "camera", "level": 1},
            {"dimension": "robot", "level": 1},
        ],
    )

    assert Counter(item.dimension for item in requests) == {
        "clean": 4,
        "camera": 2,
        "robot": 2,
    }


@pytest.mark.parametrize(
    "levels, match",
    [
        ({"camera": []}, "must not be empty"),
        ({"camera": [0, 2]}, "within L1-L5"),
        ({"camera": [2, 6]}, "within L1-L5"),
        ({"camera": [2, 2]}, "duplicates"),
        ({"unknown": [1]}, "unknown level dimensions"),
        ({"clean": [1]}, "exactly L0"),
    ],
)
def test_levels_by_dimension_validation(levels, match):
    with pytest.raises(ValueError, match=match):
        sample_perturbations(4, levels_by_dimension=levels)

"""Deterministic quota sampler for NGC Step 1 episode requests."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DIMENSION_QUOTAS = {
    "camera": 30,
    "robot": 30,
    "combination": 20,
    "layout": 10,
    "other": 10,
}
SUITE_QUOTAS = {"Long": 40, "Goal": 25, "Spatial": 20, "Object": 15}

_SUBDIMENSIONS = {
    "clean": ("none",),
    "camera": ("viewpoint",),
    "robot": ("initial_state",),
    "combination": ("camera+robot",),
    "layout": ("confounder", "target_displacement"),
    "other": ("light", "background", "noise"),
}
DEFAULT_LEVELS_BY_DIMENSION = {
    "clean": (0,),
    "camera": (3, 4, 5),
    "robot": (3, 4, 5),
    "combination": (3, 4, 5),
    "layout": (3, 4, 5),
    "other": (4, 5),
}


def validate_levels_by_dimension(
    levels_by_dimension: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    """Validate level overrides and merge them with protocol defaults."""
    levels = dict(DEFAULT_LEVELS_BY_DIMENSION)
    if levels_by_dimension is None:
        return levels
    unknown = set(levels_by_dimension) - set(_SUBDIMENSIONS)
    if unknown:
        raise ValueError(f"unknown level dimensions: {', '.join(sorted(unknown))}")
    for dimension, configured in levels_by_dimension.items():
        values = tuple(int(level) for level in configured)
        if not values:
            raise ValueError(f"levels for {dimension} must not be empty")
        allowed = {0} if dimension == "clean" else set(range(1, 6))
        invalid = sorted(set(values) - allowed)
        if invalid:
            expected = "exactly L0" if dimension == "clean" else "within L1-L5"
            raise ValueError(f"levels for {dimension} must be {expected}; got {invalid}")
        if len(set(values)) != len(values):
            raise ValueError(f"levels for {dimension} must not contain duplicates")
        levels[dimension] = values
    return levels


@dataclass(frozen=True)
class PerturbationRequest:
    index: int
    suite: str
    dimension: str
    subdimension: str
    level: int
    seed: int
    global_episode_index: int | None = None
    batch_id: int | None = None
    task_id: int | None = None
    init_state_id: int | None = None
    episode_id: str | None = None


def _apportion(total: int, quotas: Mapping[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    weight = sum(quotas.values())
    if weight <= 0 or any(value < 0 for value in quotas.values()):
        raise ValueError("quotas must be non-negative with positive total weight")
    raw = {key: total * value / weight for key, value in quotas.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(quotas, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def quota_counts(total: int) -> tuple[dict[str, int], dict[str, int]]:
    return _apportion(total, DIMENSION_QUOTAS), _apportion(total, SUITE_QUOTAS)


def _expanded(total: int, quotas: Mapping[str, int], rng: random.Random) -> list[str]:
    values = [
        name
        for name, count in _apportion(total, quotas).items()
        for _ in range(count)
    ]
    rng.shuffle(values)
    return values


def sample_perturbations(
    total: int,
    seed: int = 0,
    *,
    dimension_quotas: Mapping[str, int] | None = None,
    suite_quotas: Mapping[str, int] | None = None,
    levels_by_dimension: Mapping[str, Sequence[int]] | None = None,
    factorial_cells: Sequence[Mapping[str, object]] | None = None,
) -> list[PerturbationRequest]:
    """Return exactly ``total`` requests satisfying both quota marginals."""
    dimensions_config = dict(dimension_quotas or DIMENSION_QUOTAS)
    suites_config = dict(suite_quotas or SUITE_QUOTAS)
    unknown_dimensions = set(dimensions_config) - set(_SUBDIMENSIONS)
    if unknown_dimensions:
        raise ValueError(
            f"unknown perturbation dimensions: {', '.join(sorted(unknown_dimensions))}"
        )
    unknown_suites = set(suites_config) - set(SUITE_QUOTAS)
    if unknown_suites:
        raise ValueError(f"unknown suites: {', '.join(sorted(unknown_suites))}")
    if factorial_cells is not None:
        if dimension_quotas is not None or levels_by_dimension is not None:
            raise ValueError(
                "factorial_cells cannot be combined with dimension_quotas or "
                "levels_by_dimension"
            )
        return _sample_factorial(
            total,
            seed=seed,
            suite_weights=suites_config,
            raw_cells=factorial_cells,
        )
    levels_config = validate_levels_by_dimension(levels_by_dimension)

    rng = random.Random(seed)
    dimensions = _expanded(total, dimensions_config, rng)
    suites = _expanded(total, suites_config, rng)
    requests = []
    for index, (dimension, suite) in enumerate(zip(dimensions, suites)):
        request_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8], "big"
        ) & 0x7FFFFFFF
        local = random.Random(request_seed)
        requests.append(
            PerturbationRequest(
                index=index,
                suite=suite,
                dimension=dimension,
                subdimension=local.choice(_SUBDIMENSIONS[dimension]),
                level=local.choice(levels_config[dimension]),
                seed=request_seed,
            )
        )
    return requests


def _sample_factorial(
    total: int,
    *,
    seed: int,
    suite_weights: Mapping[str, int],
    raw_cells: Sequence[Mapping[str, object]],
) -> list[PerturbationRequest]:
    if not raw_cells:
        raise ValueError("factorial_cells must not be empty")
    if any(
        int(weight) != weight or int(weight) <= 0
        for weight in suite_weights.values()
    ):
        raise ValueError("factorial suite weights must be positive integers")

    cells: list[tuple[str, str, int, int]] = []
    for raw in raw_cells:
        dimension = str(raw.get("dimension") or "")
        if dimension not in _SUBDIMENSIONS:
            raise ValueError(f"unknown factorial dimension: {dimension!r}")
        try:
            level = int(raw["level"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("factorial cell requires an integer level") from error
        validate_levels_by_dimension({dimension: [level]})
        subdimension = str(raw.get("subdimension") or _SUBDIMENSIONS[dimension][0])
        if subdimension not in _SUBDIMENSIONS[dimension]:
            raise ValueError(
                f"invalid subdimension {subdimension!r} for {dimension!r}"
            )
        try:
            weight = int(raw.get("weight", 1))
        except (TypeError, ValueError) as error:
            raise ValueError("factorial cell weight must be a positive integer") from error
        if weight <= 0 or weight != raw.get("weight", 1):
            raise ValueError("factorial cell weight must be a positive integer")
        cells.append((dimension, subdimension, level, weight))
    identities = [
        (dimension, subdimension, level)
        for dimension, subdimension, level, _ in cells
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("factorial_cells must not contain duplicates")

    design = [
        (suite, (dimension, subdimension, level))
        for suite, weight in suite_weights.items()
        for _ in range(int(weight))
        for dimension, subdimension, level, cell_weight in cells
        for _ in range(cell_weight)
    ]
    if total < 0 or total % len(design):
        raise ValueError(
            f"factorial total {total} must be divisible by design size {len(design)}"
        )
    assignments = design * (total // len(design))
    random.Random(seed).shuffle(assignments)
    requests = []
    for index, (suite, (dimension, subdimension, level)) in enumerate(assignments):
        request_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8], "big"
        ) & 0x7FFFFFFF
        requests.append(
            PerturbationRequest(
                index=index,
                suite=suite,
                dimension=dimension,
                subdimension=subdimension,
                level=level,
                seed=request_seed,
            )
        )
    return requests


def summarize(requests: Sequence[PerturbationRequest]) -> dict[str, dict[str, int]]:
    dimensions = {key: 0 for key in DIMENSION_QUOTAS}
    suites = {key: 0 for key in SUITE_QUOTAS}
    for request in requests:
        dimensions[request.dimension] = dimensions.get(request.dimension, 0) + 1
        suites[request.suite] += 1
    return {"dimensions": dimensions, "suites": suites}

"""Stratified pool sampling for W3 statistical pilots."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import product
from typing import Any

from rase.collect.state_pool import StatePool

SUITE_ALIASES = {
    "Spatial": "Spatial",
    "Object": "Object",
    "Goal": "Goal",
    "Long": "Long",
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}

# Empirically matched to LIBERO / Plus episode horizons observed in W3 pilot
# (final_timestep after stop_reason=horizon). Used only for sampling filters.
DEFAULT_SUITE_HORIZONS: dict[str, int] = {
    "Spatial": 280,
    "Object": 280,
    "Goal": 300,
    "Long": 520,
}

VALID_STRATA = ("suite", "dim", "level", "t0_bin", "outcome")
T0Bin = tuple[str, int | None, int | None]


def remaining_steps(
    meta: Mapping[str, object],
    *,
    suite_horizons: Mapping[str, int] | None = None,
) -> int | None:
    """Return H - t0 when both horizon and snapshot step are known."""
    horizons = dict(DEFAULT_SUITE_HORIZONS)
    if suite_horizons:
        horizons.update({str(k): int(v) for k, v in suite_horizons.items()})
    suite = SUITE_ALIASES.get(str(meta.get("suite", "")), str(meta.get("suite", "")))
    horizon = horizons.get(suite)
    if horizon is None:
        return None
    step = meta.get("step", meta.get("timestep"))
    if step is None:
        return None
    return int(horizon) - int(step)


def snapshot_step(meta: Mapping[str, object]) -> int | None:
    """Return snapshot timestep t0 when present."""
    step = meta.get("step", meta.get("timestep"))
    if step is None:
        return None
    return int(step)


def normalize_t0_bins(
    bins: Mapping[str, Sequence[int | None]] | Sequence[Mapping[str, object]] | None,
) -> tuple[T0Bin, ...]:
    """Normalize named, non-overlapping ``[min, max)`` t0 bins."""
    if bins is None:
        return ()
    normalized: list[T0Bin] = []
    items: Sequence[tuple[str, object]]
    if isinstance(bins, Mapping):
        items = list(bins.items())
    else:
        items = [(str(item.get("name", "")), item) for item in bins]
    for name, raw in items:
        if not name:
            raise ValueError("t0 bin names must be non-empty")
        if isinstance(raw, Mapping):
            lower_raw, upper_raw = raw.get("min"), raw.get("max")
        else:
            values = list(raw)  # type: ignore[arg-type]
            if len(values) != 2:
                raise ValueError(f"t0 bin {name!r} must contain [min, max]")
            lower_raw, upper_raw = values
        lower = None if lower_raw is None else int(lower_raw)
        upper = None if upper_raw is None else int(upper_raw)
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"t0 bin {name!r} must have min < max")
        normalized.append((name, lower, upper))
    if len({name for name, _, _ in normalized}) != len(normalized):
        raise ValueError("t0 bin names must be unique")
    for index, (_, lower, upper) in enumerate(normalized):
        for other_name, other_lower, other_upper in normalized[index + 1 :]:
            left = float("-inf") if lower is None else lower
            right = float("inf") if upper is None else upper
            other_left = float("-inf") if other_lower is None else other_lower
            other_right = float("inf") if other_upper is None else other_upper
            if max(left, other_left) < min(right, other_right):
                raise ValueError(f"t0 bins overlap with {other_name!r}")
    return tuple(normalized)


def _t0_bin(step: int, bins: tuple[T0Bin, ...]) -> str | None:
    for name, lower, upper in bins:
        if (lower is None or step >= lower) and (upper is None or step < upper):
            return name
    return None


def _validate_options(
    *,
    strata: tuple[str, ...],
    selection: str,
    t0_bins: tuple[T0Bin, ...],
    episode_outcomes: tuple[str, ...] | None,
) -> None:
    unknown = set(strata) - set(VALID_STRATA)
    if unknown:
        raise ValueError(f"unknown strata: {', '.join(sorted(unknown))}")
    if len(set(strata)) != len(strata):
        raise ValueError("strata must not contain duplicates")
    if selection not in {"earliest", "random"}:
        raise ValueError("selection must be 'earliest' or 'random'")
    if "t0_bin" in strata and not t0_bins:
        raise ValueError("t0_bins are required when stratifying by t0_bin")
    if "outcome" in strata and not episode_outcomes:
        raise ValueError(
            "episode_outcomes are required when stratifying by outcome"
        )


def _axis_values(
    *,
    strata: tuple[str, ...],
    suites: tuple[str, ...],
    dims: tuple[str, ...],
    levels: tuple[int, ...],
    t0_bins: tuple[T0Bin, ...],
    episode_outcomes: tuple[str, ...] | None,
) -> list[tuple[Any, ...]]:
    values: dict[str, tuple[object, ...]] = {
        "suite": suites,
        "dim": dims,
        "level": levels,
        "t0_bin": tuple(name for name, _, _ in t0_bins),
        "outcome": episode_outcomes or (),
    }
    return list(product(*(values[name] for name in strata))) if strata else [()]


def _eligible_buckets(
    pool: StatePool,
    *,
    dims: tuple[str, ...],
    suites: tuple[str, ...],
    levels: tuple[int, ...],
    strata: tuple[str, ...],
    t0_bins: tuple[T0Bin, ...],
    episode_outcomes: tuple[str, ...] | None,
    excluded_keys: Collection[str],
    excluded_episode_keys: Collection[str],
    min_remaining_steps: int | None,
    max_t0: int | None,
    suite_horizons: Mapping[str, int] | None,
    distinct_episodes: bool,
) -> tuple[dict[tuple[Any, ...], list[tuple[int, str, str]]], dict[str, int]]:
    buckets: dict[tuple[Any, ...], list[tuple[int, str, str]]] = defaultdict(list)
    audit = {
        "excluded": 0,
        "excluded_episode_group": 0,
        "missing_metadata": 0,
        "outside_t0_bins": 0,
        "missing_episode_group": 0,
    }
    manifest_states = pool.manifest()["states"]
    excluded_episode_groups: set[str] = set()
    for key in excluded_episode_keys:
        entry = manifest_states.get(key)
        if entry is None:
            raise ValueError(f"episode-exclusion state is absent from pool: {key}")
        meta_path = pool.root / entry["path"] / "meta.json"
        if not meta_path.is_file():
            raise ValueError(f"episode-exclusion metadata is missing: {key}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task_id = str(meta.get("task_id", ""))
        episode_id = str(meta.get("episode_id", ""))
        if not task_id or not episode_id:
            raise ValueError(f"episode-exclusion state lacks task/episode id: {key}")
        excluded_episode_groups.add(f"{task_id}\x00{episode_id}")

    for key, entry in manifest_states.items():
        if key in excluded_keys:
            audit["excluded"] += 1
            continue
        meta_path = pool.root / entry["path"] / "meta.json"
        if not meta_path.is_file():
            audit["missing_metadata"] += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = str(meta.get("perturb_dim", ""))
        suite = SUITE_ALIASES.get(str(meta.get("suite", "")), str(meta.get("suite", "")))
        level = int(meta.get("level", 0))
        outcome = str(meta.get("episode_outcome", ""))
        if dim not in dims or suite not in suites or level not in levels:
            continue
        if episode_outcomes is not None and outcome not in episode_outcomes:
            continue
        t0 = snapshot_step(meta)
        if t0 is None:
            continue
        if max_t0 is not None and t0 > int(max_t0):
            continue
        if min_remaining_steps is not None:
            rem = remaining_steps(meta, suite_horizons=suite_horizons)
            if rem is None or rem < int(min_remaining_steps):
                continue
        bin_name = _t0_bin(t0, t0_bins)
        if "t0_bin" in strata and bin_name is None:
            audit["outside_t0_bins"] += 1
            continue
        fields: dict[str, object] = {
            "suite": suite,
            "dim": dim,
            "level": level,
            "t0_bin": bin_name,
            "outcome": outcome,
        }
        task_id = str(meta.get("task_id", ""))
        episode_id = str(meta.get("episode_id", ""))
        if distinct_episodes and (not task_id or not episode_id):
            audit["missing_episode_group"] += 1
            continue
        episode_group = f"{task_id}\x00{episode_id}" if task_id and episode_id else key
        if episode_group in excluded_episode_groups:
            audit["excluded_episode_group"] += 1
            continue
        buckets[tuple(fields[name] for name in strata)].append(
            (t0, key, episode_group)
        )
    return buckets, audit


def sample_stratified_keys(
    pool: StatePool,
    *,
    per_cell: int = 2,
    seed: int = 0,
    dims: tuple[str, ...] = ("camera", "robot"),
    suites: tuple[str, ...] = ("Spatial", "Object", "Goal", "Long"),
    levels: tuple[int, ...] = (3, 4, 5),
    min_remaining_steps: int | None = None,
    max_t0: int | None = None,
    suite_horizons: Mapping[str, int] | None = None,
    strata: tuple[str, ...] = ("suite", "dim"),
    t0_bins: Mapping[str, Sequence[int | None]] | Sequence[Mapping[str, object]] | None = None,
    selection: str = "earliest",
    episode_outcomes: tuple[str, ...] | None = None,
    excluded_keys: Collection[str] = (),
    excluded_episode_keys: Collection[str] = (),
    distinct_episodes: bool = False,
) -> list[str]:
    """Sample exactly ``per_cell`` states from each configured stratum.

    Defaults preserve the W3/W4 suite×dimension, earliest-t0 behavior. Add
    ``level`` and/or ``t0_bin`` to ``strata`` for a true frontier sample.
    """
    if per_cell < 0:
        raise ValueError("per_cell must be non-negative")
    normalized_bins = normalize_t0_bins(t0_bins)
    _validate_options(
        strata=strata,
        selection=selection,
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
    )
    buckets, _audit = _eligible_buckets(
        pool,
        dims=dims,
        suites=suites,
        levels=levels,
        strata=strata,
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
        excluded_keys=excluded_keys,
        excluded_episode_keys=excluded_episode_keys,
        min_remaining_steps=min_remaining_steps,
        max_t0=max_t0,
        suite_horizons=suite_horizons,
        distinct_episodes=distinct_episodes,
    )

    rng = random.Random(seed)
    chosen: list[str] = []
    used_episode_groups: set[str] = set()
    for cell_key in _axis_values(
        strata=strata,
        suites=suites,
        dims=dims,
        levels=levels,
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
    ):
        cell = list(buckets.get(cell_key, []))
        rng.shuffle(cell)
        if selection == "earliest":
            # Stable sort: earliest t0 first; shuffle order breaks equal-t0 ties.
            cell.sort(key=lambda item: item[0])
        if distinct_episodes:
            deduplicated = []
            seen_groups: set[str] = set()
            for item in cell:
                if item[2] not in seen_groups:
                    deduplicated.append(item)
                    seen_groups.add(item[2])
            cell = deduplicated
        available = [item for item in cell if item[2] not in used_episode_groups]
        if len(available) < per_cell:
            label = " ".join(f"{name}={value}" for name, value in zip(strata, cell_key))
            filters = []
            if min_remaining_steps is not None:
                filters.append(f"min_remaining_steps={min_remaining_steps}")
            if max_t0 is not None:
                filters.append(f"max_t0={max_t0}")
            suffix = f" after {', '.join(filters)}" if filters else ""
            raise ValueError(
                f"need {per_cell} states for {label or 'all'}, have "
                f"{len(available)} eligible episode group(s){suffix}"
            )
        selected = available[:per_cell]
        chosen.extend(key for _, key, _ in selected)
        if distinct_episodes:
            used_episode_groups.update(group for _, _, group in selected)
    return chosen


def inventory_cell_counts(
    pool: StatePool,
    *,
    dims: tuple[str, ...] = ("camera", "robot"),
    suites: tuple[str, ...] = ("Spatial", "Object", "Goal", "Long"),
    levels: tuple[int, ...] = (3, 4, 5),
    min_remaining_steps: int | None = None,
    max_t0: int | None = None,
    suite_horizons: Mapping[str, int] | None = None,
    strata: tuple[str, ...] = ("suite", "dim"),
    t0_bins: Mapping[str, Sequence[int | None]] | Sequence[Mapping[str, object]] | None = None,
    episode_outcomes: tuple[str, ...] | None = None,
    excluded_keys: Collection[str] = (),
    excluded_episode_keys: Collection[str] = (),
    distinct_episodes: bool = False,
) -> dict[str, object]:
    """Return exact eligible counts for every requested Cartesian cell."""
    normalized_bins = normalize_t0_bins(t0_bins)
    _validate_options(
        strata=strata,
        selection="earliest",
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
    )
    buckets, audit = _eligible_buckets(
        pool,
        dims=dims,
        suites=suites,
        levels=levels,
        strata=strata,
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
        excluded_keys=excluded_keys,
        excluded_episode_keys=excluded_episode_keys,
        min_remaining_steps=min_remaining_steps,
        max_t0=max_t0,
        suite_horizons=suite_horizons,
        distinct_episodes=distinct_episodes,
    )

    cells: list[dict[str, object]] = []
    counts: list[int] = []
    for cell_key in _axis_values(
        strata=strata,
        suites=suites,
        dims=dims,
        levels=levels,
        t0_bins=normalized_bins,
        episode_outcomes=episode_outcomes,
    ):
        cell = buckets.get(cell_key, ())
        n = len({item[2] for item in cell}) if distinct_episodes else len(cell)
        counts.append(n)
        cells.append({**dict(zip(strata, cell_key)), "n": n})
    max_per_cell = min(counts) if counts else 0
    return {
        "total": sum(counts),
        "max_per_cell": max_per_cell,
        "n_cells": len(cells),
        "cells": cells,
        "audit": audit,
        "filters": {
            "dims": list(dims),
            "suites": list(suites),
            "levels": list(levels),
            "strata": list(strata),
            "t0_bins": [
                {"name": name, "min": lower, "max": upper}
                for name, lower, upper in normalized_bins
            ],
            "episode_outcomes": (
                list(episode_outcomes) if episode_outcomes is not None else None
            ),
            "excluded_keys": sorted(excluded_keys),
            "min_remaining_steps": min_remaining_steps,
            "max_t0": max_t0,
            "distinct_episodes": distinct_episodes,
        },
    }

"""Stratified pool sampling for W3 statistical pilots."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping

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
) -> list[str]:
    """Sample ``per_cell`` states for each suite×dim cell (L3–L5 preferred).

    When ``min_remaining_steps`` is set, drop states with
    ``H_suite - snapshot_step < min_remaining_steps`` (ADEQUATE filter).
    When ``max_t0`` is set, drop states with snapshot step ``> max_t0``.
    Within each cell, earlier forks (smaller t0) are preferred; the RNG only
    breaks ties among equal ``t0``.
    """
    states = pool.manifest()["states"]
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for key, entry in states.items():
        meta_path = pool.root / entry["path"] / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = str(meta.get("perturb_dim", ""))
        suite = SUITE_ALIASES.get(str(meta.get("suite", "")), str(meta.get("suite", "")))
        level = int(meta.get("level", 0))
        if dim not in dims or suite not in suites or level not in levels:
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
        buckets[(suite, dim)].append((t0, key))

    rng = random.Random(seed)
    chosen: list[str] = []
    for suite in suites:
        for dim in dims:
            cell = list(buckets.get((suite, dim), []))
            rng.shuffle(cell)
            # Stable sort: earliest t0 first; shuffle order breaks equal-t0 ties.
            cell.sort(key=lambda item: item[0])
            if len(cell) < per_cell:
                filters = []
                if min_remaining_steps is not None:
                    filters.append(f"min_remaining_steps={min_remaining_steps}")
                if max_t0 is not None:
                    filters.append(f"max_t0={max_t0}")
                suffix = f" after {', '.join(filters)}" if filters else ""
                raise ValueError(
                    f"need {per_cell} states for suite={suite} dim={dim}, "
                    f"have {len(cell)}{suffix}"
                )
            chosen.extend(key for _, key in cell[:per_cell])
    return chosen

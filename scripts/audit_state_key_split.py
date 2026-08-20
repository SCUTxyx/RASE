#!/usr/bin/env python3
"""Audit state and episode-group isolation between reference and held-out keys."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_keys(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = payload if isinstance(payload, list) else payload.get("state_keys") or []
    return [str(key) for key in keys], payload if isinstance(payload, dict) else {}


def audit_split(
    reference_keys: list[str],
    heldout_keys: list[str],
    metadata: dict[str, dict[str, Any]],
    *,
    expected_states: int | None = None,
    expected_per_cell: int | None = None,
) -> dict[str, Any]:
    def step_summary(keys: list[str]) -> dict[str, Any]:
        values = sorted(
            int(metadata[key]["step"])
            for key in keys
            if metadata[key].get("step") is not None
        )
        return {
            "n": len(values),
            "n_missing": len(keys) - len(values),
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None,
            "unique": sorted(set(values)),
        }

    def group(key: str) -> tuple[str, str]:
        row = metadata[key]
        return str(row["task_id"]), str(row["episode_id"])

    reference_groups = {group(key) for key in reference_keys}
    heldout_groups = {group(key) for key in heldout_keys}
    cells = Counter(
        f"{metadata[key]['perturb_dim']}-L{int(metadata[key]['level'])}"
        for key in heldout_keys
    )
    suites = Counter(str(metadata[key]["suite"]) for key in heldout_keys)
    suite_steps = {
        suite: step_summary(
            [key for key in heldout_keys if str(metadata[key]["suite"]) == suite]
        )
        for suite in sorted(suites)
    }
    state_overlap = sorted(set(reference_keys) & set(heldout_keys))
    group_overlap = sorted(reference_groups & heldout_groups)
    failures = []
    if len(heldout_keys) != len(set(heldout_keys)):
        failures.append("duplicate held-out state keys")
    if len(heldout_groups) != len(heldout_keys):
        failures.append("held-out states are not episode-distinct")
    if state_overlap:
        failures.append("state-key overlap")
    if group_overlap:
        failures.append("episode-group overlap")
    if expected_states is not None and len(heldout_keys) != expected_states:
        failures.append(f"expected {expected_states} held-out states")
    if expected_per_cell is not None and (
        len(cells) != 4 or any(count != expected_per_cell for count in cells.values())
    ):
        failures.append(f"expected four cells with {expected_per_cell} states each")
    return {
        "valid": not failures,
        "failures": failures,
        "n_reference_states": len(reference_keys),
        "n_reference_episode_groups": len(reference_groups),
        "n_heldout_states": len(heldout_keys),
        "n_heldout_episode_groups": len(heldout_groups),
        "state_overlap": state_overlap,
        "episode_group_overlap": [list(value) for value in group_overlap],
        "heldout_cells": dict(sorted(cells.items())),
        "heldout_suites": dict(sorted(suites.items())),
        "heldout_snapshot_steps": step_summary(heldout_keys),
        "heldout_snapshot_steps_by_suite": suite_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--expected-states", type=int, default=None)
    parser.add_argument("--expected-per-cell", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reference_keys, reference_payload = _load_keys(args.reference.resolve())
    heldout_keys, heldout_payload = _load_keys(args.heldout.resolve())
    pool_value = args.pool or heldout_payload.get("pool") or reference_payload.get("pool")
    if pool_value is None:
        raise SystemExit("pool is absent from artifacts; pass --pool")
    pool_path = Path(pool_value)
    if not pool_path.is_absolute():
        pool_path = (ROOT / pool_path).resolve()

    from rase.collect.state_pool import StatePool

    pool = StatePool(pool_path)
    keys = sorted(set(reference_keys) | set(heldout_keys))
    metadata = {
        key: pool.read_state(key, load_observations=False).metadata.to_dict()
        for key in keys
    }
    result = audit_split(
        reference_keys,
        heldout_keys,
        metadata,
        expected_states=args.expected_states,
        expected_per_cell=args.expected_per_cell,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"WROTE {args.output}", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

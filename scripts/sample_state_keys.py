#!/usr/bin/env python3
"""Sample configurable stratified state keys and write an exact inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _keys_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(key) for key in payload]
    return [str(key) for key in payload.get("state_keys", ())]


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _excluded_keys(sample: dict[str, Any], cli_paths: list[Path]) -> set[str]:
    keys = {str(key) for key in (sample.get("excluded_keys") or ())}
    configured_paths = sample.get("excluded_keys_json") or ()
    if isinstance(configured_paths, (str, Path)):
        configured_paths = [configured_paths]
    paths = [Path(path) for path in configured_paths]
    paths.extend(cli_paths)
    for path in paths:
        keys.update(_keys_from_json(_resolve_path(path)))
    return keys


def _excluded_episode_keys(
    sample: dict[str, Any], cli_paths: list[Path]
) -> set[str]:
    keys = {str(key) for key in (sample.get("excluded_episode_keys") or ())}
    configured_paths = sample.get("excluded_episode_keys_json") or ()
    if isinstance(configured_paths, (str, Path)):
        configured_paths = [configured_paths]
    paths = [Path(path) for path in configured_paths]
    paths.extend(cli_paths)
    for path in paths:
        keys.update(_keys_from_json(_resolve_path(path)))
    return keys


def _sample_kwargs(
    sample: dict[str, Any],
    *,
    extra_exclude_paths: list[Path] | None = None,
    extra_exclude_episode_paths: list[Path] | None = None,
) -> dict[str, Any]:
    suite_horizons = sample.get("suite_horizons")
    outcomes = sample.get("episode_outcomes", sample.get("episode_outcome"))
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    return {
        "per_cell": int(sample.get("per_cell", 2)),
        "seed": int(sample.get("sample_seed", 0)),
        "dims": tuple(sample.get("dims") or ("camera", "robot")),
        "suites": tuple(sample.get("suites") or ("Spatial", "Object", "Goal", "Long")),
        "levels": tuple(int(x) for x in (sample.get("levels") or (3, 4, 5))),
        "strata": tuple(sample.get("strata") or ("suite", "dim")),
        "t0_bins": sample.get("t0_bins"),
        "selection": str(sample.get("selection", "earliest")),
        "episode_outcomes": (
            tuple(str(value) for value in outcomes) if outcomes is not None else None
        ),
        "distinct_episodes": bool(sample.get("distinct_episodes", False)),
        "excluded_keys": _excluded_keys(sample, extra_exclude_paths or []),
        "excluded_episode_keys": _excluded_episode_keys(
            sample, extra_exclude_episode_paths or []
        ),
        "min_remaining_steps": (
            int(sample["min_remaining_steps"])
            if sample.get("min_remaining_steps") is not None
            else None
        ),
        "max_t0": (
            int(sample["max_t0"]) if sample.get("max_t0") is not None else None
        ),
        "suite_horizons": (
            {str(k): int(v) for k, v in dict(suite_horizons).items()}
            if suite_horizons
            else None
        ),
    }


def _checksum(keys: list[str]) -> str:
    content = json.dumps(
        keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _coverage_status(
    inventory: dict[str, Any], *, per_cell: int
) -> tuple[bool, list[dict[str, Any]]]:
    deficits = [
        {**cell, "missing": per_cell - int(cell["n"])}
        for cell in inventory["cells"]
        if int(cell["n"]) < per_cell
    ]
    return not deficits, deficits


def _coverage_error(
    inventory: dict[str, Any], *, per_cell: int, deficits: list[dict[str, Any]]
) -> str:
    preview = ", ".join(
        "/".join(
            f"{key}={value}"
            for key, value in cell.items()
            if key not in {"n", "missing"}
        )
        + f" (n={cell['n']}, missing={cell['missing']})"
        for cell in deficits[:8]
    )
    return (
        f"coverage gate failed: {len(deficits)}/{inventory['n_cells']} cells "
        f"cannot satisfy per_cell={per_cell}; minimum available per cell="
        f"{inventory['max_per_cell']}. Deficits: {preview}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ngc_w4_adequate_scale.yaml",
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-keys-json",
        type=Path,
        action="append",
        default=[],
        help="Exclude keys from a list or state-key artifact; repeatable",
    )
    parser.add_argument(
        "--exclude-episode-keys-json",
        type=Path,
        action="append",
        default=[],
        help="Exclude every state sharing an episode with any key; repeatable",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only write exact inventory stats (no sampling)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero when any requested cell is below per_cell",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = _load_config(config_path)
    sample = dict(cfg.get("sample") or {})
    pool_root = Path(args.pool or cfg.get("pool") or "pool/ngc_step1_scale200")
    if not pool_root.is_absolute():
        pool_root = (ROOT / pool_root).resolve()

    from rase.collect.state_pool import StatePool
    from rase.collect.stratified_sample import inventory_cell_counts, sample_stratified_keys

    pool = StatePool(pool_root)
    kwargs = _sample_kwargs(
        sample,
        extra_exclude_paths=args.exclude_keys_json,
        extra_exclude_episode_paths=args.exclude_episode_keys_json,
    )
    inventory = inventory_cell_counts(
        pool,
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"per_cell", "seed", "selection"}
        },
    )
    coverage_complete, deficit_cells = _coverage_status(
        inventory, per_cell=kwargs["per_cell"]
    )
    payload: dict[str, Any] = {
        "artifact_version": "rase-state-keys/v2",
        "pool": str(pool_root),
        "config": str(config_path),
        "sample": {
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"seed", "excluded_keys", "excluded_episode_keys"}
            },
            "sample_seed": kwargs["seed"],
            "excluded_keys": sorted(kwargs["excluded_keys"]),
            "excluded_episode_keys": sorted(kwargs["excluded_episode_keys"]),
        },
        "inventory": inventory,
        "state_keys": [],
        "state_keys_sha256": _checksum([]),
        "n_states": 0,
        "coverage_complete": coverage_complete,
        "required_per_cell": kwargs["per_cell"],
        "deficit_cells": deficit_cells,
    }
    if not args.inventory_only and coverage_complete:
        keys = sample_stratified_keys(pool, **kwargs)
        payload["state_keys"] = keys
        payload["state_keys_sha256"] = _checksum(keys)
        payload["n_states"] = len(keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_states": payload["n_states"],
                "inventory_total": inventory["total"],
                "max_per_cell": inventory["max_per_cell"],
                "coverage_complete": coverage_complete,
                "deficit_cells": deficit_cells,
                "state_keys_sha256": payload["state_keys_sha256"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"WROTE {args.output}", flush=True)
    if not coverage_complete and (args.require_complete or not args.inventory_only):
        raise SystemExit(
            _coverage_error(
                inventory, per_cell=kwargs["per_cell"], deficits=deficit_cells
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

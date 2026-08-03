#!/usr/bin/env python3
"""Sample ADEQUATE stratified keys and report pool inventory."""

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


def _state_keys_checksum(keys: list[str]) -> str:
    encoded = json.dumps(
        keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_kwargs(sample: dict[str, Any]) -> dict[str, Any]:
    suite_horizons = sample.get("suite_horizons")
    return {
        "per_cell": int(sample.get("per_cell", 2)),
        "seed": int(sample.get("sample_seed", 0)),
        "dims": tuple(sample.get("dims") or ("camera", "robot")),
        "suites": tuple(sample.get("suites") or ("Spatial", "Object", "Goal", "Long")),
        "levels": tuple(int(x) for x in (sample.get("levels") or (3, 4, 5))),
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
        "--inventory-only",
        action="store_true",
        help="Only write inventory stats (no sampling)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    sample = dict(cfg.get("sample") or {})
    pool_root = Path(args.pool or cfg.get("pool") or "pool/ngc_step1_scale200")
    if not pool_root.is_absolute():
        pool_root = (ROOT / pool_root).resolve()

    from rase.collect.state_pool import StatePool
    from rase.collect.stratified_sample import inventory_cell_counts, sample_stratified_keys

    pool = StatePool(pool_root)
    kwargs = _sample_kwargs(sample)
    inv_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in {"per_cell", "seed"}
    }
    inventory = inventory_cell_counts(pool, **inv_kwargs)

    payload: dict[str, Any] = {
        "pool": str(pool_root),
        "config": str(args.config.resolve()),
        "sample": {
            "strategy": sample.get("strategy", "stratified"),
            "per_cell": kwargs["per_cell"],
            "sample_seed": kwargs["seed"],
            "dims": list(kwargs["dims"]),
            "suites": list(kwargs["suites"]),
            "levels": list(kwargs["levels"]),
            "min_remaining_steps": kwargs["min_remaining_steps"],
            "max_t0": kwargs["max_t0"],
            "suite_horizons": kwargs["suite_horizons"],
        },
        "inventory": inventory,
        "state_keys": [],
        "n_states": 0,
    }

    if not args.inventory_only:
        if inventory["max_per_cell"] < kwargs["per_cell"]:
            raise SystemExit(
                f"pool cannot satisfy per_cell={kwargs['per_cell']}; "
                f"max_per_cell={inventory['max_per_cell']} "
                f"(total ADEQUATE={inventory['total']})"
            )
        keys = sample_stratified_keys(pool, **kwargs)
        payload["state_keys"] = keys
        payload["n_states"] = len(keys)
        payload["state_keys_sha256"] = _state_keys_checksum(keys)

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
                "output": str(args.output.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

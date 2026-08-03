#!/usr/bin/env python3
"""Collect NGC Step 1 snapshots through a policy/environment adapter."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rase.collect.pipeline import collect, load_config


def _apply_collection_overrides(
    config: dict[str, Any],
    *,
    episodes: int | None,
    seed: int | None,
    schedule_batch: int | None = None,
) -> dict[str, Any]:
    """Return a copied config with safe batch-level collection overrides."""
    result = copy.deepcopy(config)
    collection = result["collection"]
    if episodes is not None:
        if episodes <= 0:
            raise ValueError("--episodes must be positive")
        collection["episodes"] = int(episodes)
    if seed is not None:
        if seed < 0:
            raise ValueError("--seed must be non-negative")
        collection["seed"] = int(seed)
    if schedule_batch is not None:
        from rase.collect.pipeline import CLEAN_CONTROL_PROTOCOLS
        from rase.collect.w9b_schedule import BATCH_SIZES

        version = dict(result.get("protocol") or {}).get("version")
        if version not in CLEAN_CONTROL_PROTOCOLS:
            raise ValueError(
                "--schedule-batch is only valid for W9B/W9C clean-control protocols"
            )
        if episodes is not None or seed is not None:
            raise ValueError(
                "clean-control schedule batches forbid --episodes/--seed overrides"
            )
        if schedule_batch not in range(1, len(BATCH_SIZES) + 1):
            raise ValueError("--schedule-batch must be 1, 2, or 3")
        collection["schedule_batch_id"] = schedule_batch
        collection["episodes"] = BATCH_SIZES[schedule_batch - 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use deterministic synthetic episodes while exercising real storage",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="override collection.episodes for this append-only batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override collection.seed; use a fresh seed for each top-up batch",
    )
    parser.add_argument(
        "--schedule-batch",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="select one frozen W9B/W9C schedule batch; forbids seed/count overrides",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="also persist the collection summary as JSON",
    )
    args = parser.parse_args()
    config = _apply_collection_overrides(
        load_config(args.config),
        episodes=args.episodes,
        seed=args.seed,
        schedule_batch=args.schedule_batch,
    )
    summary = collect(config, force_dry_run=args.dry_run)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(rendered, encoding="utf-8")
        print(f"WROTE {args.summary_output}", flush=True)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()

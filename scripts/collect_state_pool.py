#!/usr/bin/env python3
"""Collect NGC Step 1 snapshots through a policy/environment adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pipeline import collect, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use deterministic synthetic episodes while exercising real storage",
    )
    args = parser.parse_args()
    print(json.dumps(collect(load_config(args.config), force_dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()

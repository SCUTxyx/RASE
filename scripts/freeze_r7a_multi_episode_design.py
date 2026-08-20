#!/usr/bin/env python3
"""Freeze the R7-A multi-episode-per-task development schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.r7_schedule import build_design  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026081207)
    args = parser.parse_args()
    base = json.loads(args.base_design.read_text())
    design = build_design(base, repeats_per_task=args.repeats_per_task, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: design[key] for key in (
        "status", "n_episodes", "n_tasks", "repeats_per_task",
        "n_unique_init_state_assignments", "suite_counts", "design_sha256",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

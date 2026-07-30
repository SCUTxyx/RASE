#!/usr/bin/env python3
"""Generate or verify the frozen W9B clean-control schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.collect.w9b_schedule import (  # noqa: E402
    generate_w9b_schedule,
    load_w9b_schedule,
    schedule_sha256,
    write_w9b_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = generate_w9b_schedule(args.seed)
    generated_sha = schedule_sha256(generated)
    if args.check:
        if not args.expected_sha256:
            raise SystemExit("--check requires --expected-sha256")
        existing = load_w9b_schedule(
            args.output, expected_sha256=args.expected_sha256
        )
        if existing != generated or generated_sha != args.expected_sha256:
            raise SystemExit("W9B frozen schedule differs from deterministic generation")
        print(f"W9B_SCHEDULE_OK sha256={generated_sha} rows=140", flush=True)
        return 0

    digest = write_w9b_schedule(args.output, generated)
    print(f"WROTE {args.output} sha256={digest} rows=140", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

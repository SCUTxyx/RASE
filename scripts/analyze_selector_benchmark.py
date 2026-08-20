#!/usr/bin/env python3
"""Analyze frozen selector benchmark JSONL without training or simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--shortcut-fallback-action",
        choices=("continue_smol", "escalate_oft", "abstain"),
        default="continue_smol",
        help="preregistered fallback used if train has no evaluable global majority",
    )
    args = parser.parse_args()

    from rase.selector.benchmark_analysis import analyze_selector_benchmark

    rows = []
    for path in args.dataset:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    splits = json.loads(args.splits.read_text()) if args.splits else None
    result = analyze_selector_benchmark(
        rows,
        splits=splits,
        success_reward=args.success_reward,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        shortcut_fallback_action=args.shortcut_fallback_action,
    )
    result["sources"] = [str(path.resolve()) for path in args.dataset]
    result["split_source"] = str(args.splits.resolve()) if args.splits else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

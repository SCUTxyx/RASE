#!/usr/bin/env python3
"""Freeze state keys from a dual-oracle summary split (e.g. oft_only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-oracle", type=Path, required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="Split name under summary['splits'], e.g. oft_only",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = json.loads(args.dual_oracle.resolve().read_text(encoding="utf-8"))
    splits = summary.get("splits") or {}
    if args.split not in splits:
        available = ", ".join(sorted(splits)) or "(none)"
        raise SystemExit(f"unknown split {args.split!r}; available: {available}")
    keys = sorted(str(key) for key in splits[args.split])
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        keys = keys[: args.limit]
    digest = hashlib.sha256(
        json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "artifact_version": "rase-state-keys/v2",
        "selection": f"dual_oracle_split:{args.split}",
        "source": str(args.dual_oracle.resolve()),
        "split": args.split,
        "n_states": len(keys),
        "state_keys_sha256": digest,
        "state_keys": keys,
        "warning": (
            "These keys come from a dual-oracle split, not retained-success "
            "sampling. Do not treat screen hits as retained-success positive control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_states": len(keys),
                "split": args.split,
                "state_keys_sha256": digest,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

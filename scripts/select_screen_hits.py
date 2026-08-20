#!/usr/bin/env python3
"""Freeze state keys with at least one successful candidate in screen summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def candidate_hit_counts(payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in payload.get("per_state") or []:
        key = str(state["state_key"])
        if key in counts:
            raise ValueError(f"duplicate state_key in screen summary: {key}")
        counts[key] = sum(
            int(candidate.get("successes", 0)) > 0
            for candidate in state.get("candidates") or []
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-candidate-hits", type=int, default=1)
    args = parser.parse_args()
    if args.min_candidate_hits < 1:
        raise SystemExit("--min-candidate-hits must be >= 1")

    hit_counts: dict[str, int] = {}
    sources = []
    for source in args.summary:
        path = source.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for key, hits in candidate_hit_counts(payload).items():
            hit_counts[key] = max(hit_counts.get(key, 0), hits)

    keys = sorted(
        key for key, hits in hit_counts.items() if hits >= args.min_candidate_hits
    )
    digest = hashlib.sha256(
        json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = {
        "schema_version": "rase-frozen-state-keys/v1",
        "selection": "screen_candidate_hits",
        "min_candidate_hits": args.min_candidate_hits,
        "sources": sources,
        "n_states": len(keys),
        "state_keys_sha256": digest,
        "candidate_hits_by_state": {key: hit_counts[key] for key in keys},
        "state_keys": keys,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"n_states": len(keys), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

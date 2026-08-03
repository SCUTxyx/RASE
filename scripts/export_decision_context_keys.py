#!/usr/bin/env python3
"""Export a frozen inventory of state-pool keys supporting strict CONTINUE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def select_records(
    records: list[dict],
    *,
    step: int | None = None,
    one_per_episode: bool = False,
    max_states: int | None = None,
) -> list[dict]:
    selected = [row for row in records if step is None or int(row["step"]) == step]
    if one_per_episode:
        by_episode = {}
        for row in selected:
            by_episode.setdefault(str(row["episode_id"]), row)
        selected = list(by_episode.values())
    if max_states is not None:
        # Round-robin by task, keeping temporal order within each task.
        by_task = {}
        for row in selected:
            by_task.setdefault(row["task_id"], []).append(row)
        limited = []
        while len(limited) < max_states and any(by_task.values()):
            for task_id in sorted(by_task):
                if by_task[task_id] and len(limited) < max_states:
                    limited.append(by_task[task_id].pop(0))
        selected = limited
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--one-per-episode", action="store_true")
    parser.add_argument("--expected-states", type=int, default=None)
    parser.add_argument(
        "--include-key",
        action="append",
        default=[],
        help="Export only this exact key; repeatable and order-preserving.",
    )
    args = parser.parse_args()
    if args.max_states is not None and args.max_states < 1:
        raise SystemExit("--max-states must be positive")
    if args.step is not None and args.step < 0:
        raise SystemExit("--step must be non-negative")
    if args.expected_states is not None and args.expected_states < 1:
        raise SystemExit("--expected-states must be positive")

    from rase.collect.state_pool import StatePool
    from rase.interventions.decision_context import strict_continue_suffix

    pool = StatePool(args.pool.resolve())
    records = []
    rejected = []
    for key in pool.manifest().get("states", {}):
        loaded = pool.read_state(key, load_observations=False)
        try:
            suffix = strict_continue_suffix(loaded.controller_state)
        except ValueError as exc:
            rejected.append({"state_key": key, "reason": str(exc)})
            continue
        records.append(
            {
                "state_key": key,
                "task_id": loaded.metadata.task_id,
                "episode_id": loaded.metadata.episode_id,
                "step": loaded.metadata.step,
                "suffix_steps": len(suffix),
                "suite": loaded.metadata.suite,
                "perturbation_dimension": loaded.metadata.perturb_dim,
                "perturbation_level": loaded.metadata.level,
            }
        )
    records.sort(key=lambda row: (row["task_id"], row["episode_id"], row["step"]))
    if args.include_key:
        by_key = {str(row["state_key"]): row for row in records}
        missing = [key for key in args.include_key if key not in by_key]
        if missing:
            raise SystemExit(f"requested keys lack valid decision context: {missing}")
        if len(args.include_key) != len(set(args.include_key)):
            raise SystemExit("--include-key values must be unique")
        records = [by_key[key] for key in args.include_key]
    else:
        records = select_records(
            records,
            step=args.step,
            one_per_episode=args.one_per_episode,
            max_states=args.max_states,
        )
    if args.expected_states is not None and len(records) != args.expected_states:
        raise SystemExit(
            f"selected {len(records)} states, expected {args.expected_states}"
        )
    keys = [str(row["state_key"]) for row in records]
    payload = {
        "artifact_version": "rase-decision-context-keys/v1",
        "pool": str(args.pool.resolve()),
        "state_keys": keys,
        "state_keys_sha256": _checksum(keys),
        "n_states": len(keys),
        "selection": {
            "step": args.step,
            "one_per_episode": args.one_per_episode,
            "max_states": args.max_states,
            "expected_states": args.expected_states,
        },
        "records": records,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_states": len(keys), "n_rejected": len(rejected)}), flush=True)
    return 0 if keys else 2


if __name__ == "__main__":
    raise SystemExit(main())

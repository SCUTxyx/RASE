#!/usr/bin/env python3
"""Inventory success/failure episode-group support across local state pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _pool_stats(pool_root: Path) -> dict:
    manifest_path = pool_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outcomes: Counter[str] = Counter()
    episode_groups: dict[str, set[tuple[str, str]]] = {
        "success": set(),
        "failure": set(),
    }
    cells: Counter[tuple[str, str, int, str]] = Counter()
    metadata_errors = 0
    for key, entry in (manifest.get("states") or {}).items():
        meta_path = pool_root / str(entry["path"]) / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            outcome = str(meta.get("episode_outcome") or entry.get("outcome") or "unknown")
            outcomes[outcome] += 1
            task = str(meta.get("task_id") or entry.get("task_id") or "")
            episode = str(meta.get("episode_id") or entry.get("episode_id") or "")
            if outcome in episode_groups and task and episode:
                episode_groups[outcome].add((task, episode))
            cells[(
                str(meta.get("suite") or "unknown"),
                str(meta.get("perturb_dim") or "unknown"),
                int(meta.get("level") or 0),
                outcome,
            )] += 1
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            metadata_errors += 1
    return {
        "pool": str(pool_root),
        "n_states": sum(outcomes.values()),
        "state_outcomes": dict(sorted(outcomes.items())),
        "success_episode_groups": len(episode_groups["success"]),
        "failure_episode_groups": len(episode_groups["failure"]),
        "metadata_errors": metadata_errors,
        "success_cells": [
            {
                "suite": suite,
                "dim": dim,
                "level": level,
                "n_states": count,
            }
            for (suite, dim, level, outcome), count in sorted(cells.items())
            if outcome == "success"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=Path("pool"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = sorted(args.pool_root.glob("*/manifest.json"))
    pools = [_pool_stats(path.parent) for path in manifests]
    total_success_groups = sum(row["success_episode_groups"] for row in pools)
    result = {
        "schema_version": "rase-selector-pool-support/v1",
        "status": "complete",
        "n_pools": len(pools),
        "total_success_episode_groups_un_deduplicated": total_success_groups,
        "pools": pools,
        "recommendation": (
            "reuse_success_groups_after_cross_pool_dedup_audit"
            if total_success_groups >= 30
            else "collect_new_success_retained_control_episodes"
        ),
        "warning": (
            "Counts are summed across pools and may duplicate task/episode seeds; "
            "perform a cross-pool group audit before constructing splits."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_pools": len(pools),
        "total_success_episode_groups_un_deduplicated": total_success_groups,
        "recommendation": result["recommendation"],
        "per_pool": [
            {
                "pool": row["pool"],
                "success_states": row["state_outcomes"].get("success", 0),
                "success_episode_groups": row["success_episode_groups"],
            }
            for row in pools
        ],
    }, indent=2), flush=True)
    print(f"WROTE {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

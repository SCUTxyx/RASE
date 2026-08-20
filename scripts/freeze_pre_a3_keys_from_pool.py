#!/usr/bin/env python3
"""Join a collected PRE-A3 pool onto the frozen design without using outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rase.collect.pre_a3 import freeze_keys_from_pool_records
from rase.collect.state_pool import StatePool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    pool = StatePool(args.pool.resolve())
    manifest = pool.manifest()
    records = []
    for state_key, entry in (manifest.get("states") or {}).items():
        if int(entry.get("step", -1)) != 0:
            continue
        loaded = pool.read_state(state_key, load_observations=False)
        meta = loaded.metadata
        controller = (
            loaded.controller_state
            if isinstance(loaded.controller_state, dict)
            else {}
        )
        timestep = int(controller.get("env_counters", {}).get("timestep", meta.step))
        records.append(
            {
                "state_key": state_key,
                "task_id": meta.task_id,
                "episode_id": meta.episode_id,
                "suite": meta.suite,
                "perturbation_dimension": meta.perturb_dim,
                "perturbation_level": meta.level,
                "step": meta.step,
                "snapshot_policy_step": meta.step,
                "snapshot_simulator_timestep": timestep,
            }
        )

    frozen = freeze_keys_from_pool_records(
        records,
        design=design,
        pool=str(args.pool),
    )
    for row in frozen["records"]:
        row["logical_task_id"] = row["task_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen keys: {args.output}")
    args.output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_states": frozen["n_states"],
                "split_counts": frozen["split_counts"],
                "state_keys_sha256": frozen["state_keys_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

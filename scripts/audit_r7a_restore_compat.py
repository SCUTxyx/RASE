#!/usr/bin/env python3
"""Restore one frozen R7 state per suite/perturbation cell before labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import restore_pool_state
    from rase.collect.policy_step import current_timestep
    from rase.collect.state_pool import StatePool

    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()
    manifest = json.loads(args.initial_keys.read_text())
    selected = {}
    for row in manifest["records"]:
        selected.setdefault((row["suite"], row["perturb_dim"]), row)
    expected_cells = {
        (suite, perturb) for suite in ("Spatial", "Object", "Goal", "Long")
        for perturb in ("clean", "camera", "robot")
    }
    if set(selected) != expected_cells:
        raise ValueError(f"missing restore cells: {sorted(expected_cells - set(selected))}")

    pool = StatePool(Path(manifest["pool"]))
    records = []
    for cell, row in sorted(selected.items()):
        restored = restore_pool_state(
            pool, row["state_key"],
            libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
        )
        try:
            records.append({
                "suite": cell[0], "perturb_dim": cell[1],
                "state_key": row["state_key"],
                "init_state_id": row["init_state_id"],
                "restored_timestep": current_timestep(restored.handle.control_env),
                "status": "PASS",
            })
        finally:
            restored.close()
    result = {
        "schema_version": "rase-r7a-restore-compat/v1",
        "status": "PASS", "cells": len(records), "records": records,
        "note": (
            "StatePool v1 compatibility view drops full-controller runtime caches "
            "when raw mujoco_data is absent; stored pool bytes remain unchanged."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

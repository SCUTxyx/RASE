#!/usr/bin/env python3
"""Freeze reset-state keys for the development-only replacement audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def freeze_initial_keys(
    pool_root: Path,
    design: dict[str, Any],
    *,
    expected_reset_simulator_timestep: int,
) -> dict[str, Any]:
    from rase.collect.state_pool import StatePool

    if design.get("schema_version") != "rase-independent-factorial-design/v1":
        raise ValueError("unexpected source-design schema")
    if design.get("status") != "ready":
        raise ValueError("source design is not ready")
    planned = {
        str(row["episode_id"]): dict(row) for row in design.get("records") or []
    }
    if len(planned) != int(design.get("n_requests", -1)):
        raise ValueError("source design has duplicate episode ids")

    pool = StatePool(pool_root.resolve())
    records = []
    for key, manifest_row in pool.manifest().get("states", {}).items():
        # Avoid checksumming every later trajectory snapshot when only the
        # reset-state bundle is in scope for this audit.
        if int(manifest_row["step"]) != 0:
            continue
        loaded = pool.read_state(key, load_observations=False)
        meta = loaded.metadata
        if meta.step != 0:
            raise ValueError(f"manifest/bundle step mismatch for {key}")
        row = planned.get(meta.episode_id)
        if row is None:
            raise ValueError(f"unplanned episode in initial pool: {meta.episode_id}")
        observed_identity = (
            meta.task_id,
            meta.suite,
            meta.perturb_dim,
            meta.level,
        )
        planned_identity = (
            str(row["task_id"]),
            str(row["suite"]),
            str(row["dimension"]),
            int(row["level"]),
        )
        if observed_identity != planned_identity:
            raise ValueError(
                f"design identity mismatch for {meta.episode_id}: "
                f"{observed_identity} != {planned_identity}"
            )
        simulator_timestep = int(
            loaded.controller_state["env_counters"]["timestep"]
        )
        if simulator_timestep != expected_reset_simulator_timestep:
            raise ValueError(
                f"initial snapshot has unexpected post-reset simulator timestep "
                f"for {meta.episode_id}: observed={simulator_timestep} "
                f"expected={expected_reset_simulator_timestep}"
            )
        records.append(
            {
                "state_key": key,
                "task_id": meta.task_id,
                "episode_id": meta.episode_id,
                "suite": meta.suite,
                "step": meta.step,
                "snapshot_policy_step": 0,
                "snapshot_simulator_timestep": simulator_timestep,
                "perturbation_dimension": meta.perturb_dim,
                "perturbation_level": meta.level,
                "source_only_success": meta.episode_outcome == "success",
                "source_policy_seed": meta.seed,
                "design_request_index": int(row["request_index"]),
            }
        )
    records.sort(key=lambda row: int(row["design_request_index"]))
    keys = [str(row["state_key"]) for row in records]
    tasks = [str(row["task_id"]) for row in records]
    episodes = [str(row["episode_id"]) for row in records]
    expected = int(design["n_requests"])
    if len(records) != expected:
        raise ValueError(f"expected {expected} reset states, observed {len(records)}")
    if len(tasks) != len(set(tasks)) or len(episodes) != len(set(episodes)):
        raise ValueError("replacement cohort must have unique tasks and episodes")
    return {
        "artifact_version": "rase-replacement-initial-keys/v1",
        "status": "frozen",
        "purpose": "development-only replacement audit; excluded from hidden test",
        "selection_uses_outcomes": False,
        "reset_semantics": {
            "snapshot_policy_step": 0,
            "source_actions_before_snapshot": 0,
            "expected_post_reset_simulator_timestep": (
                expected_reset_simulator_timestep
            ),
            "note": (
                "LIBERO reset performs internal initialization steps; the simulator "
                "counter is therefore nonzero before either policy acts."
            ),
        },
        "pool": str(pool_root.resolve()),
        "n_states": len(records),
        "state_keys": keys,
        "state_keys_sha256": _checksum(keys),
        "n_unique_tasks": len(set(tasks)),
        "n_unique_episodes": len(set(episodes)),
        "source_only_successes": sum(row["source_only_success"] for row in records),
        "source_outcome_counts": dict(
            sorted(
                Counter(
                    "success" if row["source_only_success"] else "failure"
                    for row in records
                ).items()
            )
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-reset-simulator-timestep", type=int, required=True
    )
    args = parser.parse_args()
    result = freeze_initial_keys(
        args.pool,
        _read(args.design.resolve()),
        expected_reset_simulator_timestep=args.expected_reset_simulator_timestep,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_states": result["n_states"],
                "source_only_successes": result["source_only_successes"],
                "state_keys_sha256": result["state_keys_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

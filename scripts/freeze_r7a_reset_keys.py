#!/usr/bin/env python3
"""Audit and freeze the R7 independent reset-state manifest.

The reset pool deliberately writes a schema-placeholder ``episode_outcome``.
This script never consumes that field as a label.  It joins every stored state
to the frozen design by episode ID, checks exact task/init/seed provenance and
step zero, verifies StatePool readability, and emits keys for downstream
source-only rollout collection.  All four episodes of a task remain in one
outer task fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=192)
    parser.add_argument("--collection-seed", type=int, default=2_026_081_207)
    args = parser.parse_args()

    from rase.collect.r7_schedule import load_design, requests_from_design
    from rase.collect.state_pool import StatePool

    pool_path = args.pool.resolve()
    design_path = args.design.resolve()
    design = load_design(design_path)
    expected_requests = requests_from_design(design, seed=args.collection_seed)
    expected = {str(request.episode_id): request for request in expected_requests}

    meta_by_episode: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(pool_path.rglob("meta.json")):
        row = json.loads(path.read_text())
        row["_meta_path"] = str(path.resolve())
        meta_by_episode[str(row["episode_id"])].append(row)

    errors: list[dict] = []
    records: list[dict] = []
    state_keys: set[str] = set()
    state_pool = StatePool(pool_path)
    for episode_id, request in expected.items():
        rows = meta_by_episode.get(episode_id, [])
        if len(rows) != 1:
            errors.append({"episode_id": episode_id, "reason": "state_count", "observed": len(rows)})
            continue
        row = rows[0]
        design_row = design["records"][int(request.index)]
        task_id = str(design_row["task_id"])
        observed = {
            "task_id": str(row.get("task_id")),
            "suite": str(row.get("suite")),
            "init_state_id": int(row.get("init_state_id", -1)),
            "seed": int(row.get("seed", -1)),
            "step": int(row.get("step", -1)),
        }
        wanted = {
            "task_id": task_id,
            "suite": str(request.suite),
            "init_state_id": int(request.init_state_id),
            "seed": int(request.seed),
            "step": 0,
        }
        if observed != wanted:
            errors.append({
                "episode_id": episode_id, "reason": "provenance_mismatch",
                "observed": observed, "expected": wanted,
            })
            continue
        state_key = str(row["state_key"])
        if state_key in state_keys:
            errors.append({"episode_id": episode_id, "reason": "duplicate_state_key", "state_key": state_key})
            continue
        try:
            restored = state_pool.read_state(state_key, load_observations=False)
        except Exception as error:  # noqa: BLE001
            errors.append({"episode_id": episode_id, "reason": "unreadable", "error": str(error)})
            continue
        if int(restored.metadata.step) != 0:
            errors.append({"episode_id": episode_id, "reason": "restored_nonzero_step"})
            continue
        state_keys.add(state_key)
        records.append({
            "role": "natural_development_source_risk",
            "task_id": task_id,
            "task_cluster_id": task_id,
            "state_key": state_key,
            "suite": str(request.suite),
            "perturb_dim": str(request.dimension),
            "perturb_level": int(request.level),
            "step": 0,
            "episode_id": episode_id,
            "episode_repeat": int(design_row["episode_repeat"]),
            "init_state_id": int(request.init_state_id),
            "pool_seed": int(request.seed),
            "note": "independent reset state; pool episode_outcome is not a label",
        })

    unexpected = sorted(set(meta_by_episode) - set(expected))
    if unexpected:
        errors.append({"reason": "unexpected_episode_ids", "episode_ids": unexpected})
    if errors or len(records) != args.expected_episodes:
        raise SystemExit(json.dumps({
            "status": "FAIL", "records": len(records),
            "expected": args.expected_episodes, "errors": errors,
        }, indent=2, sort_keys=True))

    records.sort(key=lambda row: (row["suite"], row["task_id"], row["episode_repeat"]))
    by_task = Counter(row["task_id"] for row in records)
    if len(by_task) != 48 or set(by_task.values()) != {int(design["repeats_per_task"])}:
        raise SystemExit("R7 reset pool is not balanced at four independent episodes per task")
    ordered_keys = [row["state_key"] for row in records]
    payload = {
        "schema_version": "rase-r7a-reset-keys/v1",
        "status": "frozen",
        "scientific_scope": (
            "development-only independent reset states for source-risk learning; "
            "all episodes of a task share an outer fold"
        ),
        "pool": str(pool_path),
        "pool_episode_outcome_is_label": False,
        "selection_uses_outcomes": False,
        "design": str(design_path),
        "design_file_sha256": file_sha256(design_path),
        "design_sha256": str(design["design_sha256"]),
        "n_records": len(records),
        "n_tasks": len(by_task),
        "episodes_per_task": int(design["repeats_per_task"]),
        "suite_counts": dict(Counter(row["suite"] for row in records)),
        "cell_counts": dict(Counter(
            f"{row['suite']}|{row['perturb_dim']}:L{row['perturb_level']}" for row in records
        )),
        "state_keys": ordered_keys,
        "state_keys_sha256": canonical_sha256(ordered_keys),
        "records": records,
        "boundaries_to_collect": [0, 8, 16],
        "seed_plan": {
            "pi0fast_libero": {
                "development_policy_seeds": [0],
                "note": "add seed 1 only after source-risk representation gate passes",
            }
        },
        "locked": {
            "oft_counterfactual_collection": True,
            "selector_training": True,
            "world_model_features": True,
            "independent_validation_and_test": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "output": str(args.output),
        "n_records": payload["n_records"], "n_tasks": payload["n_tasks"],
        "suite_counts": payload["suite_counts"],
        "state_keys_sha256": payload["state_keys_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

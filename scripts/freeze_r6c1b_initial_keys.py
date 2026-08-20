#!/usr/bin/env python3
"""R6-C.1B-1: freeze the R6-C.1B candidate initial-state manifest.

Selection is STRICTLY from pre-registered pool metadata (suite, perturbation
dimension/level, episode step, task); **no R6-C outcome is used to choose the
natural-development evaluation cohort**.  The manifest marks every candidate
as either ``train_enrichment`` (source-only screening results may promote a
candidate to be collected with OFT labels for training) or
``natural_development_eval`` (frozen, may not be over-sampled; used for the
natural-distribution gate).

Design (per the R6-C rework plan, section R6-C.1B):

- Every state belongs to exactly one ``task_id``; all states, seeds and
  replicas of the same task are bound to the same outer OOF fold.
- ``natural_development_eval``: one *new* un-collected pool snapshot per task
  (different episode step / perturbation than the R6-A 48 states), evaluated
  under the new seed plan and later merged with the existing B1.2 cohort.
- ``train_enrichment``: additional per-task pool candidates (more steps,
  perturbation levels); source-only screening below decides which of these get
  OFT labels and enter training.
- Availability is validated with ``StatePool.read_state``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_order(suite: str) -> int:
    return {"Spatial": 0, "Object": 1, "Goal": 2, "Long": 3}[suite]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--initial-keys-v2", type=Path, required=True,
                        help="frozen R6-A replacement48 initial keys (marks already-collected states)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-new-states-per-task", type=int, default=1)
    parser.add_argument("--enrichment-states-per-task", type=int, default=2)
    args = parser.parse_args()

    from rase.collect.state_pool import StatePool

    pool = StatePool(args.pool.resolve())
    used = set(json.loads(args.initial_keys_v2.read_text())["state_keys"])
    collected = set(used)  # all R6-A 48 states already collected for seeds 0/1

    # Enumerate pool candidates (pre-registered metadata only).
    candidates: dict[str, dict] = {}
    for meta in sorted(args.pool.rglob("meta.json")):
        data = json.loads(meta.read_text())
        candidates[str(data["state_key"])] = data

    # Group unused candidates by task, prefer low-step and robot/camera perturbs.
    by_task: dict[str, list[dict]] = {}
    for key, data in candidates.items():
        if key in collected:
            continue
        by_task.setdefault(str(data["task_id"]), []).append(data)
    for task in by_task:
        by_task[task].sort(key=lambda d: (
            suite_order(str(d["suite"])),
            int(d["step"]),
            0 if str(d["perturb_dim"]) in ("robot", "camera") else 1,
            int(d.get("level", 0)),
            str(d["state_key"]),
        ))

    used_tasks: set[str] = set()
    records: list[dict] = []
    for task in sorted(by_task):
        members = by_task[task]
        # natural eval: first unused candidate (lowest step, robot/camera first)
        eval_pick = members[: args.eval_new_states_per_task]
        rest = members[args.eval_new_states_per_task:]
        for position, data in enumerate(eval_pick):
            records.append({
                "role": "natural_development_eval",
                "task_id": str(data["task_id"]),
                "state_key": str(data["state_key"]),
                "suite": str(data["suite"]),
                "perturb_dim": str(data["perturb_dim"]),
                "perturb_level": int(data.get("level", 0)),
                "step": int(data["step"]),
                "episode_id": str(data["episode_id"]),
                "pool_seed": int(data["seed"]),
                "note": "new pool snapshot for the natural-development eval cohort",
            })
            used_tasks.add(str(data["task_id"]))
        # train enrichment: next candidates
        for data in rest[: args.enrichment_states_per_task]:
            records.append({
                "role": "train_enrichment",
                "task_id": str(data["task_id"]),
                "state_key": str(data["state_key"]),
                "suite": str(data["suite"]),
                "perturb_dim": str(data["perturb_dim"]),
                "perturb_level": int(data.get("level", 0)),
                "step": int(data["step"]),
                "episode_id": str(data["episode_id"]),
                "pool_seed": int(data["seed"]),
                "note": "screening candidate; OFT labels only if screening keeps it",
            })

    # Availability validation via StatePool.read_state.
    unavailable = []
    for record in records:
        try:
            pool.read_state(record["state_key"], load_observations=False)
        except Exception as error:  # noqa: BLE001
            unavailable.append({"state_key": record["state_key"], "error": str(error)})
    if unavailable:
        raise SystemExit(json.dumps({"unavailable_states": unavailable}, indent=2))

    eval_tasks = sorted({r["task_id"] for r in records if r["role"] == "natural_development_eval"})
    enrichment_tasks = sorted({r["task_id"] for r in records if r["role"] == "train_enrichment"})
    state_keys = [r["state_key"] for r in records]
    payload = {
        "schema_version": "rase-r6c1b-initial-keys/v1",
        "status": "frozen",
        "scientific_scope": ("R6-C.1B candidate freeze: natural eval cohort from pre-registered "
                             "pool metadata only (no outcome-based selection); train enrichment "
                             "candidates reserved for source-only screening"),
        "pool": str(args.pool.resolve()),
        "selection_uses_outcomes": False,
        "n_records": len(records),
        "n_eval_states": len([r for r in records if r["role"] == "natural_development_eval"]),
        "n_enrichment_states": len([r for r in records if r["role"] == "train_enrichment"]),
        "n_eval_tasks": len(eval_tasks),
        "n_enrichment_tasks": len(enrichment_tasks),
        "suites": sorted({r["suite"] for r in records}),
        "state_keys": state_keys,
        "state_keys_sha256": hashlib.sha256(json.dumps(state_keys).encode()).hexdigest(),
        "records": records,
        "boundaries_to_collect": [0, 8, 16],
        "seed_plan": {
            "pi05_libero": {"new_seeds": [2, 3], "note": "seed 0/1 already collected on R6-A states"},
            "pi0fast_libero": {"new_seeds": [1], "note": "seed 0 already collected"},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["manifest_sha256"] = sha256(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

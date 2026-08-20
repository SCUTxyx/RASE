#!/usr/bin/env python3
"""Freeze the independent task-disjoint validation state manifest for R6-E.

R6-E validation states must be task-disjoint from the R6-C training task set
(the 48 frozen B1.2 tasks in ``runs/rase_ui_phase1a_replacement48_initial_keys_v2.json``).
The validation cohort is >= 100 states across the four LIBERO suites, sampled
from a frozen state pool built for validation only (never used for R6-C).

This exporter is intentionally read-only over the pool: it selects step-0
snapshots whose ``task_id`` is outside the R6-C training task set, verifies the
task-disjoint constraint and the minimum cohort size, and writes a frozen keys
manifest with a state-key checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R6C_TRAIN_KEYS = ROOT / "runs/rase_ui_phase1a_replacement48_initial_keys_v2.json"
SUITE_ORDER = ["Spatial", "Object", "Goal", "Long"]


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True,
                        help="frozen validation-only state pool root (StatePool layout)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-states", type=int, default=100)
    parser.add_argument("--per-suite", type=int, nargs="+", default=None,
                        help="target states per suite in SUITE_ORDER order")
    args = parser.parse_args()

    train = _load(R6C_TRAIN_KEYS)
    train_tasks = {str(row["task_id"]) for row in train["records"]}
    manifest = _load(args.pool / "manifest.json")
    states = manifest["states"]

    from rase.collect.state_pool import StatePool

    pool = StatePool(args.pool.resolve())
    candidates: list[dict] = []
    for key, row in states.items():
        if int(row["step"]) != 0:
            continue
        task_id = str(row.get("task_id", ""))
        if task_id in train_tasks:
            continue
        meta = pool.read_state(str(key), load_observations=False).metadata
        suite = str(meta.suite)
        candidates.append({
            "state_key": str(key), "task_id": task_id, "suite": suite,
            "episode_id": str(row.get("episode_id", "")),
            "step": int(row["step"]),
        })
    if len(candidates) < args.min_states:
        raise SystemExit(
            f"only {len(candidates)} task-disjoint step-0 candidates; need {args.min_states}")

    per_suite = {suite: [] for suite in SUITE_ORDER}
    for row in candidates:
        if row["suite"] in per_suite:
            per_suite[row["suite"]].append(row)
    target = args.per_suite or [args.min_states // len(SUITE_ORDER)] * len(SUITE_ORDER)
    selected: list[dict] = []
    for suite, count in zip(SUITE_ORDER, target):
        selected.extend(sorted(per_suite[suite], key=lambda r: r["state_key"])[:count])
    if len(selected) < args.min_states:
        raise SystemExit(f"selected {len(selected)} < {args.min_states}")

    keys = [row["state_key"] for row in selected]
    assert len(keys) == len(set(keys)), "duplicate validation state keys"
    selected_tasks = {row["task_id"] for row in selected}
    overlap = selected_tasks & train_tasks
    if overlap:
        raise SystemExit(f"validation tasks overlap R6-C training tasks: {sorted(overlap)}")

    from collections import Counter as C

    payload = {
        "schema_version": "rase-r6e-validation-initial-keys/v1",
        "status": "frozen",
        "purpose": "independent task-disjoint validation for R6-E closed-loop evaluation",
        "n_states": len(keys),
        "n_unique_tasks": len(selected_tasks),
        "pool": str(args.pool.resolve()),
        "suite_counts": dict(C(row["suite"] for row in selected)),
        "r6c_train_tasks": sorted(train_tasks),
        "state_keys_sha256": _checksum(keys),
        "state_keys": keys,
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ["n_states", "n_unique_tasks", "suite_counts", "state_keys_sha256"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

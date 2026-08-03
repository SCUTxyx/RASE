#!/usr/bin/env python3
"""Audit a collected factorial calibration pool before intervention rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cell(row: dict[str, Any]) -> str:
    return f"{row['suite']}|{row['dimension']}:L{row['level']}"


def audit_records(
    records: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_episode[str(row["episode_id"])].append(row)
    inconsistent = []
    episodes = []
    for episode_id, rows in sorted(by_episode.items()):
        identity = {
            (
                row["suite"],
                row["dimension"],
                int(row["level"]),
                row["task_id"],
                row["outcome"],
            )
            for row in rows
        }
        if len(identity) != 1:
            inconsistent.append(episode_id)
            continue
        suite, dimension, level, task_id, outcome = next(iter(identity))
        episodes.append(
            {
                "episode_id": episode_id,
                "suite": suite,
                "dimension": dimension,
                "level": level,
                "task_id": task_id,
                "outcome": outcome,
                "n_states": len(rows),
                "steps": sorted(int(row["step"]) for row in rows),
            }
        )

    observed_cells = Counter(_cell(row) for row in episodes)
    expected_cells = Counter(_cell(row) for row in expected)
    task_counts = Counter(str(row["task_id"]) for row in episodes)
    by_cell = {}
    for cell in sorted(set(observed_cells) | set(expected_cells)):
        selected = [row for row in episodes if _cell(row) == cell]
        successes = sum(row["outcome"] == "success" for row in selected)
        by_cell[cell] = {
            "expected_episodes": expected_cells[cell],
            "observed_episodes": observed_cells[cell],
            "source_successes": successes,
            "source_success_rate": successes / len(selected) if selected else None,
            "n_states": sum(row["n_states"] for row in selected),
            "n_unique_tasks": len({row["task_id"] for row in selected}),
        }
    design_matches = observed_cells == expected_cells
    return {
        "status": (
            "ready"
            if design_matches and not inconsistent and len(episodes) == len(expected)
            else "not_ready"
        ),
        "n_expected_episodes": len(expected),
        "n_observed_episodes": len(episodes),
        "n_states": len(records),
        "n_unique_tasks": len(task_counts),
        "duplicate_task_ids": sorted(
            task_id for task_id, count in task_counts.items() if count > 1
        ),
        "inconsistent_episode_ids": inconsistent,
        "design_matches": design_matches,
        "source_outcomes": dict(sorted(Counter(row["outcome"] for row in episodes).items())),
        "snapshot_step_counts": dict(
            sorted(Counter(row["step"] for row in records).items())
        ),
        "by_cell": by_cell,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from rase.collect.perturb_sampler import sample_perturbations
    from rase.collect.state_pool import StatePool
    from rase.interventions.decision_context import strict_continue_suffix

    config = json.loads(args.config.read_text(encoding="utf-8"))
    collection = config["collection"]
    sampling = config["sampling"]
    requests = sample_perturbations(
        int(collection["episodes"]),
        int(collection["seed"]),
        suite_quotas=sampling.get("suite_quotas"),
        factorial_cells=sampling.get("factorial_cells"),
    )
    expected = [
        {
            "suite": row.suite,
            "dimension": row.dimension,
            "level": row.level,
        }
        for row in requests
    ]
    pool = StatePool(args.pool.resolve())
    records = []
    strict_count = 0
    strict_rejected = []
    for state_key in pool.manifest().get("states", {}):
        loaded = pool.read_state(state_key, load_observations=False)
        metadata = loaded.metadata
        records.append(
            {
                "state_key": state_key,
                "episode_id": metadata.episode_id,
                "suite": metadata.suite,
                "dimension": metadata.perturb_dim,
                "level": metadata.level,
                "task_id": metadata.task_id,
                "outcome": metadata.episode_outcome,
                "step": metadata.step,
            }
        )
        try:
            strict_continue_suffix(loaded.controller_state)
            strict_count += 1
        except ValueError as error:
            strict_rejected.append({"state_key": state_key, "reason": str(error)})
    result = audit_records(records, expected)
    result.update(
        {
            "schema_version": "rase-factorial-pool-audit/v1",
            "pool": str(args.pool.resolve()),
            "n_strict_continue_states": strict_count,
            "strict_continue_rejected": strict_rejected,
        }
    )
    output = args.output.resolve()
    _write_json(output, result)
    print(json.dumps({"output": str(output), **result}, sort_keys=True), flush=True)
    return 0 if result["status"] == "ready" and not strict_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())

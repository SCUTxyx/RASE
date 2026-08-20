#!/usr/bin/env python3
"""Generate a task-disjoint train design for PRE-C0-R4-D with >=300 states.

Design goals:
- >=300 task-disjoint boundary states across 4 suites (spatial, object, goal, 10)
- At least 6 validation tasks (1-2 per suite, held out from training)
- Balanced rescue/harm/neutral coverage
- Suite identity encoding for cross-task transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# All available tasks in the 4 LIBERO suites (from task_classification.json)
ALL_TASKS = {
    "libero_spatial": [f"libero_spatial_task{i:02d}" for i in range(1, 11)],
    "libero_object": [f"libero_object_task{i:02d}" for i in range(1, 11)],
    "libero_goal": [f"libero_goal_task{i:02d}" for i in range(1, 11)],
    "libero_10": [f"libero_10_task{i:02d}" for i in range(1, 11)],
}

# Boundaries to collect at (in OFT steps)
BOUNDARIES = [0, 8, 16, 32, 64, 96, 128]


def generate_design(
    validation_per_suite: int = 2,
    validation_offset: int = 0,
    target_train_states: int = 300,
    states_per_train_task: int = 15,
    output: Path | None = None,
) -> dict:
    """Generate a task-disjoint train/validation split.

    Args:
        validation_per_suite: Number of validation tasks per suite.
        validation_offset: Offset into task list for selecting validation tasks.
        target_train_states: Target total boundary states for training.
        states_per_train_task: Approximate states per train task (used to
            verify feasibility, not a hard constraint at design time).
        output: If provided, write design JSON to this path.
    """
    train_tasks: dict[str, list[str]] = {}
    validation_tasks: dict[str, list[str]] = {}

    for suite, tasks in ALL_TASKS.items():
        tasks_sorted = sorted(tasks)
        n_val = min(validation_per_suite, len(tasks_sorted) - 2)  # keep >=2 train tasks
        val_start = validation_offset % len(tasks_sorted)
        val_indices = {(val_start + i) % len(tasks_sorted) for i in range(n_val)}
        validation_tasks[suite] = [tasks_sorted[i] for i in sorted(val_indices)]
        train_tasks[suite] = [t for i, t in enumerate(tasks_sorted) if i not in val_indices]

    all_train = [t for suite_tasks in train_tasks.values() for t in suite_tasks]
    all_validation = [t for suite_tasks in validation_tasks.values() for t in suite_tasks]

    estimated_states = len(all_train) * states_per_train_task
    feasible = estimated_states >= target_train_states

    design = {
        "schema_version": "rase-pre-c0-r4d-train-design/v1",
        "suites": list(ALL_TASKS.keys()),
        "boundaries": BOUNDARIES,
        "train_tasks": all_train,
        "train_tasks_by_suite": train_tasks,
        "train_task_count": len(all_train),
        "validation_tasks": all_validation,
        "validation_tasks_by_suite": validation_tasks,
        "validation_task_count": len(all_validation),
        "target_train_states": target_train_states,
        "estimated_train_states": estimated_states,
        "states_per_train_task_estimate": states_per_train_task,
        "feasible": feasible,
        "note": (
            "Train and validation tasks are strictly disjoint. Validation tasks "
            "must never be used during training, threshold selection, or calibration. "
            "Aiming for >=300 train states (states, not rows) with 7 boundaries per state. "
            "Collection should achieve balanced rescue (>=15%), harm (<=10%), and "
            "entropy (>=40%) coverage across the train state pool."
        ),
        "expected_rows": estimated_states * len(BOUNDARIES),
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
        print(f"Design written to {output}")

    return design


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-per-suite", type=int, default=2,
                        help="Tasks reserved for validation per suite")
    parser.add_argument("--validation-offset", type=int, default=0,
                        help="Offset for selecting validation tasks")
    parser.add_argument("--target-train-states", type=int, default=300)
    parser.add_argument("--states-per-train-task", type=int, default=15)
    args = parser.parse_args()

    design = generate_design(
        validation_per_suite=args.validation_per_suite,
        validation_offset=args.validation_offset,
        target_train_states=args.target_train_states,
        states_per_train_task=args.states_per_train_task,
        output=args.output,
    )

    print(json.dumps({
        "train_tasks": design["train_task_count"],
        "validation_tasks": design["validation_task_count"],
        "estimated_train_states": design["estimated_train_states"],
        "feasible": design["feasible"],
    }, indent=2))

    return 0 if design["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify grouped task-level splits: no task leakage between folds.

Checks grouped_task_folds / inner_task_split so that a task in train never
appears in val, and calibration tasks never appear in validation either.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_r4_safe_handback_wm_ridge import (  # noqa: E402
    grouped_task_folds,
    inner_task_split,
)


def synthetic_rows(n_states: int, tasks_per_state: int = 3) -> list[dict]:
    rows = []
    import hashlib

    for s in range(n_states):
        for b in range(tasks_per_state):
            task = f"task_{s // 5}"  # 5 states share a task
            rows.append({
                "state_key": f"sp1_{hashlib.md5(str(s).encode()).hexdigest()[:12]}",
                "task_id": task,
                "suite": f"suite_{task}",
                "elapsed_oft_steps": b,
                "split": "train",
            })
    return rows


def main() -> int:
    rows = synthetic_rows(60)
    folds = grouped_task_folds(rows, 5)
    assert len(folds) == 5, f"expected 5 folds, got {len(folds)}"

    for i, fold in enumerate(folds):
        train_tasks = {str(r["task_id"]) for r in fold["train"]}
        val_tasks = {str(r["task_id"]) for r in fold["val"]}
        overlap = train_tasks & val_tasks
        if overlap:
            print(f"fold {i}: task overlap between train and val: {overlap}")
            return 1

        inner_fit, calib, calib_tasks = inner_task_split(fold["train"], i)
        fit_tasks = {str(r["task_id"]) for r in inner_fit}
        calib_tasks_set = {str(r["task_id"]) for r in calib}
        if fit_tasks & calib_tasks_set:
            print(f"fold {i}: task overlap between inner-fit and calibration")
            return 1
        if calib_tasks_set & val_tasks:
            print(f"fold {i}: calibration tasks appear in validation")
            return 1
        if calib_tasks_set != {str(t) for t in calib_tasks}:
            print(f"fold {i}: calibration task set mismatch")
            return 1

    print("Grouped task splits OK: no train/val, fit/calib, calib/val overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

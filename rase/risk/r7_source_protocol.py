"""Frozen task split and seed protocol for the R7 source-risk probe."""

from __future__ import annotations

import hashlib
import random

import numpy as np


FOLD_SEED = 2026081207
TRAIN_SEEDS = (2026081207, 2026081208, 2026081209, 2026081210, 2026081211)
N_FOLDS = 5


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def task_folds(task_id: np.ndarray, suite: np.ndarray, *, count: int = N_FOLDS,
               seed: int = FOLD_SEED) -> list[set[str]]:
    """Outcome-independent, suite-balanced task folds."""
    result = [set() for _ in range(count)]
    unique_tasks = sorted(set(task_id.tolist()))
    suite_for = {task: str(suite[np.flatnonzero(task_id == task)[0]]) for task in unique_tasks}
    for suite_index, suite_name in enumerate(sorted(set(suite_for.values()))):
        values = [task for task in unique_tasks if suite_for[task] == suite_name]
        random.Random(seed ^ stable_int(suite_name)).shuffle(values)
        for index, task in enumerate(values):
            result[(index + suite_index) % count].add(task)
    if set().union(*result) != set(unique_tasks):
        raise AssertionError("task fold assignment is incomplete")
    return result


def calibration_tasks(train_tasks: set[str], task_id: np.ndarray, suite: np.ndarray,
                      *, fold: int, seed: int = FOLD_SEED) -> set[str]:
    """Choose two outcome-independent calibration tasks per suite."""
    chosen: set[str] = set()
    for suite_name in sorted(set(suite.tolist())):
        values = sorted({str(task_id[i]) for i in range(len(task_id))
                         if str(suite[i]) == suite_name and str(task_id[i]) in train_tasks})
        random.Random(seed ^ stable_int(suite_name) ^ (fold * 0x9E3779B1)).shuffle(values)
        chosen.update(values[: min(2, len(values))])
    return chosen

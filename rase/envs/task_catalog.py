"""Typed access to the LIBERO-Plus task classification catalog."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_CATEGORY_TO_DIMENSION = {
    "Camera Viewpoints": "camera",
    "Robot Initial States": "robot",
}
VALID_DIMENSIONS = frozenset(_CATEGORY_TO_DIMENSION.values())
VALID_LEVELS = frozenset(range(1, 6))


@dataclass(frozen=True, order=True)
class LiberoPlusTask:
    suite: str
    task_id: int
    name: str
    dimension: str
    difficulty: int
    category: str

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.task_id}:{self.dimension}:L{self.difficulty}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskCatalogError(ValueError):
    """The upstream task catalog is malformed or a filter is invalid."""


def _normalise(values: Iterable[str] | None, valid: frozenset[str], label: str) -> frozenset[str]:
    selected = frozenset(value.strip().lower() for value in (values or valid))
    unknown = selected - valid
    if unknown:
        raise TaskCatalogError(f"unknown {label}: {', '.join(sorted(unknown))}")
    if not selected:
        raise TaskCatalogError(f"{label} filter must not be empty")
    return selected


class LiberoPlusTaskCatalog:
    def __init__(self, tasks: Sequence[LiberoPlusTask], source: Path):
        self.tasks = tuple(sorted(tasks))
        self.source = source

    @classmethod
    def load(cls, source: str | Path) -> "LiberoPlusTaskCatalog":
        path = Path(source).expanduser()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskCatalogError(f"cannot read LIBERO-Plus catalog {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise TaskCatalogError("catalog root must map suite names to task lists")

        tasks: list[LiberoPlusTask] = []
        seen: set[tuple[str, int]] = set()
        for suite, records in raw.items():
            if not isinstance(suite, str) or not isinstance(records, list):
                raise TaskCatalogError("each suite must have a string name and task list")
            for record in records:
                if not isinstance(record, Mapping):
                    raise TaskCatalogError(f"{suite} contains a non-object task")
                category = record.get("category")
                dimension = _CATEGORY_TO_DIMENSION.get(category)
                if dimension is None:
                    continue
                try:
                    task_id = int(record["id"])
                    name = str(record["name"])
                    difficulty = int(record["difficulty_level"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise TaskCatalogError(f"invalid {category} record in {suite}: {record}") from exc
                if difficulty not in VALID_LEVELS:
                    raise TaskCatalogError(
                        f"{suite} task {task_id} has invalid difficulty {difficulty}"
                    )
                identity = (suite, task_id)
                if identity in seen:
                    raise TaskCatalogError(f"duplicate task identity {suite}:{task_id}")
                seen.add(identity)
                tasks.append(
                    LiberoPlusTask(
                        suite=suite,
                        task_id=task_id,
                        name=name,
                        dimension=dimension,
                        difficulty=difficulty,
                        category=str(category),
                    )
                )
        if not tasks:
            raise TaskCatalogError("catalog contains no camera or robot tasks")
        return cls(tasks, path)

    def select(
        self,
        *,
        dimensions: Iterable[str] | None = None,
        levels: Iterable[int] | None = None,
        suites: Iterable[str] | None = None,
        profile: str = "full",
        smoke_tasks_per_cell: int = 1,
    ) -> tuple[LiberoPlusTask, ...]:
        dims = _normalise(dimensions, VALID_DIMENSIONS, "dimension")
        chosen_levels = frozenset(int(level) for level in (levels or VALID_LEVELS))
        invalid_levels = chosen_levels - VALID_LEVELS
        if invalid_levels or not chosen_levels:
            raise TaskCatalogError(
                f"difficulty levels must be within L1-L5, got {sorted(chosen_levels)}"
            )
        available_suites = frozenset(task.suite for task in self.tasks)
        chosen_suites = (
            frozenset(suites) if suites is not None else available_suites
        )
        unknown_suites = chosen_suites - available_suites
        if unknown_suites:
            raise TaskCatalogError(f"unknown suites: {', '.join(sorted(unknown_suites))}")

        selected = [
            task
            for task in self.tasks
            if task.dimension in dims
            and task.difficulty in chosen_levels
            and task.suite in chosen_suites
        ]
        if profile == "full":
            return tuple(selected)
        if profile != "smoke":
            raise TaskCatalogError("profile must be 'smoke' or 'full'")
        if smoke_tasks_per_cell < 1:
            raise TaskCatalogError("smoke_tasks_per_cell must be positive")

        counts: dict[tuple[str, str, int], int] = {}
        smoke: list[LiberoPlusTask] = []
        for task in selected:
            cell = (task.suite, task.dimension, task.difficulty)
            if counts.get(cell, 0) < smoke_tasks_per_cell:
                smoke.append(task)
                counts[cell] = counts.get(cell, 0) + 1
        return tuple(smoke)


def parse_levels(spec: str) -> tuple[int, ...]:
    """Parse ``1,3-5`` or ``L1,L3-L5`` into sorted difficulty levels."""
    levels: set[int] = set()
    for token in spec.upper().replace("L", "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            levels.update(range(start, end + 1))
        else:
            levels.add(int(token))
    invalid = levels - VALID_LEVELS
    if not levels or invalid:
        raise TaskCatalogError(f"invalid level specification {spec!r}; expected L1-L5")
    return tuple(sorted(levels))


def _default_catalog() -> Path | None:
    explicit = os.environ.get("LIBERO_PLUS_TASK_CATALOG")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("LIBERO_PLUS_ROOT")
    if root:
        return (
            Path(root).expanduser()
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the upstream catalog without importing LIBERO or rendering."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--check", action="store_true", help="validate and print a summary")
    args = parser.parse_args(argv)
    source = args.catalog or _default_catalog()
    if source is None:
        parser.error(
            "provide --catalog, LIBERO_PLUS_TASK_CATALOG, or LIBERO_PLUS_ROOT"
        )
    catalog = LiberoPlusTaskCatalog.load(source)
    counts: dict[str, int] = {}
    for task in catalog.tasks:
        key = f"{task.suite}/{task.dimension}/L{task.difficulty}"
        counts[key] = counts.get(key, 0) + 1
    print(
        json.dumps(
            {
                "source": str(catalog.source.resolve()),
                "tasks": len(catalog.tasks),
                "cells": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

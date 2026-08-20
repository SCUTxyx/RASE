"""R7 multi-episode, task-clustered source-risk collection schedule."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "rase-r7-multi-episode-source-risk/v1"
DESIGN_VERSION = "rase-r7-multi-episode-design/v1"
SUITES = ("Spatial", "Object", "Goal", "Long")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_design(base: Mapping[str, Any], *, repeats_per_task: int = 4,
                 seed: int = 2_026_081_207) -> dict[str, Any]:
    if repeats_per_task < 3:
        raise ValueError("R7 requires at least three independent episodes per task")
    base_rows = [dict(row) for row in base.get("records") or []]
    if len(base_rows) != 48 or len({str(row["task_id"]) for row in base_rows}) != 48:
        raise ValueError("R7 base design must contain exactly 48 unique tasks")
    records = []
    request_index = 0
    for row in sorted(base_rows, key=lambda value: str(value["task_id"])):
        for repeat in range(repeats_per_task):
            records.append({
                "request_index": request_index,
                "episode_id": f"ep-r7-{seed:010d}-{request_index:06d}",
                "task_id": str(row["task_id"]),
                "suite": str(row["suite"]),
                "dimension": str(row["dimension"]),
                "level": int(row["level"]),
                "init_state_id": repeat,
                "episode_repeat": repeat,
                "role": "natural_development",
            })
            request_index += 1
    payload = {
        "artifact_version": DESIGN_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen",
        "seed": seed,
        "selection_uses_outcomes": False,
        "scientific_scope": (
            "R7 development-only independent reset states; all states/episodes "
            "for a task remain in one outer fold"
        ),
        "source_design_sha256": canonical_sha256(base),
        "repeats_per_task": repeats_per_task,
        "n_episodes": len(records),
        "n_tasks": len({row["task_id"] for row in records}),
        "n_unique_init_state_assignments": len({
            (row["task_id"], row["init_state_id"]) for row in records
        }),
        "suite_counts": dict(Counter(row["suite"] for row in records)),
        "cell_counts": dict(Counter(
            f"{row['suite']}|{row['dimension']}:L{row['level']}" for row in records
        )),
        "records": records,
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def load_design(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_version") != DESIGN_VERSION:
        raise ValueError("unsupported R7 design version")
    declared = str(payload.get("design_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("design_sha256", None)
    actual = canonical_sha256(unsigned)
    if declared != actual:
        raise ValueError(f"R7 design checksum mismatch: {declared} != {actual}")
    if expected_sha256 is not None and declared != expected_sha256:
        raise ValueError("R7 design differs from frozen expected checksum")
    records = list(payload.get("records") or [])
    tasks: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        tasks[str(row["task_id"])].append(row)
    repeats = int(payload["repeats_per_task"])
    if len(tasks) != 48 or any(len(rows) != repeats for rows in tasks.values()):
        raise ValueError("R7 design must have equal independent episodes for 48 tasks")
    if len({str(row["episode_id"]) for row in records}) != len(records):
        raise ValueError("R7 episode IDs must be unique")
    for task, rows in tasks.items():
        init_ids = {int(row["init_state_id"]) for row in rows}
        if len(init_ids) != repeats:
            raise ValueError(f"R7 task {task} repeats an init_state_id")
    return payload


def requests_from_design(design: Mapping[str, Any], *, seed: int) -> list[Any]:
    from .perturb_sampler import PerturbationRequest
    subdimensions = {"clean": "none", "camera": "viewpoint", "robot": "initial_state"}
    result = []
    for row in sorted(design["records"], key=lambda value: int(value["request_index"])):
        task_id = str(row["task_id"])
        try:
            concrete_id = int(task_id.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid R7 task ID: {task_id}") from error
        index = int(row["request_index"])
        dimension = str(row["dimension"])
        result.append(PerturbationRequest(
            index=index,
            suite=str(row["suite"]),
            dimension=dimension,
            subdimension=subdimensions[dimension],
            level=int(row["level"]),
            seed=int(canonical_sha256({
                "protocol": PROTOCOL_VERSION, "seed": seed,
                "request_index": index, "task_id": task_id,
            })[:8], 16),
            global_episode_index=index,
            batch_id=1,
            task_id=concrete_id,
            init_state_id=int(row["init_state_id"]),
            episode_id=str(row["episode_id"]),
        ))
    return result

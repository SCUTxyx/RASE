#!/usr/bin/env python3
"""Freeze outcome-independent, task-disjoint PRE-A3 train partitions for E3-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROLES_PER_SUITE = {"b0_smoke": 1, "b1_collect": 3, "b2_qualification": 2}


def checksum(values: list[str]) -> str:
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def task_rank(design_sha: str, suite: str, task: str) -> str:
    return hashlib.sha256(f"e3b-v1|{design_sha}|{suite}|{task}".encode()).hexdigest()


def partition_records(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    train = [dict(row) for row in payload.get("records") or [] if row.get("split") == "train"]
    if len(train) != 72:
        raise ValueError(f"expected exactly 72 PRE-A3 train roots, got {len(train)}")
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in train:
        by_suite_task[str(row["suite"])][str(row["logical_task_id"])].append(row)
    design_sha = str(payload.get("design_sha256") or "")
    result = {role: [] for role in ROLES_PER_SUITE}
    for suite in sorted(by_suite_task):
        tasks = sorted(by_suite_task[suite], key=lambda task: task_rank(design_sha, suite, task))
        if len(tasks) != sum(ROLES_PER_SUITE.values()):
            raise ValueError(f"{suite}: expected 6 train tasks, got {len(tasks)}")
        cursor = 0
        for role, count in ROLES_PER_SUITE.items():
            selected = tasks[cursor : cursor + count]
            cursor += count
            for task in selected:
                rows = sorted(by_suite_task[suite][task], key=lambda row: int(row["request_index"]))
                cells = {(str(row["dimension"]), int(row["level"])) for row in rows}
                if len(rows) != 3 or cells != {("clean", 0), ("camera", 1), ("robot", 1)}:
                    raise ValueError(f"{task}: expected clean/camera/robot triplet, got {cells}")
                result[role].extend(rows)
    return result


def build_artifacts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    partitions = partition_records(payload)
    role_payloads: dict[str, dict[str, Any]] = {}
    all_roots: set[str] = set()
    all_episodes: set[str] = set()
    all_tasks: set[str] = set()
    checks: dict[str, bool] = {}
    for role, records in partitions.items():
        roots = [str(row["state_key"]) for row in records]
        episodes = {str(row["episode_id"]) for row in records}
        tasks = {str(row["logical_task_id"]) for row in records}
        checks[f"{role}_roots_unique"] = len(roots) == len(set(roots))
        checks[f"{role}_suite_balance"] = Counter(row["suite"] for row in records) == {
            suite: len(records) // 4 for suite in ("Goal", "Long", "Object", "Spatial")
        }
        checks[f"{role}_cell_balance"] = all(
            Counter((row["dimension"], row["level"]) for row in records if row["suite"] == suite)
            == {("clean", 0): len(records) // 12, ("camera", 1): len(records) // 12, ("robot", 1): len(records) // 12}
            for suite in ("Goal", "Long", "Object", "Spatial")
        )
        if all_roots & set(roots) or all_episodes & episodes or all_tasks & tasks:
            raise ValueError(f"partition leakage detected for {role}")
        all_roots.update(roots)
        all_episodes.update(episodes)
        all_tasks.update(tasks)
        role_payloads[role] = {
            "schema_version": "rase-e3b-partition/v1",
            "status": "frozen",
            "role": role,
            "scientific_scope": "PRE-A3 train only; val/test sealed",
            "selection_uses_outcomes": False,
            "source_design_sha256": payload.get("design_sha256"),
            "pool": payload.get("pool"),
            "n_states": len(records),
            "n_tasks": len(tasks),
            "state_keys": roots,
            "state_keys_sha256": checksum(roots),
            "records": records,
        }
    checks.update(
        {
            "all_72_train_roots_partitioned": len(all_roots) == 72,
            "all_24_train_tasks_partitioned": len(all_tasks) == 24,
            "root_disjoint": True,
            "episode_disjoint": True,
            "logical_task_disjoint": True,
            "selection_outcome_independent": True,
        }
    )
    manifest = {
        "schema_version": "rase-e3b-partition-manifest/v1",
        "status": "frozen",
        "source": str(payload.get("artifact_version") or "PRE-A3 keys120"),
        "source_design_sha256": payload.get("design_sha256"),
        "selection_rule": "SHA256-ranked logical tasks within suite; no outcomes",
        "role_counts": {role: len(records) for role, records in partitions.items()},
        "task_counts": {
            role: len({row["logical_task_id"] for row in records})
            for role, records in partitions.items()
        },
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    return manifest, role_payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.resolve().read_text())
    manifest, roles = build_artifacts(payload)
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    for role, value in roles.items():
        atomic_json(output / f"{role}.json", value)
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"decision": manifest["decision"], **manifest["role_counts"]}, sort_keys=True))
    return 0 if manifest["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

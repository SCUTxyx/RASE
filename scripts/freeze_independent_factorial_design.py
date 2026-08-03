#!/usr/bin/env python3
"""Freeze and audit a metadata-only independent factorial collection design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def freeze(
    config: dict[str, Any],
    excluded: list[dict[str, Any]],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    from rase.collect.lerobot_libero_plus_adapter import select_catalog_task
    from rase.collect.perturb_sampler import sample_perturbations

    collection = config["collection"]
    sampling = config["sampling"]
    seed = int(collection["seed"])
    requests = sample_perturbations(
        int(collection["episodes"]),
        seed,
        suite_quotas=sampling.get("suite_quotas"),
        factorial_cells=sampling.get("factorial_cells"),
    )
    excluded_tasks = {
        str(record["task_id"])
        for payload in excluded
        for record in payload.get("records") or []
    }
    excluded_episodes = {
        str(record["episode_id"])
        for payload in excluded
        for record in payload.get("records") or []
    }
    rows = []
    for request in requests:
        task = select_catalog_task(catalog, request)
        rows.append(
            {
                "request_index": request.index,
                "episode_id": f"ep-{seed:08x}-{request.index:08d}",
                "suite": request.suite,
                "dimension": request.dimension,
                "level": request.level,
                "task_id": f"{task.suite}_{task.task_id:06d}",
            }
        )
    tasks = [str(row["task_id"]) for row in rows]
    episodes = [str(row["episode_id"]) for row in rows]
    duplicate_tasks = sorted(task for task, n in Counter(tasks).items() if n > 1)
    overlap_tasks = sorted(set(tasks) & excluded_tasks)
    overlap_episodes = sorted(set(episodes) & excluded_episodes)
    ready = not duplicate_tasks and not overlap_tasks and not overlap_episodes
    return {
        "schema_version": "rase-independent-factorial-design/v1",
        "status": "ready" if ready else "not_ready",
        "n_requests": len(rows),
        "n_unique_tasks": len(set(tasks)),
        "n_unique_episodes": len(set(episodes)),
        "duplicate_task_ids": duplicate_tasks,
        "excluded_task_overlap": overlap_tasks,
        "excluded_episode_overlap": overlap_episodes,
        "cell_counts": dict(
            sorted(
                Counter(
                    f"{row['suite']}|{row['dimension']}:L{row['level']}"
                    for row in rows
                ).items()
            )
        ),
        "records_sha256": _sha(rows),
        "records": rows,
        "selection_uses_intervention_outcomes": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exclude-keys", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config.resolve())
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.lerobot_libero_plus_adapter import _load_catalog

    adapter = config.get("adapter_config") or {}
    paths = ensure_libero_plus_paths(adapter.get("libero_plus_root"))
    catalog_path = Path(paths["benchmark_root"]) / "benchmark" / "task_classification.json"
    result = freeze(
        config,
        [_read(path.resolve()) for path in args.exclude_keys],
        _load_catalog(catalog_path),
    )
    _write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "n_requests": result["n_requests"],
                "n_unique_tasks": result["n_unique_tasks"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

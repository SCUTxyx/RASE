#!/usr/bin/env python3
"""Freeze an outcome-independent PRE-A3 120-state task/condition design."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from rase.collect.pre_a3 import CELLS, SUITES, build_design  # noqa: E402

SUITE_KEYS = {
    "Spatial": "libero_spatial",
    "Object": "libero_object",
    "Goal": "libero_goal",
    "Long": "libero_10",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _excluded_concrete_task_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        for row in payload.get("records") or payload.get("per_task") or []:
            if isinstance(row, dict) and row.get("task_id"):
                excluded.add(str(row["task_id"]))
            if isinstance(row, dict) and row.get("concrete_task_id"):
                excluded.add(str(row["concrete_task_id"]))
    return excluded


def _clean_task_ids() -> dict[str, list[str]]:
    return {
        suite: [f"{SUITE_KEYS[suite]}_{index:06d}" for index in range(10)]
        for suite in SUITES
    }


def _plus_candidates(catalog_path: Path) -> dict[str, dict[str, list[str]]]:
    raw = _load_json(catalog_path)
    out: dict[str, dict[str, list[str]]] = {
        suite: {"camera": [], "robot": []} for suite in SUITES
    }
    suite_map = {value: key for key, value in SUITE_KEYS.items()}
    category_dim = {
        "Camera Viewpoints": "camera",
        "Robot Initial States": "robot",
    }
    for suite_key, records in raw.items():
        suite = suite_map.get(suite_key)
        if suite is None:
            continue
        for record in records:
            dim = category_dim.get(str(record.get("category")))
            level = record.get("difficulty_level", record.get("difficulty"))
            if dim is None or int(level or 0) != 1:
                continue
            out[suite][dim].append(f"{suite_key}_{int(record['id']):06d}")
        for dim in ("camera", "robot"):
            out[suite][dim] = sorted(set(out[suite][dim]))
    return out


def build_records(
    *,
    seed: int,
    excluded_plus: set[str],
    clean_pool: dict[str, list[str]],
    plus_pool: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Bind 10 logical tasks/suite to clean-10 + unused Plus camera/robot L1 tasks."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    request_index = 0
    for suite in SUITES:
        clean = list(clean_pool[suite])
        if len(clean) != 10:
            raise ValueError(f"{suite}: clean-10 incomplete")
        camera = [task for task in plus_pool[suite]["camera"] if task not in excluded_plus]
        robot = [task for task in plus_pool[suite]["robot"] if task not in excluded_plus]
        if len(camera) < 10 or len(robot) < 10:
            raise ValueError(
                f"{suite}: need 10 unused Plus L1 tasks each; "
                f"camera={len(camera)} robot={len(robot)}"
            )
        rng.shuffle(camera)
        rng.shuffle(robot)
        # Keep clean in official order for reproducibility; shuffle only Plus arms.
        for task_slot, clean_task in enumerate(clean):
            logical_task = f"pre_a3_{SUITE_KEYS[suite]}_task{task_slot:02d}"
            concrete = {
                "clean": clean_task,
                "camera": camera[task_slot],
                "robot": robot[task_slot],
            }
            for dim, level in CELLS:
                records.append(
                    {
                        "request_index": request_index,
                        "suite": suite,
                        "task_id": logical_task,
                        "concrete_task_id": concrete[dim],
                        "dimension": dim,
                        "level": level,
                        "episode_id": f"ep-pre-a3-{seed}-{request_index:06d}",
                    }
                )
                request_index += 1
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plus-catalog",
        type=Path,
        default=Path(
            "/root/autodl-tmp/src/LIBERO-plus/libero/libero/benchmark/"
            "task_classification.json"
        ),
    )
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=2_026_080_401)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/rase_pre_a3_design120_v1.json",
    )
    args = parser.parse_args()

    default_excludes = [
        ROOT / "runs/rase_ui_phase0g_independent48_design.json",
        ROOT / "runs/rase_pre_a0_strict_resample12_keys_v1.json",
        ROOT / "runs/rase_ui_phase1a_replacement48_analysis_v2.json",
        ROOT / "runs/ngc_w10_object_spatial_state_keys.json",
    ]
    exclude_paths = args.exclude or default_excludes
    excluded_plus = _excluded_concrete_task_ids(exclude_paths)
    # Clean-10 must be reused with new episode seeds; strip clean IDs from Plus exclusion.
    clean_ids = {task for tasks in _clean_task_ids().values() for task in tasks}
    excluded_plus -= clean_ids

    if not args.plus_catalog.exists():
        raise SystemExit(f"missing Plus catalog: {args.plus_catalog}")

    records = build_records(
        seed=args.seed,
        excluded_plus=excluded_plus,
        clean_pool=_clean_task_ids(),
        plus_pool=_plus_candidates(args.plus_catalog),
    )
    design = build_design(
        records,
        seed=args.seed,
        excluded_task_ids=[],  # logical IDs are new by construction
    )
    concrete = {int(row["request_index"]): row["concrete_task_id"] for row in records}
    for row in design["records"]:
        row["concrete_task_id"] = concrete[int(row["request_index"])]
    design["clean10_reuse"] = (
        "Official clean-10 task names are reused with new episode seeds; "
        "camera/robot L1 concrete tasks exclude prior development cohorts."
    )
    design["excluded_concrete_plus_task_ids_n"] = len(excluded_plus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen design: {args.output}")
    args.output.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_requests": design["n_requests"],
                "n_unique_tasks": design["n_unique_tasks"],
                "split_counts": design["split_counts"],
                "design_sha256": design["design_sha256"],
                "excluded_plus_n": len(excluded_plus),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze an outcome-blind, task-balanced root plan for V6 Stage 0.

The plan deliberately contains no rollout outcomes.  It assigns each selected
LIBERO-PRO pool state to one requested cursor stratum and records all random
seeds needed by the C / R-same / R-new protocol.  The collector is the only
component allowed to attach outcomes later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "rase-v6-stage0-root-plan/v1"


def stable_seed(*parts: object) -> int:
    """Deterministic seed in the range accepted by NumPy/PyTorch."""
    token = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_suite(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "long": "libero_10",
        "libero10": "libero_10",
        "libero_10": "libero_10",
        "goal": "libero_goal",
        "spatial": "libero_spatial",
        "object": "libero_object",
    }
    return aliases.get(text, text)


def metadata_for_entry(pool: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    relative = entry.get("path")
    if not isinstance(relative, str):
        return None
    path = pool / relative / "meta.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def perturb_level(metadata: dict[str, Any]) -> float | None:
    for key in ("perturb_level", "level", "perturbation_level", "magnitude"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def perturb_dimension(metadata: dict[str, Any]) -> str:
    for key in ("perturb_dim", "dimension", "perturbation_dim"):
        value = metadata.get(key)
        if value is not None:
            return str(value).strip().lower()
    return ""


def perturb_subdimension(metadata: dict[str, Any]) -> str:
    for key in ("perturb_sub", "subdimension", "perturbation_subdimension"):
        value = metadata.get(key)
        if value is not None:
            return str(value).strip().lower()
    return ""


def logical_task_id(metadata: dict[str, Any], entry: dict[str, Any]) -> str:
    for key in ("logical_task_id", "task_id", "task", "benchmark_task"):
        value = metadata.get(key) or entry.get(key)
        if value:
            return str(value)
    raise ValueError("state has no logical task identifier")


def state_suite(metadata: dict[str, Any], entry: dict[str, Any]) -> str:
    explicit = metadata.get("suite") or entry.get("suite")
    if explicit:
        return normalize_suite(explicit)
    return normalize_suite(logical_task_id(metadata, entry).split("/")[0])


def parse_cursors(value: str, *, native_horizon: int) -> list[int]:
    cursors = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(cursors) < 2 or len(set(cursors)) != len(cursors):
        raise ValueError("--cursors must contain at least two distinct integers")
    if any(cursor <= 0 or cursor >= native_horizon for cursor in cursors):
        raise ValueError("every cursor must satisfy 0 < cursor < native_chunk_horizon")
    return cursors


def _round_robin_tasks(
    candidates: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    rng: random.Random,
    used_keys: set[str],
) -> list[dict[str, Any]]:
    """Take at most one new state per task per pass, then repeat if needed."""
    for values in candidates.values():
        rng.shuffle(values)
    task_order = list(candidates)
    rng.shuffle(task_order)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for task in task_order:
            available = candidates[task]
            while available and available[-1]["state_key"] in used_keys:
                available.pop()
            if not available:
                continue
            selected.append(available.pop())
            used_keys.add(selected[-1]["state_key"])
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def select_records(
    pool: Path,
    *,
    suite: str,
    perturb_dim: str,
    level: float,
    n_roots: int,
    cursors: Iterable[int],
    seed: int,
) -> list[dict[str, Any]]:
    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states = manifest.get("states")
    if not isinstance(states, dict):
        raise ValueError(f"invalid state-pool manifest: {manifest_path}")
    wanted_suite = normalize_suite(suite)
    wanted_dim = perturb_dim.strip().lower()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state_key, entry in states.items():
        if not isinstance(entry, dict):
            continue
        metadata = metadata_for_entry(pool, entry)
        if metadata is None:
            continue
        if state_suite(metadata, entry) != wanted_suite:
            continue
        raw_dimension = perturb_dimension(metadata)
        raw_subdimension = perturb_subdimension(metadata)
        # Existing LIBERO-PRO pools have used both encodings for a position
        # perturbation: ``perturb_dim=position`` and
        # ``perturb_dim=robot, perturb_sub=position``.  Match either but save
        # the original pair so analysis never loses the physical provenance.
        if wanted_dim not in {raw_dimension, raw_subdimension}:
            continue
        found_level = perturb_level(metadata)
        if found_level is None or abs(found_level - float(level)) > 1e-9:
            continue
        task_id = logical_task_id(metadata, entry)
        by_task[task_id].append(
            {
                "state_key": str(state_key),
                "task_id": task_id,
                "episode_id": str(metadata.get("episode_id") or entry.get("episode_id") or ""),
                "episode_seed": metadata.get("seed"),
                "perturbation_seed": metadata.get("perturb_seed") or metadata.get("perturbation_seed"),
                "init_state_id": metadata.get("init_state_id") or entry.get("init_state_id"),
                "pool_perturb_dim": raw_dimension,
                "pool_perturb_sub": raw_subdimension or None,
                "metadata_path": str(entry.get("path")),
            }
        )
    if not by_task:
        raise ValueError(
            f"no {wanted_suite}/{wanted_dim}/level={level} states in {pool}"
        )

    cursor_list = list(cursors)
    allocation = [n_roots // len(cursor_list)] * len(cursor_list)
    for index in range(n_roots % len(cursor_list)):
        allocation[index] += 1
    rng = random.Random(int(seed))
    used_keys: set[str] = set()
    selected: list[dict[str, Any]] = []
    for cursor, count in zip(cursor_list, allocation, strict=True):
        # Copy mutable lists because round-robin pops from them.
        task_candidates = {task: list(values) for task, values in by_task.items()}
        bucket = _round_robin_tasks(
            task_candidates, count=count, rng=rng, used_keys=used_keys,
        )
        if len(bucket) != count:
            raise ValueError(
                f"only found {len(bucket)} independent pool states for cursor {cursor}; "
                f"need {count}. Reduce --n-roots or collect a larger pool."
            )
        for record in bucket:
            record["cursor"] = int(cursor)
            selected.append(record)
    rng.shuffle(selected)
    return selected


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    pool = args.pool.resolve()
    native_horizon = int(args.native_chunk_horizon)
    cursors = parse_cursors(args.cursors, native_horizon=native_horizon)
    selected = select_records(
        pool,
        suite=args.suite,
        perturb_dim=args.perturb_dim,
        level=float(args.perturb_level),
        n_roots=int(args.n_roots),
        cursors=cursors,
        seed=int(args.seed),
    )
    roots = []
    for index, record in enumerate(selected):
        root_id = f"v6s0_{index:03d}_{record['state_key'][:12]}_k{record['cursor']}"
        roots.append(
            {
                **record,
                "root_id": root_id,
                "suite": normalize_suite(args.suite),
                "perturb_dim": args.perturb_dim,
                "perturb_level": float(args.perturb_level),
                "native_chunk_horizon": native_horizon,
                "requested_fraction": float(record["cursor"]) / native_horizon,
                "source_generation_seed": stable_seed("v6-stage0-source", args.seed, root_id),
                "downstream_seed": stable_seed("v6-stage0-mu", args.seed, root_id),
                "r_new_generation_seeds": [
                    stable_seed("v6-stage0-r-new", args.seed, root_id, sample)
                    for sample in range(int(args.r_new_k))
                ],
            }
        )
    return {
        "schema_version": SCHEMA,
        "created_for": "V6 Stage 0 opportunity gate",
        "selection_outcomes_used": False,
        "pool": str(pool),
        "selection": {
            "suite": normalize_suite(args.suite),
            "perturb_dim": args.perturb_dim,
            "perturb_level": float(args.perturb_level),
            "n_roots": int(args.n_roots),
            "native_chunk_horizon": native_horizon,
            "cursors": cursors,
            "r_new_k": int(args.r_new_k),
            "seed": int(args.seed),
        },
        "roots": roots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--perturb-dim", default="position")
    parser.add_argument("--perturb-level", type=float, default=0.2)
    parser.add_argument("--n-roots", type=int, default=30)
    parser.add_argument("--native-chunk-horizon", type=int, default=10)
    parser.add_argument("--cursors", default="3,5,8")
    parser.add_argument("--r-new-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.n_roots < 6:
        parser.error("--n-roots must be at least 6 for a two-sided pilot")
    if args.r_new_k < 2:
        parser.error("--r-new-k must be at least 2; pilot preregisters K=4")
    plan = build_plan(args)
    atomic_json(args.output.resolve(), plan)
    by_cursor: dict[int, int] = defaultdict(int)
    by_task: set[str] = set()
    for root in plan["roots"]:
        by_cursor[int(root["cursor"])] += 1
        by_task.add(str(root["task_id"]))
    print(
        f"wrote {args.output} roots={len(plan['roots'])} tasks={len(by_task)} "
        f"cursor_counts={dict(sorted(by_cursor.items()))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

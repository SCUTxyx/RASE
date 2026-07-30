"""Frozen deterministic W9B clean-control episode schedule."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .perturb_sampler import PerturbationRequest
from .schema import canonical_json

SCHEDULE_SCHEMA = "rase-w9b-clean-control-schedule/v1"
PROTOCOL_VERSION = "W9B-clean-control/v1"
BATCH_SIZES = (60, 40, 40)
MAX_EPISODES = sum(BATCH_SIZES)
N_CLEAN_TASKS = 10
DEFAULT_N_INIT_STATES = 50
SUITES = ("Goal", "Long", "Object", "Spatial")


def _salted_int(seed: int, salt: str, *parts: object, bits: int = 64) -> int:
    token = ":".join((str(seed), salt, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[: bits // 8], "big")


def _balanced_suites(seed: int, batch_id: int, size: int) -> list[str]:
    if size % len(SUITES):
        raise ValueError("W9B batch sizes must divide evenly across suites")
    values = [suite for suite in SUITES for _ in range(size // len(SUITES))]
    return [
        value
        for _, value in sorted(
            (
                _salted_int(seed, "suite-order/v1", batch_id, position, suite),
                suite,
            )
            for position, suite in enumerate(values)
        )
    ]


def _init_permutation(seed: int, suite: str, task_id: int, cycle: int) -> list[int]:
    return sorted(
        range(DEFAULT_N_INIT_STATES),
        key=lambda init_state_id: _salted_int(
            seed,
            "init-order/v1",
            suite,
            task_id,
            cycle,
            init_state_id,
        ),
    )


@dataclass(frozen=True)
class W9BScheduleRow:
    global_episode_index: int
    batch_id: int
    request_index: int
    suite: str
    task_id: int
    init_state_id: int
    policy_seed: int
    perturb_dim: str
    perturb_sub: str
    level: int
    episode_id: str

    def to_request(self) -> PerturbationRequest:
        return PerturbationRequest(
            index=self.request_index,
            suite=self.suite,
            dimension=self.perturb_dim,
            subdimension=self.perturb_sub,
            level=self.level,
            seed=self.policy_seed,
            global_episode_index=self.global_episode_index,
            batch_id=self.batch_id,
            task_id=self.task_id,
            init_state_id=self.init_state_id,
            episode_id=self.episode_id,
        )


def generate_w9b_schedule(seed: int) -> dict[str, Any]:
    """Generate all 140 frozen rows with independent salted mappings."""
    rows: list[W9BScheduleRow] = []
    task_init_counts: dict[tuple[str, int], int] = defaultdict(int)
    global_index = 0
    for batch_id, batch_size in enumerate(BATCH_SIZES, 1):
        suites = _balanced_suites(seed, batch_id, batch_size)
        for request_index, suite in enumerate(suites):
            task_id = (
                _salted_int(seed, "task-id/v1", global_index, suite)
                % N_CLEAN_TASKS
                + 1
            )
            task_key = (suite, task_id)
            ordinal = task_init_counts[task_key]
            cycle, within_cycle = divmod(ordinal, DEFAULT_N_INIT_STATES)
            init_state_id = _init_permutation(seed, suite, task_id, cycle)[within_cycle]
            task_init_counts[task_key] += 1
            policy_seed = _salted_int(
                seed, "policy-seed/v1", global_index, suite, task_id, bits=32
            )
            rows.append(
                W9BScheduleRow(
                    global_episode_index=global_index,
                    batch_id=batch_id,
                    request_index=request_index,
                    suite=suite,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    policy_seed=policy_seed,
                    perturb_dim="clean",
                    perturb_sub="none",
                    level=0,
                    episode_id=f"ep-w9b-{global_index:08d}",
                )
            )
            global_index += 1
    payload = {
        "schema_version": SCHEDULE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "schedule_seed": int(seed),
        "batch_sizes": list(BATCH_SIZES),
        "max_episodes": MAX_EPISODES,
        "n_clean_tasks_per_suite": N_CLEAN_TASKS,
        "n_init_states_per_task": DEFAULT_N_INIT_STATES,
        "rows": [asdict(row) for row in rows],
    }
    validate_w9b_schedule(payload)
    return payload


def schedule_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json(payload) + b"\n"


def schedule_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(schedule_bytes(payload)).hexdigest()


def write_w9b_schedule(path: Path, payload: Mapping[str, Any]) -> str:
    validate_w9b_schedule(payload)
    rendered = schedule_bytes(payload)
    digest = hashlib.sha256(rendered).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def _row_from_mapping(value: Mapping[str, Any]) -> W9BScheduleRow:
    try:
        return W9BScheduleRow(
            global_episode_index=int(value["global_episode_index"]),
            batch_id=int(value["batch_id"]),
            request_index=int(value["request_index"]),
            suite=str(value["suite"]),
            task_id=int(value["task_id"]),
            init_state_id=int(value["init_state_id"]),
            policy_seed=int(value["policy_seed"]),
            perturb_dim=str(value["perturb_dim"]),
            perturb_sub=str(value["perturb_sub"]),
            level=int(value["level"]),
            episode_id=str(value["episode_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid W9B schedule row: {value}") from exc


def validate_w9b_schedule(payload: Mapping[str, Any]) -> list[W9BScheduleRow]:
    if payload.get("schema_version") != SCHEDULE_SCHEMA:
        raise ValueError("unsupported W9B schedule schema")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("W9B protocol version mismatch")
    if tuple(payload.get("batch_sizes") or ()) != BATCH_SIZES:
        raise ValueError("W9B batch sizes must be exactly 60/40/40")
    if int(payload.get("max_episodes", -1)) != MAX_EPISODES:
        raise ValueError("W9B maximum must be exactly 140 episodes")
    if int(payload.get("n_init_states_per_task", -1)) != DEFAULT_N_INIT_STATES:
        raise ValueError("W9B schedule must target exactly 50 init states per task")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("W9B schedule rows must be an array")
    rows = [_row_from_mapping(row) for row in raw_rows]
    if len(rows) != MAX_EPISODES:
        raise ValueError(f"W9B schedule requires {MAX_EPISODES} rows")
    if [row.global_episode_index for row in rows] != list(range(MAX_EPISODES)):
        raise ValueError("W9B schedule has missing or reordered global rows")
    episode_ids = [row.episode_id for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("W9B schedule has duplicate episode_id values")

    by_batch: dict[int, list[W9BScheduleRow]] = defaultdict(list)
    seen_inits: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        if row.suite not in SUITES:
            raise ValueError(f"invalid W9B suite {row.suite!r}")
        if row.task_id not in range(1, N_CLEAN_TASKS + 1):
            raise ValueError(f"W9B task_id out of range: {row.task_id}")
        if row.init_state_id not in range(DEFAULT_N_INIT_STATES):
            raise ValueError(f"W9B init_state_id out of range: {row.init_state_id}")
        if (
            row.perturb_dim != "clean"
            or row.perturb_sub != "none"
            or row.level != 0
        ):
            raise ValueError("W9B rows must use clean/none/L0 semantics")
        by_batch[row.batch_id].append(row)
        seen_inits[(row.suite, row.task_id)].append(row.init_state_id)

    for batch_id, expected_size in enumerate(BATCH_SIZES, 1):
        batch = by_batch.get(batch_id, [])
        if len(batch) != expected_size:
            raise ValueError(f"W9B batch {batch_id} requires {expected_size} rows")
        if [row.request_index for row in batch] != list(range(expected_size)):
            raise ValueError(f"W9B batch {batch_id} request indices are not contiguous")
        suite_counts = {suite: sum(row.suite == suite for row in batch) for suite in SUITES}
        if len(set(suite_counts.values())) != 1:
            raise ValueError(f"W9B batch {batch_id} is not suite-balanced")

    for task_key, init_ids in seen_inits.items():
        for start in range(0, len(init_ids), DEFAULT_N_INIT_STATES):
            cycle = init_ids[start : start + DEFAULT_N_INIT_STATES]
            if len(cycle) != len(set(cycle)):
                raise ValueError(
                    f"W9B init_state_id repeats before exhaustion for {task_key}"
                )
    return rows


def load_w9b_schedule(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"W9B schedule SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid W9B schedule JSON: {exc}") from exc
    validate_w9b_schedule(payload)
    if raw != schedule_bytes(payload):
        raise ValueError("W9B schedule is not canonical byte-stable JSON")
    return payload


def requests_for_batch(
    payload: Mapping[str, Any], batch_id: int
) -> list[PerturbationRequest]:
    rows = validate_w9b_schedule(payload)
    if batch_id not in range(1, len(BATCH_SIZES) + 1):
        raise ValueError("W9B batch_id must be 1, 2, or 3")
    return [row.to_request() for row in rows if row.batch_id == batch_id]

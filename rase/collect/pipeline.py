"""NGC Step 1 collection loop (importable; CLI lives under scripts/)."""

from __future__ import annotations

import base64
import importlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .perturb_sampler import PerturbationRequest, sample_perturbations, summarize
from .schema import StateMetadata
from .state_pool import StatePool, retain_snapshot, snapshot_steps
from .w9b_schedule import (
    BATCH_SIZES,
    MAX_EPISODES,
    load_w9b_schedule,
)
from .w9b_schedule import PROTOCOL_VERSION as W9B_PROTOCOL_VERSION
from .w9b_schedule import requests_for_batch as w9b_requests_for_batch
from .w9c_schedule import PROTOCOL_VERSION as W9C_PROTOCOL_VERSION
from .w9c_schedule import (
    load_w9c_schedule,
)
from .w9c_schedule import requests_for_batch as w9c_requests_for_batch
from .pre_a3_schedule import PROTOCOL_VERSION as PRE_A3_PROTOCOL_VERSION
from .pre_a3_schedule import load_pre_a3_design, requests_from_design
from .pre_c0 import PROTOCOL_VERSION as PRE_C0_PROTOCOL_VERSION
from .pre_c0 import load_pre_c0_design
from .pre_c0 import requests_from_design as pre_c0_requests_from_design
from .r7_schedule import PROTOCOL_VERSION as R7_PROTOCOL_VERSION
from .r7_schedule import load_design as load_r7_design
from .r7_schedule import requests_from_design as r7_requests_from_design

CLEAN_CONTROL_PROTOCOLS = frozenset({W9B_PROTOCOL_VERSION, W9C_PROTOCOL_VERSION})
SCHEDULED_PROTOCOLS = frozenset(
    {
        W9B_PROTOCOL_VERSION,
        W9C_PROTOCOL_VERSION,
        PRE_A3_PROTOCOL_VERSION,
        PRE_C0_PROTOCOL_VERSION,
        R7_PROTOCOL_VERSION,
    }
)
# Backward-compatible alias used by older imports/tests.
PROTOCOL_VERSION = W9B_PROTOCOL_VERSION

# Valid 1x1 RGB PNG. Dry-run writes real, independently decodable image files.
_DRY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class EpisodeSnapshot:
    step: int
    sim_state: np.ndarray
    controller_state: Any
    rng_state: Any
    observations: Mapping[str, bytes]
    proprio: np.ndarray


@dataclass(frozen=True)
class EpisodeResult:
    outcome: str
    task_id: str
    instruction: str
    snapshots: tuple[EpisodeSnapshot, ...]
    suite: str | None = None
    init_state_id: int | None = None
    clean_task_name: str | None = None
    bddl_stem: str | None = None
    libero_flavor: str | None = None
    metrics: Mapping[str, Any] | None = None


class DryRunAdapter:
    """Deterministic CPU-only adapter used by smoke tests and storage checks."""

    def __init__(self, config: Mapping[str, Any]):
        self.action_chunks = int(config["collection"]["action_chunks_per_episode"])

    def run_episode(
        self, request: PerturbationRequest, episode_id: str, cadence: int
    ) -> EpisodeResult:
        outcome = "success" if request.index % 5 == 0 else "failure"
        steps = ([0] if self.action_chunks == 0 else
                 list(snapshot_steps(self.action_chunks, cadence)))
        snapshots = tuple(
            EpisodeSnapshot(
                step=step,
                sim_state=np.asarray([request.seed, step], dtype=np.int64),
                controller_state={"mode": "dry-run", "step": step},
                rng_state={"seed": request.seed},
                observations={
                    "agentview": _DRY_PNG,
                    "wrist": _DRY_PNG,
                    "side": _DRY_PNG,
                },
                proprio=np.asarray([step, request.level], dtype=np.float32),
            )
            for step in steps
        )
        return EpisodeResult(
            outcome=outcome,
            task_id=(
                _scheduled_task_id(request)
                if request.task_id is not None
                else f"{request.suite.lower()}_{request.index:06d}"
            ),
            instruction=f"dry-run task {request.index}",
            snapshots=snapshots,
            suite=request.suite,
            init_state_id=request.init_state_id,
        )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    collection = config["collection"]
    protocol = dict(config.get("protocol") or {})
    version = protocol.get("version")
    cadence = int(collection["snapshot_cadence_action_chunks"])
    # PRE-A3 recovery120 needs reset/step-0 snapshots only (cadence=1, max_chunks=1).
    if version == PRE_A3_PROTOCOL_VERSION:
        if cadence != 1:
            raise ValueError(
                "PRE-A3 recovery120 snapshot cadence must be 1 action chunk"
            )
        maximum = int(
            collection.get(
                "max_action_chunks", collection.get("action_chunks_per_episode", 0)
            )
        )
        if maximum != 1:
            raise ValueError("PRE-A3 recovery120 max_action_chunks must be 1")
    elif version == PRE_C0_PROTOCOL_VERSION:
        if cadence != 1:
            raise ValueError("PRE-C0 trajectory snapshot cadence must be 1 action chunk")
        if int(collection.get("max_action_chunks", 0)) < 2:
            raise ValueError("PRE-C0 requires full trajectories, not reset-only snapshots")
        if float(collection["successful_snapshot_retention"]) != 1.0:
            raise ValueError("PRE-C0 must retain all success controls")
    elif version == R7_PROTOCOL_VERSION:
        if cadence != 1:
            raise ValueError("R7 reset pool snapshot cadence must be one chunk")
        if not bool(collection.get("reset_only", False)):
            raise ValueError("R7 reset pool requires reset_only=true")
        if int(collection.get("max_action_chunks", -1)) != 0:
            raise ValueError("R7 reset pool must execute zero source action chunks")
        if int(collection.get("action_chunks_per_episode", -1)) != 0:
            raise ValueError("R7 reset pool action_chunks_per_episode must be zero")
        if float(collection["successful_snapshot_retention"]) != 1.0:
            raise ValueError("R7 must retain every scheduled reset state")
    elif cadence != 2:
        raise ValueError("NGC Step 1 snapshot cadence must be 2 action chunks")
    retention = float(collection["successful_snapshot_retention"])
    if not 0.0 <= retention <= 1.0:
        raise ValueError("successful snapshot retention must be within [0, 1]")
    if version == W9B_PROTOCOL_VERSION:
        _validate_w9b_config(config)
    elif version == W9C_PROTOCOL_VERSION:
        _validate_w9c_config(config)
    return config


def _scheduled_task_id(request: PerturbationRequest) -> str:
    suite_names = {
        "Spatial": "libero_spatial",
        "Object": "libero_object",
        "Goal": "libero_goal",
        "Long": "libero_10",
    }
    if request.task_id is None:
        raise ValueError("scheduled request is missing task_id")
    return f"{suite_names[request.suite]}_{request.task_id:06d}"


def _validate_w9b_config(config: Mapping[str, Any]) -> None:
    collection = dict(config["collection"])
    protocol = dict(config.get("protocol") or {})
    output = Path(str(collection["output_dir"]))
    output_text = output.as_posix()
    smoke_mode = bool(collection.get("smoke_mode", False))
    smoke_output = (
        smoke_mode
        and "/runs/ngc_w9b_" in f"/{output_text}"
        and "smoke" in output.name
    )
    if output_text != "pool/ngc_w9b_clean_controls" and not smoke_output:
        raise ValueError(
            "W9B output_dir must be the formal W9B pool or an explicit "
            "runs/ngc_w9b_*_smoke path"
        )
    if output_text == "pool/ngc_w9_clean_controls":
        raise ValueError("W9B must never write into the legacy W9 pool")
    if float(collection["successful_snapshot_retention"]) != 1.0:
        raise ValueError("W9B successful_snapshot_retention must be exactly 1.0")
    if int(protocol.get("maximum_episodes", -1)) != MAX_EPISODES:
        raise ValueError("W9B maximum_episodes must be exactly 140")
    if not protocol.get("schedule_path") or not protocol.get("schedule_sha256"):
        raise ValueError("W9B requires schedule_path and schedule_sha256")


def _validate_w9c_config(config: Mapping[str, Any]) -> None:
    collection = dict(config["collection"])
    protocol = dict(config.get("protocol") or {})
    output = Path(str(collection["output_dir"]))
    output_text = output.as_posix()
    smoke_mode = bool(collection.get("smoke_mode", False))
    smoke_output = (
        smoke_mode
        and "/runs/ngc_w9c_" in f"/{output_text}"
        and ("smoke" in output.name or "probe" in output.name)
    )
    if output_text != "pool/ngc_w9c_clean_controls" and not smoke_output:
        raise ValueError(
            "W9C output_dir must be the formal W9C pool or an explicit "
            "runs/ngc_w9c_*_{smoke,probe} path"
        )
    for banned in (
        "pool/ngc_w9_clean_controls",
        "pool/ngc_w9b_clean_controls",
        "pool/ngc_w9a_clean_controls",
    ):
        if output_text == banned:
            raise ValueError(f"W9C must never write into {banned}")
    if float(collection["successful_snapshot_retention"]) != 1.0:
        raise ValueError("W9C successful_snapshot_retention must be exactly 1.0")
    if int(protocol.get("maximum_episodes", -1)) != MAX_EPISODES:
        raise ValueError("W9C maximum_episodes must be exactly 140")
    if not protocol.get("schedule_path") or not protocol.get("schedule_sha256"):
        raise ValueError("W9C requires schedule_path and schedule_sha256")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _scheduled_requests(
    config: Mapping[str, Any],
) -> tuple[list[PerturbationRequest], dict[str, Any]]:
    protocol = dict(config.get("protocol") or {})
    collection = dict(config["collection"])
    version = protocol.get("version")
    if version == W9B_PROTOCOL_VERSION:
        payload = load_w9b_schedule(
            Path(str(protocol["schedule_path"])),
            expected_sha256=str(protocol["schedule_sha256"]),
        )
        batch_id = int(collection.get("schedule_batch_id", 1))
        requests = w9b_requests_for_batch(payload, batch_id)
        label = "W9B"
        if len(requests) != BATCH_SIZES[batch_id - 1]:
            raise ValueError(f"{label} schedule batch has the wrong number of rows")
    elif version == W9C_PROTOCOL_VERSION:
        payload = load_w9c_schedule(
            Path(str(protocol["schedule_path"])),
            expected_sha256=str(protocol["schedule_sha256"]),
        )
        batch_id = int(collection.get("schedule_batch_id", 1))
        requests = w9c_requests_for_batch(payload, batch_id)
        label = "W9C"
        if len(requests) != BATCH_SIZES[batch_id - 1]:
            raise ValueError(f"{label} schedule batch has the wrong number of rows")
    elif version == PRE_A3_PROTOCOL_VERSION:
        design_path = Path(
            str(protocol.get("design") or (config.get("sampling") or {}).get("design"))
        )
        payload = load_pre_a3_design(
            design_path,
            expected_sha256=(
                str(protocol["design_sha256"])
                if protocol.get("design_sha256")
                else None
            ),
        )
        requests = requests_from_design(payload, seed=int(collection["seed"]))
        label = "PRE-A3"
        expected = int(protocol.get("expected_episodes") or collection["episodes"])
        if len(requests) != expected:
            raise ValueError(
                f"{label} schedule has {len(requests)} rows, expected {expected}"
            )
    elif version == PRE_C0_PROTOCOL_VERSION:
        design_path = Path(
            str(protocol.get("design") or (config.get("sampling") or {}).get("design"))
        )
        payload = load_pre_c0_design(
            design_path,
            expected_sha256=(
                str(protocol["design_sha256"])
                if protocol.get("design_sha256")
                else None
            ),
        )
        requests = pre_c0_requests_from_design(
            payload, seed=int(collection["seed"])
        )
        label = "PRE-C0"
        expected = int(protocol.get("expected_episodes") or collection["episodes"])
        if len(requests) != expected:
            raise ValueError(
                f"{label} schedule has {len(requests)} rows, expected {expected}"
            )
    elif version == R7_PROTOCOL_VERSION:
        design_path = Path(str(protocol.get("design") or (config.get("sampling") or {}).get("design")))
        payload = load_r7_design(
            design_path,
            expected_sha256=(str(protocol["design_sha256"])
                             if protocol.get("design_sha256") else None),
        )
        requests = r7_requests_from_design(payload, seed=int(collection["seed"]))
        label = "R7"
        expected = int(protocol.get("expected_episodes") or collection["episodes"])
        if len(requests) != expected:
            raise ValueError(f"R7 schedule has {len(requests)} rows, expected {expected}")
    else:
        raise ValueError(f"unsupported scheduled protocol version {version!r}")
    if int(collection["episodes"]) != len(requests):
        raise ValueError(f"{label} collection.episodes does not match schedule batch")
    return requests, payload


def _load_adapter(spec: str, config: Mapping[str, Any]) -> Any:
    if ":" not in spec:
        raise ValueError("adapter must use 'module:factory' syntax")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    adapter = factory(config)
    if not callable(getattr(adapter, "run_episode", None)):
        raise TypeError("adapter must define run_episode(request, episode_id, cadence)")
    return adapter


def _skip_path(output_dir: Path) -> Path:
    return output_dir / ".skip_episodes.json"


def _current_path(output_dir: Path) -> Path:
    return output_dir / ".collect_current_episode.json"


def load_skip_indices(output_dir: Path, config_skips: Any = None) -> set[int]:
    skips: set[int] = set()
    if config_skips is not None:
        skips.update(int(x) for x in config_skips)
    path = _skip_path(output_dir)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        skips.update(int(x) for x in raw.get("indices", []))
    return skips


def existing_episode_ids(output_dir: Path) -> set[str]:
    """Episode ids that already have at least one published state in the pool."""
    path = output_dir / "manifest.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["episode_id"])
        for entry in payload.get("states", {}).values()
        if entry.get("episode_id")
    }


def read_current_episode(output_dir: Path) -> dict[str, Any] | None:
    path = _current_path(output_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def clear_current_episode(output_dir: Path) -> None:
    _current_path(output_dir).unlink(missing_ok=True)


def record_skip_index(output_dir: Path, index: int, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _skip_path(output_dir)
    data: dict[str, Any] = {"indices": [], "reasons": {}}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    indices = {int(x) for x in data.get("indices", [])}
    indices.add(int(index))
    data["indices"] = sorted(indices)
    reasons = dict(data.get("reasons") or {})
    reasons[str(int(index))] = reason
    data["reasons"] = reasons
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect(config: Mapping[str, Any], *, force_dry_run: bool = False) -> dict[str, Any]:
    collection = config["collection"]
    episodes = int(collection["episodes"])
    seed = int(collection["seed"])
    cadence = int(collection["snapshot_cadence_action_chunks"])
    retention = float(collection["successful_snapshot_retention"])
    output_dir = Path(collection["output_dir"])
    pool = StatePool(output_dir)
    sampling = dict(config.get("sampling") or {})
    protocol = dict(config.get("protocol") or {})
    schedule_payload: dict[str, Any] | None = None
    if protocol.get("version") in SCHEDULED_PROTOCOLS:
        if protocol.get("version") == W9B_PROTOCOL_VERSION:
            _validate_w9b_config(config)
        elif protocol.get("version") == W9C_PROTOCOL_VERSION:
            _validate_w9c_config(config)
        requests, schedule_payload = _scheduled_requests(config)
    else:
        requests = sample_perturbations(
            episodes,
            seed,
            dimension_quotas=sampling.get("dimension_quotas"),
            suite_quotas=sampling.get("suite_quotas"),
            levels_by_dimension=sampling.get("levels_by_dimension"),
            factorial_cells=sampling.get("factorial_cells"),
        )
    skip_indices = load_skip_indices(output_dir, collection.get("skip_episode_indices"))
    already_done = existing_episode_ids(output_dir)
    adapter_spec = config.get("adapter")
    if force_dry_run or bool(collection.get("dry_run", False)):
        adapter = DryRunAdapter(config)
        adapter_name = "dry-run"
    elif adapter_spec:
        adapter = _load_adapter(str(adapter_spec), config)
        adapter_name = str(adapter_spec)
    else:
        raise ValueError("non-dry collection requires an adapter 'module:factory' hook")

    if skip_indices:
        print(
            f"COLLECT_SKIP_LIST count={len(skip_indices)} indices={sorted(skip_indices)}",
            flush=True,
        )
    if already_done:
        print(
            f"COLLECT_RESUME_EPISODES already_in_pool={len(already_done)}",
            flush=True,
        )

    created = skipped = retained = seen = 0
    episode_metrics: list[dict[str, Any]] = []
    outcomes = {"success": 0, "failure": 0}
    episodes_skipped_crash = 0
    episodes_skipped_resume = 0
    for request in requests:
        episode_id = request.episode_id or f"ep-{seed:08x}-{request.index:08d}"
        if (
            schedule_payload is not None
            and protocol.get("version") in CLEAN_CONTROL_PROTOCOLS
            and request.init_state_id is None
        ):
            raise ValueError("clean-control schedule row is missing init_state_id")
        if request.index in skip_indices:
            episodes_skipped_crash += 1
            print(
                f"COLLECT_EPISODE_SKIP index={request.index} episode_id={episode_id} "
                f"reason=skip_list",
                flush=True,
            )
            continue
        if episode_id in already_done:
            # Real MuJoCo/GPU rollouts are not bit-identical across process restarts.
            # Re-running hits FileExistsError: same state_key, different payload checksums.
            episodes_skipped_resume += 1
            print(
                f"COLLECT_EPISODE_SKIP index={request.index} episode_id={episode_id} "
                f"reason=already_in_pool",
                flush=True,
            )
            continue
        _current_path(output_dir).write_text(
            json.dumps(
                {
                    "index": request.index,
                    "episode_id": episode_id,
                    "suite": request.suite,
                    "task_id": request.task_id,
                    "init_state_id": request.init_state_id,
                    "batch_id": request.batch_id,
                    "global_episode_index": request.global_episode_index,
                    "dimension": request.dimension,
                    "level": request.level,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"COLLECT_EPISODE_START index={request.index} episode_id={episode_id} "
            f"suite={request.suite} dim={request.dimension} level={request.level}",
            flush=True,
        )
        result = adapter.run_episode(request, episode_id, cadence)
        if schedule_payload is not None:
            expected_task = _scheduled_task_id(request)
            init_mismatch = (
                request.init_state_id is not None
                and result.init_state_id != request.init_state_id
            )
            if (
                result.task_id != expected_task
                or result.suite != request.suite
                or init_mismatch
            ):
                raise ValueError(
                    "scheduled row differs from actual suite/task/init: "
                    f"expected=({request.suite},{expected_task},{request.init_state_id}) "
                    f"actual=({result.suite},{result.task_id},{result.init_state_id})"
                )
        if result.outcome not in outcomes:
            raise ValueError(f"adapter returned invalid outcome {result.outcome!r}")
        outcomes[result.outcome] += 1
        if result.metrics is not None:
            episode_metrics.append(
                {
                    "episode_id": episode_id,
                    "task_id": result.task_id,
                    "suite": request.suite,
                    "perturbation_dimension": request.dimension,
                    "perturbation_level": request.level,
                    "outcome": result.outcome,
                    **dict(result.metrics),
                }
            )
        for snapshot in result.snapshots:
            seen += 1
            if snapshot.step % cadence:
                raise ValueError("adapter returned a snapshot outside configured cadence")
            if not retain_snapshot(
                result.outcome, episode_id, snapshot.step, seed, retention
            ):
                continue
            retained += 1
            metadata = StateMetadata(
                task_id=result.task_id,
                instruction=result.instruction,
                suite=request.suite,
                episode_id=episode_id,
                step=snapshot.step,
                perturb_dim=request.dimension,
                perturb_sub=request.subdimension,
                level=request.level,
                episode_outcome=result.outcome,
                seed=request.seed,
                init_state_id=request.init_state_id,
                clean_task_name=result.clean_task_name,
                bddl_stem=result.bddl_stem,
                libero_flavor=result.libero_flavor
                or ("clean" if request.dimension == "clean" else None),
            )
            write = pool.write_state(
                metadata,
                sim_state=snapshot.sim_state,
                controller_state=snapshot.controller_state,
                rng_state=snapshot.rng_state,
                observations=snapshot.observations,
                proprio=snapshot.proprio,
            )
            created += int(write.created)
            skipped += int(not write.created)
        already_done.add(episode_id)
        print(
            f"COLLECT_EPISODE_DONE index={request.index} outcome={result.outcome} "
            f"snapshots={len(result.snapshots)}",
            flush=True,
        )
    clear_current_episode(output_dir)
    summary = {
        "adapter": adapter_name,
        "episodes": episodes,
        "outcomes": outcomes,
        "snapshots_seen": seen,
        "snapshots_retained": retained,
        "states_created": created,
        "states_idempotently_skipped": skipped,
        "episodes_skipped_crash_list": episodes_skipped_crash,
        "episodes_skipped_already_in_pool": episodes_skipped_resume,
        "quotas": summarize(requests),
        "manifest": str(pool.manifest_path),
        "episode_metrics": episode_metrics,
    }
    if schedule_payload is not None:
        if protocol.get("version") in {
            PRE_A3_PROTOCOL_VERSION,
            PRE_C0_PROTOCOL_VERSION,
            R7_PROTOCOL_VERSION,
        }:
            summary["provenance"] = {
                "protocol_version": protocol["version"],
                "design": protocol.get("design")
                or (config.get("sampling") or {}).get("design"),
                "design_sha256": schedule_payload.get("design_sha256")
                or protocol.get("design_sha256"),
                "git_commit": _git_commit(),
            }
        else:
            summary["provenance"] = {
                "protocol_version": protocol["version"],
                "schedule_sha256": protocol["schedule_sha256"],
                "schedule_path": protocol["schedule_path"],
                "schedule_batch_id": int(collection.get("schedule_batch_id", 1)),
                "git_commit": _git_commit(),
            }
        summary["scheduled_episodes"] = [
            {
                "episode_id": request.episode_id,
                "global_episode_index": request.global_episode_index,
                "batch_id": request.batch_id,
                "request_index": request.index,
                "suite": request.suite,
                "task_id": request.task_id,
                "init_state_id": request.init_state_id,
                "policy_seed": request.seed,
            }
            for request in requests
        ]
    return summary

#!/usr/bin/env python3
"""Collect the V6 exact-root C / R-same / R-new Stage-0 experiment.

This is intentionally a *single-source* collector.  It never loads a fallback
policy, never chooses the best of K refreshes, and always evaluates a frozen
candidate prefix followed by the same seeded downstream controller ``mu``.
The K=4 R-new branches estimate the expected K=1 refresh value; their maximum
is recorded only as a diagnostic and is not used by the eligibility gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "rase-v6-stage0-branch/v1"
ROOT_SCHEMA = "rase-v6-stage0-root/v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return sha256_bytes(array.tobytes())


def snapshot_sha256(snapshot: Any) -> str:
    """Hash the physical simulator state retained by a ForkableEnv snapshot."""
    payload = getattr(snapshot, "payload", {})
    for key in ("sim_state", "mujoco_data"):
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if value is None:
            continue
        if isinstance(value, Mapping):
            digest = hashlib.sha256()
            for name in sorted(value):
                array = np.ascontiguousarray(np.asarray(value[name]))
                digest.update(str(name).encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
            return digest.hexdigest()
        array = np.ascontiguousarray(np.asarray(value))
        return sha256_bytes(array.tobytes())
    # This branch is only a provenance fallback; it is not used as a restore
    # mechanism.  ForkableEnv snapshots used by RASE expose sim_state.
    return sha256_bytes(repr(snapshot).encode("utf-8"))


def terminal_values(term: Any, trunc: Any, info: Any) -> tuple[bool, bool]:
    from rase.collect.policy_step import success_from_info

    terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    return terminal, bool(success_from_info(info)) if terminal else False


def flatten_numeric_observation(value: Any, *, prefix: str = "obs") -> dict[str, np.ndarray]:
    """Persist all numeric observation leaves without pickle/object arrays."""
    result: dict[str, np.ndarray] = {}

    def visit(item: Any, name: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], f"{name}__{str(key).replace('.', '_')}")
            return
        if isinstance(item, (str, bytes, bytearray)) or item is None:
            return
        try:
            array = np.asarray(item)
        except Exception:
            return
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            return
        result[name] = array.copy()

    visit(value, prefix)
    return result


def proprio_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    """Store a stable numeric proprio audit view, independent of later features."""
    robot = observation.get("robot_state")
    leaves = flatten_numeric_observation(robot, prefix="proprio") if robot is not None else {}
    if not leaves:
        leaves = {
            name: value for name, value in flatten_numeric_observation(observation).items()
            if "pixel" not in name and "image" not in name
        }
    vectors = [np.asarray(leaves[name], dtype=np.float32).reshape(-1) for name in sorted(leaves)]
    return np.concatenate(vectors).astype(np.float32, copy=False) if vectors else np.empty(0, dtype=np.float32)


def fixed_prefix(chunk: np.ndarray, *, length: int, label: str) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7 or array.shape[0] < length:
        raise ValueError(f"{label} must supply at least {length} env actions [T,7], got {array.shape}")
    return array[:length].copy()


def force_fresh_inference(
    bundle: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    task: str,
    boundary_step: int,
    generation_seed: int,
    horizon: int,
    temperature: float | None,
) -> tuple[np.ndarray, Any]:
    """Run one fresh forward without mutating another arm's queue or RNG state."""
    from rase.collect.candidates import seed_everything
    from rase.collect.policy_step import (
        capture_inference_event,
        clear_policy_queues,
        policy_state_restore,
        policy_state_snapshot,
    )

    before = policy_state_snapshot(bundle)
    try:
        bundle["policy"].reset()
        clear_policy_queues(bundle["policy"])
        seed_everything(int(generation_seed))
        return capture_inference_event(
            bundle,
            observation,
            task=task,
            boundary_step=boundary_step,
            generation_seed=int(generation_seed),
            horizon=horizon,
            temperature=temperature,
        )
    finally:
        policy_state_restore(bundle, before)


def capture_boundary(
    restored: Any,
    bundle: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    source_temperature: float | None,
) -> dict[str, Any]:
    """Replay source until a cursor inside its first native inference event."""
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    cursor = int(root["cursor"])
    native_horizon = int(root["native_chunk_horizon"])
    restored.forkable.restore(
        restored.snapshot, check_task_fingerprint=restored.check_task_fingerprint,
    )
    single = restored.handle.vector_env.envs[0]
    vector_env = restored.handle.vector_env
    instruction = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
    source = InProcessSmolVLAContinuation(
        bundle, temperature=source_temperature, seed=int(root["source_generation_seed"]),
    )
    source.enable_capture(horizon=native_horizon)
    source.note_boundary_step(cursor)
    source.reset_metrics()
    source.reset()
    observation = observation_from_libero_env(single)
    prefix_actions: list[np.ndarray] = []
    horizon = int(getattr(single, "_max_episode_steps", 600))
    terminal = False
    success = False
    try:
        for step in range(cursor):
            if current_timestep(restored.handle.control_env) >= horizon:
                return {"available": False, "reason": "horizon_before_cursor", "prefix_steps": step}
            action = np.asarray(source.act(observation, task=instruction), dtype=np.float32).reshape(7)
            event = source.current_inference_event()
            if event is None:
                return {"available": False, "reason": "missing_inference_event", "prefix_steps": step}
            if event.env_chunk.shape[0] < native_horizon:
                return {
                    "available": False,
                    "reason": "native_chunk_shorter_than_contract",
                    "prefix_steps": step,
                    "native_chunk_steps": int(event.env_chunk.shape[0]),
                }
            prefix_actions.append(action.copy())
            observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                return {
                    "available": False,
                    "reason": "terminal_before_cursor",
                    "prefix_steps": step + 1,
                    "terminal_success": bool(success),
                }
        event = source.current_inference_event()
        consumed = source.consumed_in_current_event()
        if event is None or consumed != cursor:
            return {
                "available": False,
                "reason": "cursor_not_in_first_native_event",
                "prefix_steps": cursor,
                "consumed": int(consumed),
            }
        if event.env_chunk.shape[0] <= cursor:
            return {"available": False, "reason": "empty_continue_suffix", "prefix_steps": cursor}
        boundary_snapshot = restored.forkable.snapshot()
        boundary_observation = observation_from_libero_env(single)
        return {
            "available": True,
            "instruction": instruction,
            "snapshot": boundary_snapshot,
            "snapshot_sha256": snapshot_sha256(boundary_snapshot),
            "observation": boundary_observation,
            "proprio": proprio_from_observation(boundary_observation),
            "source_event": event,
            "old_chunk": event.env_chunk.copy(),
            "old_chunk_sha256": array_sha256(event.env_chunk),
            "prefix_actions": np.asarray(prefix_actions, dtype=np.float32),
            "prefix_action_sha256": array_sha256(np.asarray(prefix_actions, dtype=np.float32)),
            "source_metrics": source.metrics(),
            "simulator_timestep": int(current_timestep(restored.handle.control_env)),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": "source_prefix_error",
            "prefix_steps": len(prefix_actions),
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:1000],
        }


def make_branch_row(
    root: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    arm: str,
    arm_index: int | None,
    candidate: np.ndarray,
    candidate_generation_seed: int | None,
    candidate_full_sha256: str | None,
    downstream_seed: int,
    downstream_temperature: float | None,
    result: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "root_id": root["root_id"],
        "state_key": root["state_key"],
        "task_id": root["task_id"],
        "suite": root["suite"],
        "perturb_dim": root["perturb_dim"],
        "perturb_level": root["perturb_level"],
        "cursor": int(root["cursor"]),
        "native_chunk_horizon": int(root["native_chunk_horizon"]),
        "actual_cursor_fraction": float(root["cursor"]) / int(root["native_chunk_horizon"]),
        "root_snapshot_sha256": boundary["snapshot_sha256"],
        "source_generation_seed": int(root["source_generation_seed"]),
        "source_temperature": None if root.get("source_temperature") is None else float(root["source_temperature"]),
        "arm": arm,
        "arm_index": arm_index,
        "candidate_generation_seed": candidate_generation_seed,
        "candidate_chunk_steps": int(candidate.shape[0]),
        "candidate_chunk_sha256": array_sha256(candidate),
        "candidate_full_chunk_sha256": candidate_full_sha256,
        "downstream_controller": "same_source_fixed_mu",
        "downstream_seed": int(downstream_seed),
        "downstream_temperature": downstream_temperature,
        "success": bool(result.success),
        "rollout": result.to_dict(),
        "source_prefix_action_sha256": boundary["prefix_action_sha256"],
        "source_inference_event_id": boundary["source_event"].inference_event_id,
    }


def collect_one_root(
    root: dict[str, Any],
    *,
    pool: Any,
    bundle: Mapping[str, Any],
    adapter: Mapping[str, Any],
    protocol: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        evaluate_candidate,
        restore_pool_state,
    )

    source_temperature = protocol.get("source_temperature", 0.5)
    source_temperature = None if source_temperature is None else float(source_temperature)
    downstream_temperature = protocol.get("downstream_temperature", source_temperature)
    downstream_temperature = None if downstream_temperature is None else float(downstream_temperature)
    r_new_k = int(protocol.get("r_new_k", 4))
    root = dict(root)
    root["source_temperature"] = source_temperature
    started = time.perf_counter()
    restored = restore_pool_state(
        pool,
        str(root["state_key"]),
        libero_plus_root=adapter.get("libero_plus_root"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        strict_fingerprint=bool(protocol.get("strict_fingerprint", False)),
    )
    try:
        boundary = capture_boundary(
            restored, bundle, root, source_temperature=source_temperature,
        )
        if not boundary["available"]:
            return {
                "schema_version": ROOT_SCHEMA,
                "status": "unavailable",
                "root": root,
                "reason": boundary.get("reason"),
                "boundary": {key: value for key, value in boundary.items() if key not in {"snapshot", "observation", "source_event"}},
                "branches": [],
                "elapsed_s": round(time.perf_counter() - started, 6),
            }
        old_chunk = np.asarray(boundary["old_chunk"], dtype=np.float32)
        cursor = int(root["cursor"])
        continue_chunk = fixed_prefix(
            old_chunk[cursor:], length=old_chunk.shape[0] - cursor, label="continue suffix",
        )
        candidate_horizon = int(continue_chunk.shape[0])
        observation = boundary["observation"]
        instruction = str(boundary["instruction"])
        native_horizon = int(root["native_chunk_horizon"])
        # Freeze every candidate before any branch rollout.  Later outcome
        # branches therefore cannot perturb the source policy queue/RNG used
        # to create candidates for another arm.
        _same_first, same_event = force_fresh_inference(
            bundle,
            observation,
            task=instruction,
            boundary_step=cursor,
            generation_seed=int(root["source_generation_seed"]),
            horizon=native_horizon,
            temperature=source_temperature,
        )
        same_chunk = fixed_prefix(same_event.env_chunk, length=candidate_horizon, label="R-same chunk")
        planned_new_seeds = [int(value) for value in root.get("r_new_generation_seeds", [])]
        if len(planned_new_seeds) != r_new_k:
            raise ValueError(
                f"root {root['root_id']} has {len(planned_new_seeds)} R-new seeds; expected {r_new_k}"
            )
        if int(root["source_generation_seed"]) in planned_new_seeds or len(set(planned_new_seeds)) != r_new_k:
            raise ValueError("R-new seeds must be unique and distinct from the matched source seed")
        new_events: list[tuple[int, Any, np.ndarray]] = []
        for index, generation_seed in enumerate(planned_new_seeds):
            _first, event = force_fresh_inference(
                bundle,
                observation,
                task=instruction,
                boundary_step=cursor,
                generation_seed=generation_seed,
                horizon=native_horizon,
                temperature=source_temperature,
            )
            new_events.append((index, event, fixed_prefix(event.env_chunk, length=candidate_horizon, label=f"R-new[{index}] chunk")))

        branch_state = dataclasses.replace(restored, snapshot=boundary["snapshot"])
        downstream_seed = int(root["downstream_seed"])

        def evaluate(chunk: np.ndarray) -> Any:
            downstream = InProcessSmolVLAContinuation(
                bundle, temperature=downstream_temperature, seed=downstream_seed,
            )
            return evaluate_candidate(branch_state, chunk, downstream)

        branches: list[dict[str, Any]] = []
        c_result = evaluate(continue_chunk)
        branches.append(make_branch_row(
            root, boundary, arm="C", arm_index=None, candidate=continue_chunk,
            candidate_generation_seed=None, candidate_full_sha256=boundary["old_chunk_sha256"],
            downstream_seed=downstream_seed, downstream_temperature=downstream_temperature,
            result=c_result,
        ))
        same_result = evaluate(same_chunk)
        branches.append(make_branch_row(
            root, boundary, arm="R_same", arm_index=None, candidate=same_chunk,
            candidate_generation_seed=int(root["source_generation_seed"]),
            candidate_full_sha256=array_sha256(same_event.env_chunk),
            downstream_seed=downstream_seed, downstream_temperature=downstream_temperature,
            result=same_result,
        ))
        for index, event, chunk in new_events:
            result = evaluate(chunk)
            branches.append(make_branch_row(
                root, boundary, arm="R_new", arm_index=index, candidate=chunk,
                candidate_generation_seed=int(event.candidate_generation_seed),
                candidate_full_sha256=array_sha256(event.env_chunk),
                downstream_seed=downstream_seed, downstream_temperature=downstream_temperature,
                result=result,
            ))

        if len({row["root_snapshot_sha256"] for row in branches}) != 1:
            raise AssertionError("branch root snapshot drift")
        if len({row["downstream_seed"] for row in branches}) != 1:
            raise AssertionError("fixed downstream seed drift")
        if branches[1]["candidate_generation_seed"] != root["source_generation_seed"]:
            raise AssertionError("R-same did not use the source event generation seed")

        artifact_arrays = {
            "old_chunk": old_chunk,
            "continue_suffix": continue_chunk,
            "refresh_same": same_chunk,
            "refresh_new": np.stack([chunk for _index, _event, chunk in new_events]),
            "proprio": np.asarray(boundary["proprio"], dtype=np.float32),
            "prefix_actions": np.asarray(boundary["prefix_actions"], dtype=np.float32),
        }
        artifact_arrays.update(flatten_numeric_observation(observation))
        artifact_path = artifact_dir / f"{root['root_id']}.npz"
        atomic_npz(artifact_path, artifact_arrays)
        return {
            "schema_version": ROOT_SCHEMA,
            "status": "complete",
            "root": root,
            "boundary": {
                "snapshot_sha256": boundary["snapshot_sha256"],
                "simulator_timestep": boundary["simulator_timestep"],
                "old_chunk_sha256": boundary["old_chunk_sha256"],
                "source_inference_event_id": boundary["source_event"].inference_event_id,
                "source_event_generation_seed": boundary["source_event"].candidate_generation_seed,
                "source_event_chunk_steps": int(boundary["source_event"].env_chunk.shape[0]),
                "candidate_horizon": candidate_horizon,
                "prefix_action_sha256": boundary["prefix_action_sha256"],
                "source_metrics": boundary["source_metrics"],
                "artifact": str(artifact_path),
            },
            "branches": branches,
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
    finally:
        restored.close()


def merge_jsonl(root_dir: Path, output: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(root_dir.glob("*.json")):
        record = read_json(path)
        rows.extend(record.get("branches") or [])
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, help="override config.pool after checking the server asset name")
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    plan = read_json(args.root_plan.resolve())
    if plan.get("schema_version") != "rase-v6-stage0-root-plan/v1":
        raise ValueError("unsupported V6 Stage-0 root-plan schema")
    if plan.get("selection_outcomes_used") is not False:
        raise ValueError("V6 Stage 0 root plan must certify outcome-blind selection")
    roots = plan.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("root plan has no roots")
    protocol = dict(config.get("protocol") or {})
    adapter = dict(config.get("adapter_config") or {})
    expected_horizon = int(protocol.get("native_chunk_horizon", 10))
    expected_k = int(protocol.get("r_new_k", 4))
    for root in roots:
        if not isinstance(root, dict):
            raise ValueError("root plan entries must be objects")
        if int(root.get("native_chunk_horizon", -1)) != expected_horizon:
            raise ValueError(
                "root plan native_chunk_horizon disagrees with the frozen protocol"
            )
        if len(root.get("r_new_generation_seeds") or []) != expected_k:
            raise ValueError("root plan R-new seed count disagrees with the frozen protocol")
    pool_value = args.pool or config.get("pool") or plan.get("pool")
    if not pool_value:
        raise ValueError("supply --pool or config.pool")
    pool_path = Path(pool_value)
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    output = args.output_dir.resolve()
    roots_dir = output / "roots"
    artifacts_dir = output / "decision_artifacts"
    if args.fresh_run and output.exists():
        raise SystemExit(f"--fresh-run refuses existing output: {output}")
    roots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import load_smolvla_policy_bundle
    from rase.collect.state_pool import StatePool

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    pool = StatePool(pool_path.resolve())
    run_manifest = {
        "schema_version": "rase-v6-stage0-run/v1",
        "config": str(args.config.resolve()),
        "config_sha256": sha256_bytes(args.config.resolve().read_bytes()),
        "root_plan": str(args.root_plan.resolve()),
        "root_plan_sha256": sha256_bytes(args.root_plan.resolve().read_bytes()),
        "pool": str(pool_path.resolve()),
        "protocol": protocol,
        "source_policy": "SmolVLA only",
        "selection_outcomes_used": False,
        "fixed_downstream_contract": "same source VLA; per-root identical seed and temperature across C/R arms",
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    failures = 0
    for index, root in enumerate(roots):
        target = roots_dir / f"{root['root_id']}.json"
        if target.is_file():
            print(f"V6 Stage0 skip {index + 1}/{len(roots)} root={root['root_id']}", flush=True)
            continue
        try:
            record = collect_one_root(
                root, pool=pool, bundle=bundle, adapter=adapter,
                protocol=protocol, artifact_dir=artifacts_dir,
            )
        except Exception as exc:
            failures += 1
            record = {
                "schema_version": ROOT_SCHEMA,
                "status": "error",
                "root": root,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "branches": [],
            }
        atomic_json(target, record)
        print(
            f"V6 Stage0 {index + 1}/{len(roots)} root={root['root_id']} status={record['status']}",
            flush=True,
        )
    branch_count = merge_jsonl(roots_dir, output / "stage0_records.jsonl")
    summary = {
        "n_planned_roots": len(roots),
        "n_root_files": len(list(roots_dir.glob("*.json"))),
        "n_branch_rows": branch_count,
        "uncaught_root_errors": failures,
        "records": str(output / "stage0_records.jsonl"),
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if branch_count else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect grouped source-trajectory boundaries and persistent-takeover labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_chw(value: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def terminal_values(term: Any, trunc: Any, info: Any) -> tuple[bool, bool]:
    from rase.collect.policy_step import success_from_info
    terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    return terminal, bool(success_from_info(info)) if terminal else False


def stack_or_empty(sequence: list[np.ndarray]) -> np.ndarray:
    if not sequence:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(sequence)


def is_invalid_action_token_error(exc: BaseException) -> bool:
    """Return whether Pi0Fast failed its explicit action grammar contract."""
    return (
        isinstance(exc, AssertionError)
        and "Token sequence does not start with ['Action', ':']" in str(exc)
    )


def observable_state_summary(control_env: Any, *, tag: str) -> dict[str, Any]:
    """Snapshot observable delay/sampling state for bisection diagnostics."""
    task_env = getattr(control_env, "env", None)
    observables = getattr(task_env, "_observables", None)
    if not observables:
        return {"tag": tag, "detail": "no task-env _observables"}
    names = [name for name in ("agentview_image", "robot0_eye_in_hand_image") if name in observables]
    if not names:
        names = sorted(observables)[:4]
    return {
        "tag": tag,
        "observables": {
            name: {
                "sampled": bool(getattr(observables[name], "_sampled", None)),
                "delay": float(getattr(observables[name], "_current_delay", float("nan"))),
                "time_since_last_sample": float(
                    getattr(observables[name], "_time_since_last_sample", float("nan"))
                ),
            }
            for name in names
        },
    }


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Prevent counterfactual simulator restore from perturbing the source VLA."""
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def persistent_branch(branch: Any, snapshot: Any, client: Any,
                      instruction: str, *, record_chunk_trace: bool = False,
                      return_action_trace: bool = False) -> dict[str, Any]:
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    branch.forkable.restore(snapshot, check_task_fingerprint=branch.check_task_fingerprint)
    single = branch.handle.vector_env.envs[0]
    vector_env = branch.handle.vector_env
    horizon = int(getattr(single, "_max_episode_steps", 600))
    observation = observation_from_libero_env(single)
    policy = OracleChunkContinuation(
        client, instruction=instruction, record_chunk_trace=record_chunk_trace,
    )
    policy.bind_control_env(branch.handle.control_env)
    policy.reset()
    steps = 0
    first_action = None
    action_trace: list[np.ndarray] = []
    while current_timestep(branch.handle.control_env) < horizon:
        action = np.asarray(policy.act(observation, task=instruction), dtype=np.float32).reshape(-1, 7)[0]
        action_trace.append(action.copy())
        if first_action is None:
            first_action = action.copy()
        observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            trace = np.stack(action_trace).astype(np.float32, copy=False)
            result = {"success": success, "steps": steps, "first_action": first_action,
                    "action_trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
                    "action_trace_shape": list(trace.shape)}
            if record_chunk_trace:
                result["chunk_query_records"] = policy.chunk_query_records
            if return_action_trace:
                result["action_trace"] = trace.copy()
            return result
    trace = (np.stack(action_trace).astype(np.float32, copy=False)
             if action_trace else np.empty((0, 7), dtype=np.float32))
    result = {"success": False, "steps": steps,
            "first_action": np.zeros(7, dtype=np.float32) if first_action is None else first_action,
            "action_trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
            "action_trace_shape": list(trace.shape)}
    if record_chunk_trace:
        result["chunk_query_records"] = policy.chunk_query_records
    if return_action_trace:
        result["action_trace"] = trace.copy()
    return result


def collect_trajectory(*, pool: Any, state_key: str, bundle: Any, policy_id: str,
                       rollout_seed_value: int, seed_index: int, client: Any,
                       boundaries: list[int], image_size: int,
                       libero_plus_root: str | None,
                       bookkeeping: str = "full", skip_oft: bool = False,
                       debug: bool = False, rollout_index: int = 0,
                       temporal_history: int = 0,
                       record_oft_trace_hash: bool = False,
                       record_oft_chunk_trace: bool = False) -> dict[str, Any]:
    from rase.collect.forked_rollout import InProcessLeRobotContinuation, restore_pool_state
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.risk.canonical_action import summary_from_chunk
    from rase.risk.vla_action_adapters import create_vla_adapter

    main = restore_pool_state(pool, state_key, libero_plus_root=libero_plus_root)
    branch = restore_pool_state(pool, state_key, libero_plus_root=libero_plus_root)
    source = InProcessLeRobotContinuation(bundle, seed=rollout_seed_value)
    adapter = create_vla_adapter(policy_id)
    rows: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    source_actions: list[np.ndarray] = []
    oft_actions: list[np.ndarray] = []
    source_summaries: list[np.ndarray] = []
    oft_summaries: list[np.ndarray] = []
    source_action_trace: list[np.ndarray] = []
    # Optional R9 causal history.  These arrays are captured from the exact
    # observation consumed before each source action; they are never read from
    # a future frame or from the OFT branch.
    temporal_images: list[np.ndarray] = []
    temporal_proprios: list[np.ndarray] = []
    temporal_boundary_records: list[dict[str, np.ndarray]] = []
    obs_debug: list[dict[str, Any]] = []
    # Source rollout must be side-effect-free: nothing but the environment
    # itself may touch the main trajectory.  Boundary feature capture and every
    # counterfactual branch therefore run after the source outcome is fixed,
    # from restored snapshot environments.  In-loop force-updated observable
    # reads (``obs_only``) resample observable delays/RNG and are retained only
    # as the bisection control that reproduces the historical 149-step drift.
    boundary_snapshots: list[Any] = []
    try:
        main.forkable.restore(main.snapshot, check_task_fingerprint=main.check_task_fingerprint)
        single = main.handle.vector_env.envs[0]
        vector_env = main.handle.vector_env
        horizon = int(getattr(single, "_max_episode_steps", 600))
        instruction = str(getattr(single, "task_description", "") or main.loaded.metadata.instruction)
        observation = observation_from_libero_env(single)
        source.reset_metrics(); source.reset()
        elapsed = 0
        success = False
        stop_reason = "horizon"
        policy_inference_error: dict[str, Any] | None = None
        boundary_set = set(boundaries)
        # The saved snapshot may already be at a nonzero environment timestep.
        # Match ``run_one_forked_rollout`` by using the simulator's absolute
        # horizon, not ``elapsed < horizon`` from this snapshot.
        while current_timestep(main.handle.control_env) < horizon:
            if temporal_history:
                pixels = observation["pixels"]
                agent = np.asarray(pixels["image"])[0]
                wrist = np.asarray(pixels["image2"])[0]
                temporal_images.append(np.stack([
                    resize_chw(agent, image_size), resize_chw(wrist, image_size),
                ]))
                _, _, causal_proprio = raw_libero_to_oracle_arrays(
                    main.handle.control_env, force_update=False
                )
                temporal_proprios.append(np.asarray(causal_proprio, dtype=np.float32))
            try:
                source_action = np.asarray(
                    source.act(observation, task=instruction), dtype=np.float32
                ).reshape(-1, 7)[0]
            except AssertionError as exc:
                if not is_invalid_action_token_error(exc):
                    raise
                # This is a deploy-time source-policy failure, not a collector
                # failure.  Once the initial ten-step proposal is complete, the
                # t=0 risk row remains fully causal and eligible: record the
                # decoder failure as the source outcome instead of crashing the
                # entire cohort.  Before ten actions there is no canonical t=0
                # proposal, so fail closed rather than manufacture a feature.
                if len(source_action_trace) < 10:
                    raise RuntimeError(
                        "source policy produced an invalid action token sequence "
                        "before a complete t=0 action proposal was available: "
                        f"state_key={state_key} elapsed_source_steps={elapsed} "
                        f"simulator_timestep={current_timestep(main.handle.control_env)}"
                    ) from exc
                policy_inference_error = {
                    "type": "invalid_action_token_sequence",
                    "exception_type": type(exc).__name__,
                    "elapsed_source_steps": int(elapsed),
                    "simulator_timestep": int(current_timestep(main.handle.control_env)),
                    "initial_10_step_proposal_complete": True,
                }
                stop_reason = "policy_inference_error"
                success = False
                break
            source_action_trace.append(source_action.copy())
            if elapsed in boundary_set:
                row_base = {
                    "state_key": state_key,
                    "task_id": main.loaded.metadata.task_id,
                    "episode_id": main.loaded.metadata.episode_id,
                    "suite": main.loaded.metadata.suite,
                    "policy_id": policy_id,
                    "seed_index": seed_index,
                    "rollout_seed": rollout_seed_value,
                    "group_id": f"{state_key}:{policy_id}:seed{seed_index}"
                                + (f":rep{rollout_index}" if rollout_index else ""),
                    "elapsed_source_steps": elapsed,
                    "simulator_timestep": current_timestep(main.handle.control_env),
                    "instruction": instruction,
                }
                if bookkeeping in ("snapshot_only", "full"):
                    boundary_snapshots.append(main.forkable.snapshot())
                    row_base["snapshot_recorded"] = True
                else:
                    row_base["snapshot_recorded"] = False
                if bookkeeping == "obs_only":
                    # Legacy in-loop force-updated observable read: perturbs the
                    # source trajectory (reproduces the 149-step drift).
                    if debug:
                        obs_debug.append(observable_state_summary(main.handle.control_env, tag=f"obs_before_t{elapsed}"))
                    _, _, proprio = raw_libero_to_oracle_arrays(main.handle.control_env)
                    if debug:
                        obs_debug.append(observable_state_summary(main.handle.control_env, tag=f"obs_after_t{elapsed}"))
                    cache = single._get_observations(force_update=True) if hasattr(single, "_get_observations") else None
                    if cache is None:
                        pixels = observation["pixels"]
                        agent = np.asarray(pixels["image"])[0]
                        wrist = np.asarray(pixels["image2"])[0]
                    else:
                        agent = np.asarray(cache["agentview_image"])
                        wrist = np.asarray(cache["robot0_eye_in_hand_image"])
                    images.append(np.stack([resize_chw(agent, image_size), resize_chw(wrist, image_size)]))
                    proprios.append(np.asarray(proprio, dtype=np.float32))
                    source_actions.append(source_action.astype(np.float32))
                    source_summaries.append(
                        summary_from_chunk(adapter.to_canonical(source_action)).cpu().numpy().astype(np.float32)
                    )
                    row_base["obs_recorded"] = True
                else:
                    source_actions.append(source_action.astype(np.float32))
                    row_base["obs_recorded"] = False
                if temporal_history:
                    start = max(0, elapsed - temporal_history + 1)
                    image_window = temporal_images[start:elapsed + 1]
                    proprio_window = temporal_proprios[start:elapsed + 1]
                    action_trace = [np.asarray(value, dtype=np.float32)
                                    for value in source_action_trace[start:elapsed + 1]]
                    pad = temporal_history - len(image_window)
                    temporal_boundary_records.append({
                        "image": np.concatenate([
                            np.zeros((pad, 2, 3, image_size, image_size), dtype=np.uint8),
                            np.stack(image_window),
                        ], axis=0),
                        "proprio": np.concatenate([
                            np.zeros((pad, 8), dtype=np.float32),
                            np.stack(proprio_window),
                        ], axis=0),
                        "action": np.concatenate([
                            np.zeros((pad, 7), dtype=np.float32),
                            np.stack(action_trace),
                        ], axis=0),
                    })
                rows.append(row_base)
            observation, _, term, trunc, info = vector_env.step(as_batched_action(source_action))
            elapsed += 1
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                stop_reason = "success" if success else "terminal_failure"
                break
        for row in rows:
            remaining = elapsed - int(row["elapsed_source_steps"])
            row.update({
                "source_final_success": bool(success),
                "source_total_steps": elapsed,
                "source_remaining_steps": remaining,
                "source_success_within_8": bool(success and remaining <= 8),
                "source_success_within_16": bool(success and remaining <= 16),
                "source_success_within_32": bool(success and remaining <= 32),
                "source_stop_reason": stop_reason,
                "source_policy_inference_error": policy_inference_error,
            })
        # Source rollout is now complete.  Everything that follows runs in the
        # separate restored ``branch`` environment inside ``preserve_rng_state``,
        # so it cannot affect the source policy queues, native samplers, or the
        # source outcome provenance fixed above.
        if bookkeeping == "full":
            # Feature capture is strictly post-hoc: restore the boundary
            # snapshot into the branch env, then read observations there.  This
            # yields the same deployable features the source policy consumed
            # (same sim state) without touching the live main trajectory.
            for row, snapshot, boundary_action in zip(rows, boundary_snapshots, source_actions, strict=True):
                with preserve_rng_state():
                    branch.forkable.restore(snapshot, check_task_fingerprint=branch.check_task_fingerprint)
                    branch_single = branch.handle.vector_env.envs[0]
                    branch_observation = observation_from_libero_env(branch_single)
                    _, _, proprio = raw_libero_to_oracle_arrays(branch.handle.control_env)
                    pixels = branch_observation["pixels"]
                    agent = np.asarray(pixels["image"])[0]
                    wrist = np.asarray(pixels["image2"])[0]
                    images.append(np.stack([resize_chw(agent, image_size), resize_chw(wrist, image_size)]))
                    proprios.append(np.asarray(proprio, dtype=np.float32))
                    source_summaries.append(
                        summary_from_chunk(adapter.to_canonical(boundary_action)).cpu().numpy().astype(np.float32)
                    )
                    if client is not None and not skip_oft:
                        oft = persistent_branch(
                            branch, snapshot, client, instruction,
                            record_chunk_trace=record_oft_chunk_trace,
                        )
                        row.update({
                            "persistent_success_if_enter_now": bool(oft["success"]),
                            "persistent_teacher_steps_if_enter_now": int(oft["steps"]),
                            "counterfactual_timing": "after_source_rollout",
                        })
                        if record_oft_trace_hash or record_oft_chunk_trace:
                            row.update({
                                "persistent_action_trace_sha256": oft["action_trace_sha256"],
                                "persistent_action_trace_shape": oft["action_trace_shape"],
                            })
                        if record_oft_chunk_trace:
                            row["persistent_chunk_query_records"] = oft["chunk_query_records"]
                        oft_action = np.asarray(oft["first_action"], dtype=np.float32)
                        oft_actions.append(oft_action)
                        oft_summaries.append(
                            summary_from_chunk(create_vla_adapter("oft").to_canonical(oft_action)).cpu().numpy().astype(np.float32)
                        )
                    else:
                        row.update({
                            "persistent_success_if_enter_now": None,
                            "persistent_teacher_steps_if_enter_now": None,
                            "counterfactual_timing": "skipped",
                        })
            for row in rows:
                if "counterfactual_timing" not in row:
                    row["counterfactual_timing"] = "skipped"
                    row.setdefault("persistent_success_if_enter_now", None)
                    row.setdefault("persistent_teacher_steps_if_enter_now", None)
        else:
            for row in rows:
                row.update({
                    "persistent_success_if_enter_now": None,
                    "persistent_teacher_steps_if_enter_now": None,
                    "counterfactual_timing": "skipped",
                })
        return {
            "rows": rows, "image": stack_or_empty(images),
            "proprio": stack_or_empty(proprios),
            "source_action": stack_or_empty(source_actions),
            "oft_action": stack_or_empty(oft_actions),
            "source_action_summary": stack_or_empty(source_summaries),
            "oft_action_summary": stack_or_empty(oft_summaries),
            "source_action_trace": stack_or_empty(source_action_trace),
            "temporal_image_history": (
                np.stack([row["image"] for row in temporal_boundary_records])
                if temporal_boundary_records else np.empty((0, 0, 2, 3, image_size, image_size), dtype=np.uint8)
            ),
            "temporal_proprio_history": (
                np.stack([row["proprio"] for row in temporal_boundary_records])
                if temporal_boundary_records else np.empty((0, 0, 8), dtype=np.float32)
            ),
            "temporal_action_history": (
                np.stack([row["action"] for row in temporal_boundary_records])
                if temporal_boundary_records else np.empty((0, 0, 7), dtype=np.float32)
            ),
            "obs_debug": obs_debug,
            "source_success": bool(success), "source_steps": elapsed,
            "stop_reason": stop_reason,
            "policy_inference_error": policy_inference_error,
            "policy_metrics": source.metrics(),
        }
    finally:
        branch.close(); main.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed-index", type=int, required=True,
                        help="rollout seed candidate index (>=0; deterministic via rollout_seed). "
                        "R6-B1.2 froze seed indices {0,1}; R6-C.1B uses new indices "
                        "(Pi0.5 2-3, Pi0Fast 1) with the seed-derivation-version field below.")
    parser.add_argument("--rollout-index", type=int, default=0,
                        help="rollout repetition index inside the same seed index (0,1,...). "
                        "Used by the R6-C.1B reproducibility protocol: two repeated rollouts "
                        "of the same (state, policy, seed) must be stable; a third run decides "
                        "when the first two disagree.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boundary", type=int, action="append", default=[])
    parser.add_argument("--state-key", action="append", default=[])
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--bookkeeping-mode", choices=("none", "snapshot_only", "obs_only", "full"),
        default="full",
        help="bisection control for in-loop boundary bookkeeping: none (source only), "
        "snapshot_only (forkable.snapshot only), obs_only (force-updated observable "
        "read only), full (production feature capture).",
    )
    parser.add_argument("--skip-oft", action="store_true",
                        help="skip counterfactual persistent-OFT branches (bisection only).")
    parser.add_argument("--no-oracle", action="store_true",
                        help="skip the OFT oracle client entirely; forces --skip-oft.")
    parser.add_argument("--debug", action="store_true",
                        help="record observable delay/sampling state around boundary reads.")
    parser.add_argument("--temporal-history", type=int, default=0,
                        help="optional causal observation/action history length for R9; "
                             "zero preserves the legacy R6/R7 NPZ contract.")
    parser.add_argument(
        "--record-oft-trace-hash", action="store_true",
        help="record SHA256 and shape of each full persistent-OFT action trace; "
             "diagnostic metadata only, never a model input.",
    )
    parser.add_argument(
        "--record-oft-chunk-trace", action="store_true",
        help="record SHA256/shape for every OFT chunk-query input and output; "
             "diagnostic metadata only, never a model input.",
    )
    args = parser.parse_args()
    if args.temporal_history < 0 or args.temporal_history > 16:
        raise ValueError("--temporal-history must be in [0, 16]")
    boundaries = args.boundary or [0, 16, 32, 64, 96, 128]
    if boundaries != sorted(set(boundaries)) or boundaries[0] != 0:
        raise ValueError("boundaries must be sorted, unique, and start at zero")
    skip_oft = args.skip_oft or args.no_oracle

    keys_payload = read_json(args.initial_keys.resolve())
    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import load_lerobot_policy_bundle, rollout_seed
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient
    from scripts.generate_oft_pool_candidates import _suite
    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT")); _patch_lerobot_init_states()
    pool = StatePool(Path(str(keys_payload["pool"])).resolve())
    requested = set(args.state_key)
    keys = []
    for key in keys_payload["state_keys"]:
        meta = pool.read_state(str(key), load_observations=False).metadata
        if _suite(meta.task_id) == args.suite and (not requested or str(key) in requested):
            keys.append(str(key))
    if args.max_states:
        keys = keys[:args.max_states]
    if not keys:
        raise ValueError("no matching states")
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device=args.device, num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path, action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    client = None
    if not args.no_oracle:
        client = OracleClient(args.endpoint, timeout_ms=60_000)
        model_info = client.model_info()
        if model_info.get("suite") not in {None, args.suite}:
            raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    salt = int.from_bytes(hashlib.sha256(args.policy_id.encode()).digest()[:4], "big")
    started = time.perf_counter()
    for position, state_key in enumerate(keys, 1):
        # Exact R6-A seed reuse is required for source-trajectory parity.  A
        # new stage-specific salt would silently create a different outcome
        # cohort and invalidate the policy-pair opportunity lock.
        # ``rollout_index`` identifies a repeat of the *same* stochastic
        # rollout.  It must not alter rollout_seed; otherwise the repro audit
        # compares two different random trials rather than exact-repeat
        # determinism.  The replica index is used only in filenames/group IDs.
        seed = rollout_seed(state_key, args.seed_index, 0,
                            salt=salt ^ (0xA16A0000 + args.seed_index))
        result = collect_trajectory(
            pool=pool, state_key=state_key, bundle=bundle, policy_id=args.policy_id,
            rollout_seed_value=seed, seed_index=args.seed_index, client=client,
            boundaries=boundaries, image_size=args.image_size,
            libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
            bookkeeping=args.bookkeeping_mode, skip_oft=skip_oft, debug=args.debug,
            rollout_index=args.rollout_index, temporal_history=args.temporal_history,
            record_oft_trace_hash=args.record_oft_trace_hash,
            record_oft_chunk_trace=args.record_oft_chunk_trace,
        )
        rows = result.pop("rows")
        arrays = {key: result.pop(key) for key in [
            "image", "proprio", "source_action", "oft_action",
            "source_action_summary", "oft_action_summary", "source_action_trace",
            "temporal_image_history", "temporal_proprio_history",
            "temporal_action_history"]}
        obs_debug = result.pop("obs_debug")
        stem = f"{state_key}__seed{args.seed_index}"
        if args.rollout_index:
            stem += f"__rep{args.rollout_index}"
        npz = args.output_dir / f"{stem}.npz"
        np.savez_compressed(npz, **arrays)
        metadata = {"rows": rows, **result, "npz": str(npz.resolve()), "npz_sha256": sha256(npz), "rollout_index": args.rollout_index}
        if obs_debug:
            metadata["observable_debug"] = obs_debug
        (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        summaries.append({key: metadata[key] for key in ["source_success", "source_steps", "stop_reason", "npz", "npz_sha256"]} | {"state_key": state_key, "n_boundaries": len(rows)})
        print(f"R6B1 policy={args.policy_id} seed={args.seed_index} state={position}/{len(keys)} boundaries={len(rows)} success={metadata['source_success']}", flush=True)
    report = {
        "schema_version": "rase-r6b1-dynamic-boundary-pilot/v1",
        "status": "complete", "scientific_scope": "development dynamic-boundary collection",
        "policy_id": args.policy_id, "suite": args.suite, "seed_index": args.seed_index,
        "seed_derivation_version": "rollout_seed(state_key, seed_index, 0, salt=policy_salt ^ (0xA16A0000 + seed_index))",
        "boundaries": boundaries, "n_states": len(summaries),
        "n_boundaries": sum(row["n_boundaries"] for row in summaries),
        "bookkeeping_mode": args.bookkeeping_mode, "skip_oft": bool(skip_oft),
        "temporal_history": int(args.temporal_history),
        "record_oft_trace_hash": bool(args.record_oft_trace_hash),
        "record_oft_chunk_trace": bool(args.record_oft_chunk_trace),
        "initial_keys_sha256": sha256(args.initial_keys.resolve()),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "oracle_model_info": None if client is None else model_info,
        "elapsed_wall_s": time.perf_counter() - started,
        "trajectories": summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

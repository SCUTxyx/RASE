#!/usr/bin/env python3
"""V6 Stage-0 exact-root opportunity pilot on dynamic LIBERO position shifts.

This collector is for the existing position asset, which is produced by moving
free-joint object qpos values at episode start rather than read from a
prebuilt StatePool.  It implements exactly one frozen SmolVLA source:

  The old chunk is generated on the clean observation; its prefix is executed;
  then the position shift is applied at the native-chunk decision boundary.

  C       stale old native-chunk suffix -> fixed seeded mu -> terminal
  R-same  fresh boundary observation + source event seed -> same mu -> terminal
  R-new   fresh boundary observation + K new seeds -> same mu -> terminal

All candidates are frozen before any branch rollout.  R-new values are later
averaged by ``analyze_v6_refresh_opportunity.py``; this script never selects a
best sampled branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "rase-v6-stage0-branch/v2"
ROOT_SCHEMA = "rase-v6-stage0-root/v2"
PLAN_SCHEMA = "rase-v6-stage0-direct-position-plan/v3"


def parse_optional_temperature(value: str | float | None) -> float | None:
    """Parse an explicit sampling-temperature choice without coercing default."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "default"}:
        return None
    result = float(text)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"temperature must be non-negative finite or 'default', got {value!r}")
    return result


def stable_seed(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


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


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def qpos_sha256(single: Any) -> str:
    qpos = np.ascontiguousarray(np.asarray(single._env.sim.data.qpos, dtype=np.float64))
    return hashlib.sha256(qpos.tobytes()).hexdigest()


def bddl_objects_of_interest(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\(:obj_of_interest\s+(.*?)\)", text, re.S)
    if not match:
        return []
    return [token for token in match.group(1).split() if token]


def exact_bddl_objects(bddl_dir: Path, task_name: str) -> tuple[list[str], Path]:
    """Load the BDDL that exactly names the frozen catalog task.

    Long task descriptions have deliberately repetitive natural-language
    prefixes (for example, several begin with ``put both the``).  Selecting a
    BDDL by a text prefix can therefore perturb objects belonging to a
    different task while the simulator executes the requested one.  The
    task-name catalog is the authority here: require its exact file and retain
    the resolved path in each root record for audit.
    """
    path = bddl_dir / f"{task_name}.bddl"
    if not path.is_file():
        raise FileNotFoundError(f"missing exact BDDL for catalog task {task_name!r}: {path}")
    return bddl_objects_of_interest(path), path


def _bddl_tokens(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", text)


def _parse_bddl_node(tokens: list[str], start: int) -> tuple[Any, int]:
    if start >= len(tokens):
        raise ValueError("unexpected end of BDDL")
    if tokens[start] != "(":
        return tokens[start], start + 1
    node: list[Any] = []
    index = start + 1
    while index < len(tokens) and tokens[index] != ")":
        child, index = _parse_bddl_node(tokens, index)
        node.append(child)
    if index >= len(tokens):
        raise ValueError("unclosed BDDL expression")
    return node, index + 1


def _bddl_section(text: str, section: str) -> Any:
    tokens = _bddl_tokens(text)
    wanted = f":{section}"
    for index, token in enumerate(tokens):
        if token == wanted:
            if index == 0 or tokens[index - 1] != "(":
                raise ValueError(f"malformed {wanted} section")
            node, _end = _parse_bddl_node(tokens, index - 1)
            if not isinstance(node, list) or not node or node[0] != wanted:
                raise ValueError(f"malformed {wanted} section")
            content = node[1:]
            if not content:
                raise ValueError(f"empty {wanted} section")
            return content[0] if len(content) == 1 else content
    raise ValueError(f"BDDL has no {wanted} section")


def bddl_goal_manipulated_objects(path: Path) -> list[str]:
    """Return goal-subject objects, excluding passive containers/fixtures.

    ``:obj_of_interest`` commonly contains both a manipulated object and its
    destination (for example a bowl and a basket).  Position perturbation is
    intended to move the object whose state the action chunk must correct, not
    to move the receptacle/fixture as well.  We derive that set from the first
    argument of goal predicates, restricted to objects declared in
    ``:objects``.  The parser retains multiple subjects for multi-object Long
    tasks and ignores fixture-only predicates such as opening a cabinet.
    """
    text = path.read_text(encoding="utf-8")
    objects_node = _bddl_section(text, "objects")
    goal_node = _bddl_section(text, "goal")
    if not isinstance(objects_node, list) or not isinstance(goal_node, list):
        raise ValueError(f"invalid objects/goal section in {path}")
    declared: set[str] = set()
    pending_names: list[str] = []
    skip_type = False
    for token in objects_node:
        if token == "-":
            declared.update(pending_names)
            pending_names = []
            skip_type = True
        elif skip_type:
            # BDDL's object declaration is ``name ... - type``.
            skip_type = False
        elif isinstance(token, str):
            pending_names.append(token)
    subjects: list[str] = []

    def visit(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = str(node[0]).lower()
        if head in {"and", "or", "not"}:
            for child in node[1:]:
                visit(child)
            return
        if len(node) >= 2 and isinstance(node[1], str) and node[1] in declared:
            subjects.append(node[1])

    visit(goal_node)
    result = list(dict.fromkeys(subjects))
    if not result:
        raise ValueError(f"no declared goal-subject object in {path}")
    return result


def choose_perturbation_targets(goal_subjects: list[str], *, mode: str) -> list[str]:
    """Choose targets deterministically, before observing any rollout outcome.

    ``all_goal_subjects`` reproduces the original stress test.  The
    ``first_goal_subject`` condition is a deliberately local disturbance: it
    shifts one BDDL-declared object rather than jointly moving every object in
    a multi-object task.  Neither condition consults a success label.
    """
    if not goal_subjects:
        raise ValueError("goal_subjects is empty")
    if mode == "all_goal_subjects":
        return list(goal_subjects)
    if mode == "first_goal_subject":
        return [goal_subjects[0]]
    raise ValueError(f"unsupported position target mode: {mode!r}")


def clean_bddl_directory(libero_clean_root: Path, suite: str) -> Path:
    """Return the BDDL directory belonging to the environment being rolled out.

    ``make_libero_env_for_task(..., libero_flavor='clean')`` constructs the
    execution environment from ``libero_clean_root``.  Its task objects and
    this collector's perturbation targets must come from the *same* asset
    tree.  A similarly named LIBERO-PRO tree is not interchangeable: its
    BDDL contents can diverge even when filenames agree.
    """
    path = libero_clean_root / "libero" / "libero" / "bddl_files" / suite
    if not path.is_dir():
        raise FileNotFoundError(f"missing clean execution BDDL directory: {path}")
    return path


_STYLE_TOKENS = frozenset({
    "big", "bigger", "small", "smaller", "large", "little",
    "black", "blue", "brown", "green", "orange", "red",
    "white", "yellow", "pink", "purple", "gray", "grey",
})


def canonical_object_name(value: str) -> str:
    """Map BDDL asset names and simulator body names to the same key.

    LIBERO-PRO BDDL names such as ``bigger_alphabet_soup_1`` do not exactly
    equal simulator bodies such as ``alphabet_soup_1_main``.  Exact string
    matching therefore creates a dangerous no-op perturbation.  This mapping
    only removes documented style/name suffixes; object identity and index
    stay intact.
    """
    tokens = [token for token in str(value).lower().split("_") if token]
    tokens = [token for token in tokens if token not in _STYLE_TOKENS and token != "main"]
    return "_".join(tokens)


def target_matches_body(target: str, body: str) -> bool:
    """Match the immutable object-id tail while retaining its instance index.

    Some simulator assets prepend a manufacturer token (for example
    ``akita_black_bowl_1_main``) that is absent from BDDL's
    ``black_bowl_1``.  After canonicalization the latter is a suffix of the
    former.  Matching only a complete indexed suffix is intentionally stricter
    than matching a generic noun such as ``bowl``.
    """
    target_key = canonical_object_name(target)
    body_key = canonical_object_name(body)
    return (
        target_key == body_key
        or body_key.endswith(f"_{target_key}")
        or target_key.endswith(f"_{body_key}")
    )


def apply_position_perturbation(
    single: Any, *, level: float, seed: int, targets: list[str],
) -> list[dict[str, float | str]]:
    """Apply one exact-radial xy shift to every target free joint and forward."""
    sim = single._env.sim
    rng = np.random.default_rng(seed)
    joint_names = {str(name) for name in sim.model.joint_names}
    moved: list[dict[str, float | str]] = []
    for raw_body in sim.model.body_names:
        body_name = str(raw_body)
        base_name = body_name[:-len("_main")] if body_name.endswith("_main") else body_name
        if not any(target_matches_body(target, base_name) for target in targets):
            continue
        joint_name = f"{base_name}_joint0"
        if joint_name not in joint_names:
            continue
        joint_id = sim.model.joint_name2id(joint_name)
        if int(sim.model.jnt_type[joint_id]) != 0:  # MuJoCo free joint
            continue
        address = int(sim.model.jnt_qposadr[joint_id])
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        before_xy = sim.data.qpos[address : address + 2].copy()
        sim.data.qpos[address] += float(level) * np.cos(angle)
        sim.data.qpos[address + 1] += float(level) * np.sin(angle)
        after_xy = sim.data.qpos[address : address + 2].copy()
        delta = after_xy - before_xy
        moved.append({
            "body": base_name,
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "xy_displacement": float(np.linalg.norm(delta)),
        })
    sim.forward()
    return moved


def force_fresh_inference(
    bundle: Mapping[str, Any], observation: Mapping[str, Any], *, task: str,
    boundary_step: int, generation_seed: int, horizon: int, temperature: float | None,
) -> Any:
    """Capture one native chunk under an isolated queue/RNG state."""
    from rase.collect.candidates import seed_everything
    from rase.collect.policy_step import (
        capture_inference_event,
        clear_policy_queues,
        policy_state_restore,
        policy_state_snapshot,
    )

    snapshot = policy_state_snapshot(bundle)
    try:
        bundle["policy"].reset()
        clear_policy_queues(bundle["policy"])
        seed_everything(int(generation_seed))
        _first, event = capture_inference_event(
            bundle, observation, task=task, boundary_step=boundary_step,
            generation_seed=int(generation_seed), horizon=horizon,
            temperature=temperature,
        )
        return event
    finally:
        policy_state_restore(bundle, snapshot)


def terminal_values(term: Any, trunc: Any, info: Any) -> tuple[bool, bool]:
    from rase.collect.policy_step import success_from_info

    terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    return terminal, bool(success_from_info(info)) if terminal else False


def flatten_numeric_observation(value: Any, *, prefix: str = "obs") -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    def visit(item: Any, name: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], f"{name}__{str(key).replace('.', '_')}")
            return
        if item is None or isinstance(item, (str, bytes, bytearray)):
            return
        try:
            array = np.asarray(item)
        except Exception:
            return
        if not array.dtype.hasobject and array.dtype.kind in "biuf":
            output[name] = array.copy()

    visit(value, prefix)
    return output


def candidate_prefix(event: Any, length: int, label: str) -> np.ndarray:
    chunk = np.asarray(event.env_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or chunk.shape[0] < length:
        raise ValueError(f"{label} has invalid native env chunk shape {chunk.shape}; need >=({length},7)")
    return chunk[:length].copy()


def make_root_plan(args: argparse.Namespace) -> dict[str, Any]:
    cursors = [int(item) for item in args.cursors.split(",") if item.strip()]
    if len(cursors) != 3 or len(set(cursors)) != 3:
        raise ValueError("the preregistered pilot needs three distinct cursor strata")
    if any(cursor <= 0 or cursor >= args.native_horizon for cursor in cursors):
        raise ValueError("every cursor must satisfy 0 < cursor < native horizon")
    tasks = [int(item) for item in args.tasks.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks is empty")
    pre_decision_chunks = [
        int(item) for item in str(getattr(args, "pre_decision_chunks", "0")).split(",")
        if item.strip()
    ]
    if not pre_decision_chunks or any(value < 0 for value in pre_decision_chunks):
        raise ValueError("--pre-decision-chunks must be a non-empty list of non-negative integers")
    if args.replicates_per_cell < 1:
        raise ValueError("--replicates-per-cell must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(args.plan_label)):
        raise ValueError("--plan-label must contain only letters, numbers, '_' or '-'")
    seed_namespace = str(getattr(args, "seed_namespace", None) or args.plan_label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", seed_namespace):
        raise ValueError("--seed-namespace must contain only letters, numbers, '_' or '-'")
    roots: list[dict[str, Any]] = []
    for stratum, cursor in enumerate(cursors):
        for task_offset, task_number in enumerate(tasks):
            for replicate in range(args.replicates_per_cell):
                root_index = len(roots)
                pre_chunks = pre_decision_chunks[
                    (stratum + task_offset + replicate) % len(pre_decision_chunks)
                ]
                root_id = (
                    f"{args.plan_label}_s{stratum}_t{task_number:02d}"
                    f"_r{replicate:02d}"
                    + (f"_pc{pre_chunks}" if any(pre_decision_chunks) else "")
                )
                # Keep the rollout identity explicit: output labels may change
                # between a severity sweep, while a shared seed namespace makes
                # every stochastic factor and perturbation direction matched.
                seed_key = (
                    f"{seed_namespace}_s{stratum}_t{task_number:02d}"
                    f"_r{replicate:02d}"
                    + (f"_pc{pre_chunks}" if any(pre_decision_chunks) else "")
                )
                roots.append({
                    "root_id": root_id,
                    "task_id": f"{args.suite}_{task_number:06d}",
                    "suite": args.suite,
                    # Offset replica state IDs by all cursor strata so the
                    # 12 formal roots for a task explore the available
                    # LIBERO initial-state pool before any wraparound.
                    "init_state_id": int(
                        (task_offset + stratum + replicate * len(cursors)) % args.init_states
                    ),
                    "replicate": replicate,
                    "plan_label": args.plan_label,
                    "seed_namespace": seed_namespace,
                    "seed_key": seed_key,
                    "environment_seed": stable_seed("v6-stage0-env", args.seed, seed_key),
                    "perturbation_seed": stable_seed("v6-stage0-pos", args.seed, seed_key),
                    "source_generation_seed": stable_seed("v6-stage0-source", args.seed, seed_key),
                    "pre_decision_chunks": pre_chunks,
                    "pre_decision_generation_seeds": [
                        stable_seed("v6-stage0-predecision", args.seed, seed_key, chunk_index)
                        for chunk_index in range(pre_chunks)
                    ],
                    "downstream_seed": stable_seed("v6-stage0-mu", args.seed, seed_key),
                    "r_new_generation_seeds": [
                        stable_seed("v6-stage0-r-new", args.seed, seed_key, index)
                        for index in range(args.r_new_k)
                    ],
                    "cursor": cursor,
                    "native_chunk_horizon": args.native_horizon,
                    "requested_fraction": float(cursor) / args.native_horizon,
                    "perturb_dim": "position",
                    "perturb_level": float(args.perturb_level),
                    "position_target_mode": args.position_target_mode,
                    "selection_outcomes_used": False,
                    "selection_index": root_index,
                })
    return {
        "schema_version": PLAN_SCHEMA,
        "selection_outcomes_used": False,
        "selection": {
            "suite": args.suite,
            "tasks": tasks,
            "cursors": cursors,
            "native_chunk_horizon": args.native_horizon,
            "r_new_k": args.r_new_k,
            "replicates_per_cell": args.replicates_per_cell,
            "plan_label": args.plan_label,
            "seed_namespace": seed_namespace,
            "position_perturbation_level": args.perturb_level,
            "position_target_mode": args.position_target_mode,
            "pre_decision_chunks": pre_decision_chunks,
            "seed": args.seed,
            "design": (
                "all requested tasks exactly once per cursor; source chunk phase is "
                "assigned cyclically before rollout; no outcome-based filtering"
            ),
        },
        "roots": roots,
    }


def run_branch(
    *, handle: Any, forkable: Any, boundary_snapshot: Any, expected_qpos_sha: str,
    task: str, candidate: np.ndarray, bundle: Mapping[str, Any],
    downstream_seed: int, downstream_temperature: float,
) -> dict[str, Any]:
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    forkable.restore(boundary_snapshot, check_task_fingerprint=True)
    single = handle.vector_env.envs[0]
    if qpos_sha256(single) != expected_qpos_sha:
        raise AssertionError("branch restore qpos checksum drift")
    observation = observation_from_libero_env(single)
    horizon = int(getattr(single, "_max_episode_steps", 600))
    candidate_steps = 0
    continuation_steps = 0
    success = False
    stop = "horizon"
    started = time.perf_counter()
    for action in candidate:
        if current_timestep(handle.control_env) >= horizon:
            return {
                "success": False, "candidate_steps": candidate_steps,
                "continuation_steps": continuation_steps, "stop_reason": "horizon",
                "elapsed_s": round(time.perf_counter() - started, 6),
            }
        observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        candidate_steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            return {
                "success": success, "candidate_steps": candidate_steps,
                "continuation_steps": continuation_steps,
                "stop_reason": "success" if success else "terminal_failure",
                "elapsed_s": round(time.perf_counter() - started, 6),
            }
    mu = InProcessSmolVLAContinuation(
        bundle, temperature=downstream_temperature, seed=downstream_seed,
    )
    mu.reset()
    while current_timestep(handle.control_env) < horizon:
        action = mu.act(observation, task=task)
        observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        continuation_steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            stop = "success" if success else "terminal_failure"
            break
    return {
        "success": bool(success), "candidate_steps": candidate_steps,
        "continuation_steps": continuation_steps, "stop_reason": stop,
        "elapsed_s": round(time.perf_counter() - started, 6),
    }


def branch_row(
    root: Mapping[str, Any], *, arm: str, arm_index: int | None,
    snapshot_sha: str, candidate: np.ndarray, candidate_seed: int | None,
    full_sha: str, result: Mapping[str, Any], source_temperature: float | None,
    downstream_temperature: float | None, causal_metadata: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "root_id": root["root_id"], "state_key": root["root_id"],
        "task_id": root["task_id"], "suite": root["suite"],
        "perturb_dim": root["perturb_dim"], "perturb_level": root["perturb_level"],
        "cursor": root["cursor"], "native_chunk_horizon": root["native_chunk_horizon"],
        "pre_decision_chunks": int(root.get("pre_decision_chunks", 0)),
        "actual_cursor_fraction": root["requested_fraction"],
        "root_snapshot_sha256": snapshot_sha,
        "old_chunk_source_qpos_sha256": causal_metadata["old_chunk_source_qpos_sha256"],
        "pre_perturb_boundary_qpos_sha256": causal_metadata["pre_perturb_boundary_qpos_sha256"],
        "perturbation_timing": causal_metadata["perturbation_timing"],
        "position_target_mode": root["position_target_mode"],
        "source_generation_seed": root["source_generation_seed"],
        "source_temperature": source_temperature,
        "arm": arm, "arm_index": arm_index,
        "candidate_generation_seed": candidate_seed,
        "candidate_chunk_steps": int(candidate.shape[0]),
        "candidate_chunk_sha256": array_sha256(candidate),
        "candidate_full_chunk_sha256": full_sha,
        "downstream_controller": "same_source_fixed_mu",
        "downstream_seed": root["downstream_seed"],
        "downstream_temperature": downstream_temperature,
        "success": bool(result["success"]),
        "rollout": dict(result),
    }


def collect_root(
    root: Mapping[str, Any], *, bundle: Mapping[str, Any], args: argparse.Namespace,
    bddl_dir: Path, artifact_dir: Path,
) -> dict[str, Any]:
    from rase.backends.libero_clean import clean_task_name
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.envs.forkable_env import ForkableEnv

    handle = make_libero_env_for_task(
        str(root["task_id"]), init_state_id=int(root["init_state_id"]),
        seed=int(root["environment_seed"]), observation_height=args.observation_height,
        observation_width=args.observation_width, libero_clean_root=str(args.libero_clean_root),
        libero_flavor="clean",
    )
    try:
        single = handle.vector_env.envs[0]
        task = str(single.task_description)
        task_number = int(str(root["task_id"]).rsplit("_", 1)[-1])
        catalog_task_name = clean_task_name(str(root["suite"]), task_number)
        interest_targets, bddl_path = exact_bddl_objects(bddl_dir, catalog_task_name)
        all_goal_targets = bddl_goal_manipulated_objects(bddl_path)
        targets = choose_perturbation_targets(
            all_goal_targets, mode=str(root["position_target_mode"]),
        )
        forkable = ForkableEnv(handle.control_env)
        # Build a clean, pre-registered source trajectory before freezing the
        # old action chunk.  This moves the decision root into a later policy
        # phase without consulting terminal outcomes.  Each preceding native
        # chunk has its own planned seed, so replays are exact and do not share
        # mutable action queues with C/R candidate generation.
        observation = observation_from_libero_env(single)
        predecision_events: list[dict[str, Any]] = []
        predecision_actions: list[np.ndarray] = []
        pre_chunks = int(root.get("pre_decision_chunks", 0))
        pre_seeds = [int(value) for value in root.get("pre_decision_generation_seeds", [])]
        if len(pre_seeds) != pre_chunks or len(set(pre_seeds)) != pre_chunks:
            raise ValueError("pre-decision source seeds must be unique and match pre_decision_chunks")
        for chunk_index, generation_seed in enumerate(pre_seeds):
            event = force_fresh_inference(
                bundle, observation, task=task,
                boundary_step=chunk_index * int(root["native_chunk_horizon"]),
                generation_seed=generation_seed,
                horizon=int(root["native_chunk_horizon"]), temperature=args.source_temperature,
            )
            chunk = candidate_prefix(
                event, int(root["native_chunk_horizon"]), f"pre-decision chunk {chunk_index}",
            )
            predecision_events.append({
                "chunk_index": chunk_index,
                "generation_seed": generation_seed,
                "event_id": event.inference_event_id,
                "chunk_sha256": array_sha256(chunk),
            })
            for action in chunk:
                observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
                predecision_actions.append(np.asarray(action, dtype=np.float32).copy())
                terminal, success = terminal_values(term, trunc, info)
                if terminal:
                    return {
                        "schema_version": ROOT_SCHEMA, "status": "unavailable", "root": dict(root),
                        "reason": "terminal_before_predecision_phase",
                        "terminal_success": success, "branches": [],
                        "predecision_events": predecision_events,
                        "bddl_source": "clean_execution_assets",
                        "catalog_task_name": catalog_task_name,
                        "bddl_path": str(bddl_path), "targets_from_goal": all_goal_targets,
                        "perturbation_targets": targets,
                        "position_target_mode": root["position_target_mode"],
                        "targets_from_obj_of_interest": interest_targets,
                    }
        old_source_observation = observation
        old_chunk_source_qpos_sha = qpos_sha256(single)
        old_event = force_fresh_inference(
            bundle, old_source_observation, task=task,
            boundary_step=pre_chunks * int(root["native_chunk_horizon"]),
            generation_seed=int(root["source_generation_seed"]),
            horizon=int(root["native_chunk_horizon"]), temperature=args.source_temperature,
        )
        old_chunk = candidate_prefix(old_event, int(root["native_chunk_horizon"]), "source old chunk")
        cursor = int(root["cursor"])
        observation = old_source_observation
        for action in old_chunk[:cursor]:
            observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            terminal, success = terminal_values(term, trunc, info)
            if terminal:
                return {
                    "schema_version": ROOT_SCHEMA, "status": "unavailable", "root": dict(root),
                    "reason": "terminal_before_decision", "terminal_success": success,
                    "branches": [], "bddl_source": "clean_execution_assets",
                    "catalog_task_name": catalog_task_name,
                    "bddl_path": str(bddl_path), "targets_from_goal": all_goal_targets,
                    "perturbation_targets": targets,
                    "position_target_mode": root["position_target_mode"],
                    "targets_from_obj_of_interest": interest_targets,
                    "old_chunk_source_qpos_sha256": old_chunk_source_qpos_sha,
                }
        pre_perturb_boundary_qpos_sha = qpos_sha256(single)
        moved = apply_position_perturbation(
            single, level=float(root["perturb_level"]),
            seed=int(root["perturbation_seed"]), targets=targets,
        )
        if not moved:
            return {
                "schema_version": ROOT_SCHEMA, "status": "unavailable", "root": dict(root),
                "reason": "no_target_free_joint_moved", "branches": [],
                "bddl_source": "clean_execution_assets",
                "catalog_task_name": catalog_task_name, "bddl_path": str(bddl_path),
                "targets_from_goal": all_goal_targets, "perturbation_targets": targets,
                "position_target_mode": root["position_target_mode"],
                "targets_from_obj_of_interest": interest_targets,
                "old_chunk_source_qpos_sha256": old_chunk_source_qpos_sha,
                "pre_perturb_boundary_qpos_sha256": pre_perturb_boundary_qpos_sha,
            }
        boundary_snapshot = forkable.snapshot()
        boundary_qpos_sha = qpos_sha256(single)
        if float(root["perturb_level"]) > 0.0 and boundary_qpos_sha == pre_perturb_boundary_qpos_sha:
            raise AssertionError("position perturbation did not change branch-root qpos")
        boundary_observation = observation_from_libero_env(single)
        continue_chunk = old_chunk[cursor:]
        candidate_horizon = int(continue_chunk.shape[0])
        same_event = force_fresh_inference(
            bundle, boundary_observation, task=task, boundary_step=cursor,
            generation_seed=int(root["source_generation_seed"]),
            horizon=int(root["native_chunk_horizon"]), temperature=args.source_temperature,
        )
        same_chunk = candidate_prefix(same_event, candidate_horizon, "R-same")
        new_events: list[tuple[int, Any, np.ndarray]] = []
        for index, generation_seed in enumerate(root["r_new_generation_seeds"]):
            event = force_fresh_inference(
                bundle, boundary_observation, task=task, boundary_step=cursor,
                generation_seed=int(generation_seed), horizon=int(root["native_chunk_horizon"]),
                temperature=args.source_temperature,
            )
            new_events.append((index, event, candidate_prefix(event, candidate_horizon, f"R-new[{index}]")))
        if len({int(seed) for seed in root["r_new_generation_seeds"]}) != args.r_new_k:
            raise ValueError("R-new seeds must be unique")
        if int(root["source_generation_seed"]) in set(root["r_new_generation_seeds"]):
            raise ValueError("R-new must not reuse source generation seed")

        def evaluate(chunk: np.ndarray) -> dict[str, Any]:
            return run_branch(
                handle=handle, forkable=forkable, boundary_snapshot=boundary_snapshot,
                expected_qpos_sha=boundary_qpos_sha, task=task, candidate=chunk,
                bundle=bundle, downstream_seed=int(root["downstream_seed"]),
                downstream_temperature=args.downstream_temperature,
            )

        causal_metadata = {
            "old_chunk_source_qpos_sha256": old_chunk_source_qpos_sha,
            "pre_perturb_boundary_qpos_sha256": pre_perturb_boundary_qpos_sha,
            "perturbation_timing": "after_old_prefix_before_branch_decision",
        }
        branches: list[dict[str, Any]] = []
        branches.append(branch_row(
            root, arm="C", arm_index=None, snapshot_sha=boundary_qpos_sha,
            candidate=continue_chunk, candidate_seed=None, full_sha=array_sha256(old_chunk),
            result=evaluate(continue_chunk), source_temperature=args.source_temperature,
            downstream_temperature=args.downstream_temperature,
            causal_metadata=causal_metadata,
        ))
        branches.append(branch_row(
            root, arm="R_same", arm_index=None, snapshot_sha=boundary_qpos_sha,
            candidate=same_chunk, candidate_seed=int(root["source_generation_seed"]),
            full_sha=array_sha256(same_event.env_chunk), result=evaluate(same_chunk),
            source_temperature=args.source_temperature, downstream_temperature=args.downstream_temperature,
            causal_metadata=causal_metadata,
        ))
        for index, event, chunk in new_events:
            branches.append(branch_row(
                root, arm="R_new", arm_index=index, snapshot_sha=boundary_qpos_sha,
                candidate=chunk, candidate_seed=int(event.candidate_generation_seed),
                full_sha=array_sha256(event.env_chunk), result=evaluate(chunk),
                source_temperature=args.source_temperature, downstream_temperature=args.downstream_temperature,
                causal_metadata=causal_metadata,
            ))
        if len({row["root_snapshot_sha256"] for row in branches}) != 1:
            raise AssertionError("same-root snapshot drift")
        if len({row["downstream_seed"] for row in branches}) != 1:
            raise AssertionError("fixed downstream seed drift")
        artifact_arrays = {
            "old_chunk": old_chunk, "continue_suffix": continue_chunk,
            "refresh_same": same_chunk,
            "refresh_new": np.stack([chunk for _index, _event, chunk in new_events]),
        }
        if predecision_actions:
            artifact_arrays["predecision_actions"] = np.stack(predecision_actions)
        artifact_arrays.update(flatten_numeric_observation(boundary_observation))
        artifact_path = artifact_dir / f"{root['root_id']}.npz"
        atomic_npz(artifact_path, artifact_arrays)
        return {
            "schema_version": ROOT_SCHEMA, "status": "complete", "root": dict(root),
            "bddl_source": "clean_execution_assets",
            "catalog_task_name": catalog_task_name, "bddl_path": str(bddl_path),
            "targets_from_goal": all_goal_targets,
            "perturbation_targets": targets,
            "position_target_mode": root["position_target_mode"],
            "targets_from_obj_of_interest": interest_targets, "moved_objects": moved,
            "predecision_events": predecision_events,
            "boundary": {
                "qpos_sha256": boundary_qpos_sha,
                "pre_perturb_qpos_sha256": pre_perturb_boundary_qpos_sha,
                "old_chunk_source_qpos_sha256": old_chunk_source_qpos_sha,
                "perturbation_timing": "after_old_prefix_before_branch_decision",
                "old_chunk_sha256": array_sha256(old_chunk),
                "source_event_id": old_event.inference_event_id,
                "source_event_seed": old_event.candidate_generation_seed,
                "candidate_horizon": candidate_horizon,
                "pre_decision_chunks": pre_chunks,
                "artifact": str(artifact_path),
            },
            "branches": branches,
        }
    finally:
        handle.close()


def merge_jsonl(root_dir: Path, output: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(root_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(value.get("branches") or [])
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--init-states", type=int, default=10)
    parser.add_argument("--cursors", default="3,5,8")
    parser.add_argument(
        "--pre-decision-chunks", default="0",
        help=(
            "Comma-separated source chunk indices to execute before the decision chunk. "
            "Use e.g. 1,2,3 for a later-phase pilot; default 0 preserves the initial-phase protocol."
        ),
    )
    parser.add_argument("--native-horizon", type=int, default=10)
    parser.add_argument("--r-new-k", type=int, default=4)
    parser.add_argument("--replicates-per-cell", type=int, default=1)
    parser.add_argument("--plan-label", default="stage0")
    parser.add_argument(
        "--seed-namespace", default=None,
        help=(
            "Optional seed identity independent of --plan-label. Reuse it across "
            "perturbation severities to pair environment state, sampling noise, and shift direction."
        ),
    )
    parser.add_argument("--perturb-level", type=float, default=0.2)
    parser.add_argument(
        "--position-target-mode", choices=("all_goal_subjects", "first_goal_subject"),
        default="all_goal_subjects",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--source-temperature", type=parse_optional_temperature, default=0.5)
    parser.add_argument("--downstream-temperature", type=parse_optional_temperature, default=0.5)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--libero-clean-root", type=Path, default=Path("/root/autodl-tmp/src/LIBERO"))
    parser.add_argument("--libero-pro-root", type=Path, default=Path("/root/autodl-tmp/libero_pro_root_object"))
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    if args.r_new_k != 4:
        parser.error("V6 Stage-0 pilot preregisters --r-new-k 4")
    if args.init_states < 1:
        parser.error("--init-states must be positive")
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        parser.error(f"--fresh-run refuses existing output {output}")
    output.mkdir(parents=True, exist_ok=True)
    roots_dir = output / "roots"
    artifacts_dir = output / "decision_artifacts"
    roots_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    plan = make_root_plan(args)
    plan_path = output / "root_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("existing root plan differs; choose a new output directory")
    else:
        atomic_json(plan_path, plan)

    from libero.libero.utils import get_libero_path, set_libero_path
    from rase.collect.forked_rollout import load_lerobot_policy_bundle

    previous_bddl = get_libero_path("bddl_files")
    bddl_dir = clean_bddl_directory(args.libero_clean_root, str(args.suite))
    if not args.policy.is_absolute():
        args.policy = ROOT / args.policy
    if not args.tokenizer.is_absolute():
        args.tokenizer = ROOT / args.tokenizer
    bundle = load_lerobot_policy_bundle(
        args.policy, device="cuda", num_steps=args.native_horizon,
        n_action_steps=args.native_horizon, tokenizer_path=args.tokenizer,
        observation_height=args.observation_height, observation_width=args.observation_width,
    )
    run_manifest = {
        "schema_version": "rase-v6-stage0-direct-position-run/v2",
        "root_plan": str(plan_path), "selection_outcomes_used": False,
        "source_policy": "frozen SmolVLA only",
        "bddl_source": "clean_execution_assets",
        "bddl_dir": str(bddl_dir),
        "position_target_definition": "deterministic BDDL goal-subject subset selected before rollout",
        "candidate_protocol": "execute any preregistered clean source chunks; freeze old chunk; execute old prefix; perturb decision boundary; C stale suffix; R-same matched seed; R-new four distinct seeds; candidates frozen before rollout",
        "downstream_protocol": "same source VLA fixed per-root seed/temperature across every branch",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    started = time.perf_counter()
    errors = 0
    try:
        for index, root in enumerate(plan["roots"]):
            target = roots_dir / f"{root['root_id']}.json"
            if target.exists():
                print(f"V6 Stage0 skip {index + 1}/{len(plan['roots'])} {root['root_id']}", flush=True)
                continue
            try:
                record = collect_root(root, bundle=bundle, args=args, bddl_dir=bddl_dir, artifact_dir=artifacts_dir)
            except Exception as exc:
                errors += 1
                record = {
                    "schema_version": ROOT_SCHEMA, "status": "error", "root": root,
                    "error_type": type(exc).__name__, "error": str(exc)[:2000], "branches": [],
                }
            atomic_json(target, record)
            elapsed_min = (time.perf_counter() - started) / 60.0
            print(f"V6 Stage0 {index + 1}/{len(plan['roots'])} {root['root_id']} status={record['status']} elapsed_min={elapsed_min:.1f}", flush=True)
    finally:
        set_libero_path(previous_bddl)
    n_rows = merge_jsonl(roots_dir, output / "stage0_records.jsonl")
    summary = {
        "n_planned_roots": len(plan["roots"]), "n_branch_rows": n_rows,
        "uncaught_root_errors": errors, "records": str(output / "stage0_records.jsonl"),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if n_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

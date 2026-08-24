#!/usr/bin/env python3
"""Append a preregistered K=8 refinement to mixed-outcome V6 Stage-0 roots.

The original Stage-0 collection remains immutable.  This script selects every
root whose original K=4 R-new outcomes contain both a success and a failure,
reconstructs its *exact* branch root, verifies its qpos checksum, and appends
four disjoint R-new samples (indices 4--7).  Selection is symmetric in the
sign of the observed advantage: only Monte-Carlo ambiguity, never outcome
direction, determines whether a root receives K=8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_v6_stage0_direct_position import (
    ROOT_SCHEMA,
    array_sha256,
    atomic_json,
    atomic_npz,
    bddl_goal_manipulated_objects,
    branch_row,
    candidate_prefix,
    choose_perturbation_targets,
    clean_bddl_directory,
    force_fresh_inference,
    parse_optional_temperature,
    qpos_sha256,
    run_branch,
    stable_seed,
    terminal_values,
)


SCHEMA = "rase-v6-stage0-k8-refinement/v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def root_branch_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = record.get("branches") or []
    if not isinstance(rows, list):
        raise ValueError("root branches must be a list")
    typed = [row for row in rows if isinstance(row, dict)]
    if len(typed) != len(rows):
        raise ValueError("root branches must contain only objects")
    return typed


def select_mixed_roots(source_dir: Path) -> list[dict[str, Any]]:
    """Select every and only K=4 root with mixed raw R-new outcomes."""
    selected: list[dict[str, Any]] = []
    for path in sorted((source_dir / "roots").glob("*.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            continue
        rows = root_branch_rows(record)
        r_new = sorted(
            (row for row in rows if row.get("arm") == "R_new"),
            key=lambda row: int(row.get("arm_index", -1)),
        )
        if len(r_new) != 4:
            raise ValueError(f"{path.name}: expected original K=4, got {len(r_new)}")
        outcomes = [bool(row.get("success")) for row in r_new]
        if any(outcomes) and not all(outcomes):
            selected.append({
                "path": str(path),
                "root": dict(record["root"]),
                "record": record,
                "original_r_new_seeds": [int(row["candidate_generation_seed"]) for row in r_new],
                "original_r_new_outcomes": outcomes,
            })
    return selected


def refinement_seeds(root: Mapping[str, Any], original: list[int]) -> list[int]:
    source_seed = int(root["source_generation_seed"])
    seeds = [stable_seed("v6-stage0-k8-refinement", root["root_id"], index) for index in range(4, 8)]
    if len(set(seeds)) != 4:
        raise AssertionError("refinement seed collision")
    if source_seed in seeds or set(original) & set(seeds):
        raise AssertionError("refinement reused an original generation seed")
    return seeds


def _exact_boundary(
    root: Mapping[str, Any], *, bundle: Mapping[str, Any], args: argparse.Namespace,
    bddl_dir: Path, expected_targets: list[str],
) -> tuple[Any, Any, Any, str, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Recreate the old-prefix/perturbation root from the original protocol."""
    from rase.backends.libero_clean import clean_task_name
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.envs.forkable_env import ForkableEnv
    from collect_v6_stage0_direct_position import apply_position_perturbation, exact_bddl_objects

    handle = make_libero_env_for_task(
        str(root["task_id"]), init_state_id=int(root["init_state_id"]),
        seed=int(root["environment_seed"]), observation_height=args.observation_height,
        observation_width=args.observation_width, libero_clean_root=str(args.libero_clean_root),
        libero_flavor="clean",
    )
    single = handle.vector_env.envs[0]
    task = str(single.task_description)
    task_number = int(str(root["task_id"]).rsplit("_", 1)[-1])
    catalog_task_name = clean_task_name(str(root["suite"]), task_number)
    _interest, bddl_path = exact_bddl_objects(bddl_dir, catalog_task_name)
    all_goal_targets = bddl_goal_manipulated_objects(bddl_path)
    targets = choose_perturbation_targets(all_goal_targets, mode=str(root["position_target_mode"]))
    if targets != expected_targets:
        handle.close()
        raise AssertionError("target selection drifted from the original root record")
    forkable = ForkableEnv(handle.control_env)
    old_observation = observation_from_libero_env(single)
    old_qpos = qpos_sha256(single)
    old_event = force_fresh_inference(
        bundle, old_observation, task=task, boundary_step=0,
        generation_seed=int(root["source_generation_seed"]),
        horizon=int(root["native_chunk_horizon"]), temperature=args.source_temperature,
    )
    old_chunk = candidate_prefix(old_event, int(root["native_chunk_horizon"]), "source old chunk")
    observation = old_observation
    for action in old_chunk[:int(root["cursor"])]:
        observation, _, terminated, truncated, info = handle.vector_env.step(as_batched_action(action))
        terminal, _success = terminal_values(terminated, truncated, info)
        if terminal:
            handle.close()
            raise AssertionError("originally complete root became terminal before K8 boundary")
    pre_perturb = qpos_sha256(single)
    moved = apply_position_perturbation(
        single, level=float(root["perturb_level"]), seed=int(root["perturbation_seed"]), targets=targets,
    )
    if not moved:
        handle.close()
        raise AssertionError("original perturbation no longer moved a target")
    snapshot = forkable.snapshot()
    root_hash = qpos_sha256(single)
    boundary_observation = observation_from_libero_env(single)
    suffix = old_chunk[int(root["cursor"]) :]
    if suffix.ndim != 2 or suffix.shape[0] < 1:
        handle.close()
        raise AssertionError("invalid stale suffix")
    metadata = {
        "old_chunk_source_qpos_sha256": old_qpos,
        "pre_perturb_boundary_qpos_sha256": pre_perturb,
        "perturbation_timing": "after_old_prefix_before_branch_decision",
        "old_chunk_sha256": array_sha256(old_chunk),
        "candidate_horizon": int(suffix.shape[0]),
    }
    return handle, forkable, snapshot, task, boundary_observation, suffix, metadata, moved


def refine_root(
    entry: Mapping[str, Any], *, bundle: Mapping[str, Any], args: argparse.Namespace,
    bddl_dir: Path, artifact_dir: Path,
) -> dict[str, Any]:
    root = dict(entry["root"])
    existing = entry["record"]
    rows = root_branch_rows(existing)
    c_row = next(row for row in rows if row.get("arm") == "C")
    original_seeds = [int(value) for value in entry["original_r_new_seeds"]]
    extra_seeds = refinement_seeds(root, original_seeds)
    handle = None
    try:
        handle, forkable, snapshot, task, observation, suffix, metadata, moved = _exact_boundary(
            root, bundle=bundle, args=args, bddl_dir=bddl_dir,
            expected_targets=list(existing.get("perturbation_targets") or []),
        )
        expected_root_hash = str(c_row["root_snapshot_sha256"])
        actual_root_hash = qpos_sha256(handle.vector_env.envs[0])
        if actual_root_hash != expected_root_hash:
            raise AssertionError(
                f"root snapshot mismatch expected={expected_root_hash} actual={actual_root_hash}"
            )
        if metadata["old_chunk_source_qpos_sha256"] != c_row["old_chunk_source_qpos_sha256"]:
            raise AssertionError("old-source checksum mismatch")
        if metadata["pre_perturb_boundary_qpos_sha256"] != c_row["pre_perturb_boundary_qpos_sha256"]:
            raise AssertionError("pre-perturb checksum mismatch")
        if metadata["old_chunk_sha256"] != str(existing["boundary"]["old_chunk_sha256"]):
            raise AssertionError("old chunk mismatch")
        # Freeze every extra candidate before evaluating any branch.  Otherwise
        # a preceding rollout could alter mutable policy/RNG state and quietly
        # make K=8 a sequential rather than same-root candidate comparison.
        extra_events: list[tuple[int, Any, np.ndarray]] = []
        for index, seed in enumerate(extra_seeds, start=4):
            event = force_fresh_inference(
                bundle, observation, task=task, boundary_step=int(root["cursor"]),
                generation_seed=seed, horizon=int(root["native_chunk_horizon"]),
                temperature=args.source_temperature,
            )
            chunk = candidate_prefix(event, int(suffix.shape[0]), f"R-new refinement[{index}]")
            extra_events.append((index, event, chunk))
        branches: list[dict[str, Any]] = []
        extra_chunks: list[np.ndarray] = []
        for index, event, chunk in extra_events:
            result = run_branch(
                handle=handle, forkable=forkable, boundary_snapshot=snapshot,
                expected_qpos_sha=actual_root_hash, task=task, candidate=chunk,
                bundle=bundle, downstream_seed=int(root["downstream_seed"]),
                downstream_temperature=args.downstream_temperature,
            )
            branch = branch_row(
                root, arm="R_new", arm_index=index, snapshot_sha=actual_root_hash,
                candidate=chunk, candidate_seed=int(event.candidate_generation_seed),
                full_sha=array_sha256(event.env_chunk), result=result,
                source_temperature=args.source_temperature,
                downstream_temperature=args.downstream_temperature,
                causal_metadata={
                    "old_chunk_source_qpos_sha256": metadata["old_chunk_source_qpos_sha256"],
                    "pre_perturb_boundary_qpos_sha256": metadata["pre_perturb_boundary_qpos_sha256"],
                    "perturbation_timing": metadata["perturbation_timing"],
                },
            )
            branch.update({"refinement_round": 1, "refinement_protocol": "mixed_k4_to_k8"})
            branches.append(branch)
            extra_chunks.append(chunk)
        if len({row["root_snapshot_sha256"] for row in branches}) != 1:
            raise AssertionError("refinement branch snapshot drift")
        artifact_path = artifact_dir / f"{root['root_id']}.npz"
        atomic_npz(artifact_path, {"refresh_new_extra": np.stack(extra_chunks)})
        return {
            "schema_version": SCHEMA,
            "status": "complete",
            "root_id": root["root_id"],
            "root": root,
            "selection_reason": "all_and_only_mixed_original_k4_r_new_outcomes",
            "selection_outcomes_used": "only to determine Monte-Carlo ambiguity, not advantage direction",
            "original_r_new_outcomes": list(entry["original_r_new_outcomes"]),
            "original_r_new_seeds": original_seeds,
            "extra_r_new_seeds": extra_seeds,
            "expected_root_snapshot_sha256": expected_root_hash,
            "actual_root_snapshot_sha256": actual_root_hash,
            "moved_objects": moved,
            "artifact": str(artifact_path),
            "branches": branches,
        }
    finally:
        if handle is not None:
            handle.close()


def merge_records(source_dir: Path, refinement_dir: Path, output: Path) -> int:
    rows: list[dict[str, Any]] = []
    source = source_dir / "stage0_records.jsonl"
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    for path in sorted((refinement_dir / "roots").glob("*.json")):
        record = read_json(path)
        if record.get("status") == "complete":
            rows.extend(root_branch_rows(record))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("ckpts/smolvla_libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path("ckpts/SmolVLM2-500M-Instruct"))
    parser.add_argument("--libero-clean-root", type=Path, default=Path("/root/autodl-tmp/src/LIBERO"))
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument(
        "--max-roots", type=int, default=None,
        help="Optional resumable smoke limit; selection itself remains unchanged.",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output == source_dir:
        parser.error("--output-dir must be distinct from immutable --source-dir")
    source_manifest = read_json(source_dir / "run_manifest.json")
    source_args = source_manifest.get("args") or {}
    if str(source_args.get("suite")) != "libero_10":
        parser.error("K8 refinement currently supports the audited libero_10 direct-position source only")
    args.source_temperature = parse_optional_temperature(source_args.get("source_temperature"))
    args.downstream_temperature = parse_optional_temperature(source_args.get("downstream_temperature"))
    args.native_horizon = int(source_args.get("native_horizon", 10))
    if not args.policy.is_absolute():
        args.policy = ROOT / args.policy
    if not args.tokenizer.is_absolute():
        args.tokenizer = ROOT / args.tokenizer
    selected = select_mixed_roots(source_dir)
    if args.max_roots is not None and args.max_roots < 1:
        parser.error("--max-roots must be positive")
    worklist = selected if args.max_roots is None else selected[: args.max_roots]
    output.mkdir(parents=True, exist_ok=True)
    roots_dir = output / "roots"
    artifacts_dir = output / "decision_artifacts"
    roots_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    atomic_json(output / "selection_manifest.json", {
        "schema_version": SCHEMA,
        "source_dir": str(source_dir),
        "selection_rule": "all_and_only_original_K4_R_new_mixed_outcomes",
        "selected_root_ids": [entry["root"]["root_id"] for entry in selected],
        "n_selected": len(selected),
        "n_processed_this_invocation": len(worklist),
        "selection_outcomes_used": "only K ambiguity; refresh-better and continue-better both retained",
    })
    from libero.libero.utils import get_libero_path, set_libero_path
    from rase.collect.forked_rollout import load_lerobot_policy_bundle

    previous_bddl = get_libero_path("bddl_files")
    bddl_dir = clean_bddl_directory(args.libero_clean_root, "libero_10")
    bundle = load_lerobot_policy_bundle(
        args.policy, device="cuda", num_steps=args.native_horizon,
        n_action_steps=args.native_horizon, tokenizer_path=args.tokenizer,
        observation_height=args.observation_height, observation_width=args.observation_width,
    )
    started = time.perf_counter()
    errors = 0
    try:
        for index, entry in enumerate(worklist, start=1):
            root_id = str(entry["root"]["root_id"])
            target = roots_dir / f"{root_id}.json"
            if target.exists():
                print(f"K8 refinement skip {index}/{len(worklist)} {root_id}", flush=True)
                continue
            try:
                record = refine_root(
                    entry, bundle=bundle, args=args, bddl_dir=bddl_dir, artifact_dir=artifacts_dir,
                )
            except Exception as exc:
                errors += 1
                record = {
                    "schema_version": SCHEMA, "status": "error", "root_id": root_id,
                    "root": entry["root"], "error_type": type(exc).__name__,
                    "error": str(exc)[:2000], "branches": [],
                }
            atomic_json(target, record)
            elapsed_min = (time.perf_counter() - started) / 60.0
            print(f"K8 refinement {index}/{len(worklist)} {root_id} status={record['status']} elapsed_min={elapsed_min:.1f}", flush=True)
    finally:
        set_libero_path(previous_bddl)
    merged = merge_records(source_dir, output, output / "stage0_records_k8.jsonl")
    summary = {
        "schema_version": SCHEMA,
        "source_roots": 120,
        "selected_mixed_roots": len(selected),
        "new_branch_rows": 4 * len(selected),
        "merged_branch_rows": merged,
        "uncaught_refinement_errors": errors,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "records": str(output / "stage0_records_k8.jsonl"),
    }
    atomic_json(output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

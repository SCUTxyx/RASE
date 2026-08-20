#!/usr/bin/env python3
"""Simulator-verified Phase-D0 action perturbation feasibility smoke.

This is an A-PARTIAL/pi0-fast development-only smoke.  It verifies that
predeclared physical perturbations can be executed from the same boundary and
produce nontrivial outcome support.  It is not semantic pretraining or D-GATE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.candidate_capture import array_sha256, file_sha256
from rase.vnext.phase_c_pilot import stable_seed


PERTURBATIONS = (
    "identity",
    "translation_sign_flip",
    "rotation_sign_flip",
    "temporal_reverse",
    "gripper_phase_shift",
)
BASE_OPERATOR = "fallback.persistent"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def transform(actions: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32).copy()
    if name == "identity":
        pass
    elif name == "translation_sign_flip":
        value[:, :3] *= -1.0
    elif name == "rotation_sign_flip":
        value[:, 3:6] *= -1.0
    elif name == "temporal_reverse":
        value = value[::-1].copy()
    elif name == "gripper_phase_shift":
        original = value[:, 6].copy()
        shifted = np.roll(original, 2)
        # A constant open/close chunk has no timing event to shift.  The
        # preregistered fallback is convention inversion, still a gripper-only
        # perturbation and decided from actions rather than outcomes.
        value[:, 6] = -original if np.array_equal(shifted, original) else shifted
    else:
        raise ValueError(name)
    return np.clip(value, -1.0, 1.0).astype(np.float32, copy=False)


def terminal_values(term: Any, trunc: Any, info: Any) -> tuple[bool, bool]:
    from rase.collect.policy_step import success_from_info
    terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
    return terminal, bool(success_from_info(info)) if terminal else False


def execute_candidate(
    restored: Any, snapshot: Any, continuation: Any, *, actions: np.ndarray,
    instruction: str, continuation_seed: int,
) -> dict[str, Any]:
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    restored.forkable.restore(
        snapshot, check_task_fingerprint=restored.check_task_fingerprint,
    )
    vector_env = restored.handle.vector_env
    single = vector_env.envs[0]
    horizon = int(getattr(single, "_max_episode_steps", 600))
    observation = observation_from_libero_env(single)
    steps = 0
    success = False
    stop_reason = "horizon"
    started = time.perf_counter()
    for action in actions:
        if current_timestep(restored.handle.control_env) >= horizon:
            break
        observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            stop_reason = "success" if success else "terminal_failure"
            break
    candidate_steps = steps
    continuation.seed = continuation_seed
    continuation.reset_metrics(); continuation.reset()
    while stop_reason == "horizon" and current_timestep(restored.handle.control_env) < horizon:
        try:
            action = np.asarray(
                continuation.act(observation, task=instruction), dtype=np.float32,
            ).reshape(-1, 7)[0]
        except Exception as exc:
            stop_reason = "policy_inference_error"
            return {
                "success": False, "stop_reason": stop_reason, "steps": steps,
                "candidate_steps": candidate_steps,
                "exception_type": type(exc).__name__, "exception": str(exc)[:1000],
                "wall_s": time.perf_counter() - started,
            }
        observation, _, term, trunc, info = vector_env.step(as_batched_action(action))
        steps += 1
        terminal, success = terminal_values(term, trunc, info)
        if terminal:
            stop_reason = "success" if success else "terminal_failure"
            break
    return {
        "success": bool(success), "stop_reason": stop_reason, "steps": steps,
        "candidate_steps": candidate_steps, "wall_s": time.perf_counter() - started,
        "continuation_metrics": continuation.metrics(),
    }


def freeze_protocol(capture_dir: Path, output_dir: Path) -> dict[str, Any]:
    captures = []
    for path in sorted(capture_dir.glob("*.json")):
        meta = json.loads(path.read_text())
        captures.append({
            "metadata_path": str(path.resolve()), "metadata_sha256": file_sha256(path),
            "arrays_path": meta["arrays_path"], "arrays_sha256": meta["arrays_sha256"],
            "group_key": meta["group_key"], "task_id": meta["task_id"],
            "suite": meta["suite"], "policy_id": meta["policy_id"],
            "decision_point_id": meta["decision_point_id"],
        })
    if len(captures) != 4 or {item["suite"] for item in captures} != {
        "Spatial", "Object", "Goal", "Long"
    }:
        raise SystemExit("D0 requires exactly one capture per suite")
    protocol = {
        "schema_version": "rase-vnext-d0-semantic-feasibility-protocol/v3",
        "status": "FROZEN_BEFORE_OUTCOMES",
        "scientific_scope": "A_PARTIAL_PI0FAST_FEASIBILITY_NOT_D_GATE",
        "selection_rule": "inherits metadata-only B2 capture smoke cohort",
        "perturbations": list(PERTURBATIONS),
        "candidate_source_operator": BASE_OPERATOR,
        "candidate_source_rule": (
            "use the synchronously captured and actually executed full 10-step "
            "persistent-fallback chunk; "
            "continue.source is retained only for exact boundary-replay validation"
        ),
        "fixed_repeats": 1,
        "maximum_rollouts": len(captures) * len(PERTURBATIONS),
        "primary_endpoint": "structural execution and within-root outcome diversity",
        "expansion_criterion": (
            "all artifacts valid and at least one root has both success and failure; "
            "K1 effect sizes are never claimed"
        ),
        "kill_criteria": [
            "capture hash mismatch", "boundary continue hash mismatch",
            "non-finite or duplicate perturbation actions",
        ],
        "captures": captures,
        "forbidden_claims": [
            "D_PASS", "semantic_pretraining_gain", "multi_VLA", "closed_loop_gain",
        ],
    }
    protocol_path = output_dir / "PROTOCOL.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit("output directory is bound to a different D0 protocol")
    else:
        atomic_json(protocol_path, protocol)
    return protocol


def preflight_candidates(protocol: dict[str, Any]) -> None:
    """Reject malformed or degenerate perturbations before loading the VLA."""
    for capture in protocol["captures"]:
        meta_path = Path(capture["metadata_path"])
        if file_sha256(meta_path) != capture["metadata_sha256"]:
            raise RuntimeError(f"capture metadata hash mismatch: {meta_path}")
        meta = json.loads(meta_path.read_text())
        arrays_path = Path(meta["arrays_path"])
        if file_sha256(arrays_path) != meta["arrays_sha256"]:
            raise RuntimeError(f"capture arrays hash mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            operator_index = meta["operator_order"].index(BASE_OPERATOR)
            original = arrays["actions"][operator_index][
                arrays["action_step_mask"][operator_index]
            ]
        if original.shape != (10, 7) or not np.isfinite(original).all():
            raise RuntimeError(
                f"{BASE_OPERATOR} must be one finite 10x7 action chunk: "
                f"{meta['group_key']} has {original.shape}"
            )
        candidates = {name: transform(original, name) for name in PERTURBATIONS}
        if len({array_sha256(value) for value in candidates.values()}) != len(candidates):
            raise RuntimeError(f"duplicate perturbation actions for {meta['group_key']}")


def run(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    preflight_candidates(protocol)
    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessLeRobotContinuation, load_lerobot_policy_bundle, restore_pool_state,
    )
    from rase.collect.state_pool import StatePool
    from scripts.collect_rase_vnext_discovery import action_hash, prefix_to_decision

    manifest = json.loads(args.manifest.read_text())
    jobs_by_key: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for job in manifest["jobs"]:
        key = (
            str(job["root_id"]), str(job["policy_id"]),
            str(job["decision_point"]["decision_point_id"]),
            int(job["seed_ledger"]["exact_repeat_replica"]),
        )
        jobs_by_key.setdefault(key, []).append(job)
    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT")); _patch_lerobot_init_states()
    pool = StatePool(Path(str(manifest["root_catalog_pool"])).resolve())
    bundle = load_lerobot_policy_bundle(
        args.policy_path, device=args.device, num_steps=10, n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    continuation = InProcessLeRobotContinuation(bundle)
    result_dir = args.output_dir / "trials"; result_dir.mkdir(parents=True, exist_ok=True)
    for capture in protocol["captures"]:
        meta = json.loads(Path(capture["metadata_path"]).read_text())
        if file_sha256(Path(meta["arrays_path"])) != meta["arrays_sha256"]:
            raise RuntimeError("capture arrays hash mismatch")
        key = tuple(meta["group_key"])
        jobs = jobs_by_key[key]
        exemplar = jobs[0]
        with np.load(meta["arrays_path"], allow_pickle=False) as arrays:
            operator_index = meta["operator_order"].index(BASE_OPERATOR)
            original = arrays["actions"][operator_index][arrays["action_step_mask"][operator_index]]
        candidates = {name: transform(original, name) for name in PERTURBATIONS}
        if len({array_sha256(value) for value in candidates.values()}) != len(candidates):
            raise RuntimeError(f"duplicate perturbation actions for {key}")
        main = restore_pool_state(
            pool, exemplar["state_key"], libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
        )
        source = InProcessLeRobotContinuation(
            bundle, seed=int(exemplar["seed_ledger"]["source_sampling_seed"]),
        )
        try:
            prefix = prefix_to_decision(
                main, source, decision_step=int(exemplar["decision_point"]["value"]),
            )
            if not prefix["available"]:
                raise RuntimeError(f"boundary unavailable for {key}: {prefix.get('reason')}")
            observed = action_hash(prefix["boundary_action"])
            expected = meta["candidate_first_action_sha256"]["continue.source"]
            if observed != expected:
                raise RuntimeError(f"boundary continue hash mismatch for {key}")
            for name, actions in candidates.items():
                digest = hashlib.sha256(("\x1f".join(map(str, key)) + "\x1f" + name).encode()).hexdigest()[:24]
                path = result_dir / f"{digest}.json"
                if path.exists():
                    continue
                branch = restore_pool_state(
                    pool, exemplar["state_key"],
                    libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
                )
                seed = stable_seed("rase-vnext-d0-continuation-v3", *key)
                try:
                    outcome = execute_candidate(
                        branch, prefix["snapshot"], continuation, actions=actions,
                        instruction=str(prefix["instruction"]), continuation_seed=seed,
                    )
                finally:
                    branch.close()
                atomic_json(path, {
                    "schema_version": "rase-vnext-d0-semantic-trial/v3",
                    "group_key": list(key), "task_id": meta["task_id"],
                    "suite": meta["suite"], "perturbation": name,
                    "candidate_source_operator": BASE_OPERATOR,
                    "candidate_action_sha256": array_sha256(actions),
                    "candidate_steps": len(actions), "continuation_seed": seed,
                    **outcome,
                })
                print(f"D0 {meta['suite']} {name} success={outcome['success']}", flush=True)
        finally:
            main.close()


def summarize(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in sorted((output_dir / "trials").glob("*.json"))]
    expected = protocol["maximum_rollouts"]
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows: by_group.setdefault(tuple(row["group_key"]), []).append(row)
    diverse = sum(len({bool(row["success"]) for row in group}) > 1 for group in by_group.values())
    structural = len(rows) == expected and all(
        len(group) == len(PERTURBATIONS) for group in by_group.values()
    )
    if not structural:
        status = "D0_INCOMPLETE"
    elif diverse:
        status = "D0_FEASIBILITY_PASS"
    else:
        status = "D0_NO_OUTCOME_DIVERSITY"
    result = {
        "schema_version": "rase-vnext-d0-semantic-feasibility-summary/v3",
        "status": status,
        "scientific_scope": "FEASIBILITY_ONLY_NOT_D_GATE",
        "expected_rollouts": expected, "observed_rollouts": len(rows),
        "roots": len(by_group), "roots_with_outcome_diversity": diverse,
        "successes_by_perturbation": {
            name: sum(row["perturbation"] == name and bool(row["success"]) for row in rows)
            for name in PERTURBATIONS
        },
        "trials_sha256": {
            path.name: file_sha256(path) for path in sorted((output_dir / "trials").glob("*.json"))
        },
        "next_action": (
            "freeze_independent_K3_semantic_pilot"
            if status == "D0_FEASIBILITY_PASS" else "repair_or_stop_perturbation_design"
        ),
        "forbidden_claims": protocol["forbidden_claims"],
    }
    atomic_json(output_dir / "SUMMARY.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(args.capture_dir.resolve(), args.output_dir.resolve())
    if args.summarize:
        result = summarize(args.output_dir.resolve(), protocol)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "D0_FEASIBILITY_PASS" else 2
    if not args.policy_path or not args.tokenizer_path or not args.action_tokenizer_path:
        raise SystemExit("collection requires policy/tokenizer/action-tokenizer paths")
    run(args, protocol)
    result = summarize(args.output_dir.resolve(), protocol)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "D0_FEASIBILITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

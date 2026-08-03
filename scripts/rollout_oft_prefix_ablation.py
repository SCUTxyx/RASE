#!/usr/bin/env python3
"""Run resumable named action-prefix ablations with an OFT continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_keys(path: Path) -> tuple[list[str], str]:
    payload = _load(path.resolve())
    values = payload if isinstance(payload, list) else payload.get("state_keys") or []
    keys = [str(key) for key in values]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state-key artifact must contain unique non-empty keys")
    checksum = _checksum(keys)
    declared = payload.get("state_keys_sha256") if isinstance(payload, dict) else None
    if declared is not None and str(declared) != checksum:
        raise ValueError("state-key checksum mismatch")
    return keys, checksum


def _suite_from_task_id(task_id: str) -> str:
    if task_id.startswith("libero_spatial"):
        return "libero_spatial"
    if task_id.startswith("libero_object"):
        return "libero_object"
    if task_id.startswith("libero_goal"):
        return "libero_goal"
    if task_id.startswith("libero_10"):
        return "libero_10"
    raise ValueError(f"cannot map task_id to suite: {task_id}")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _adapter_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return adapter options from both supported collection-config layouts."""
    explicit = cfg.get("adapter_config")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise TypeError("adapter_config must be a mapping")
        return dict(explicit)

    legacy = cfg.get("adapter")
    if legacy is None or isinstance(legacy, str):
        return {}
    if not isinstance(legacy, Mapping):
        raise TypeError("adapter must be an import string or mapping")
    return dict(legacy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--arms",
        choices=("full", "direct", "decision-suffix", "suffix-prefix-grid"),
        default="full",
        help=(
            "run candidate grid, direct OFT, strict decision-suffix switch, or "
            "every prefix length of the active suffix"
        ),
    )
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--fresh-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    pool_root = _resolve(ROOT, cfg.get("pool") or "pool/ngc_w5_l1_l2_camera_robot")
    candidates_dir = _resolve(
        ROOT,
        args.candidates_dir
        or cfg.get("candidates_dir")
        or "runs/ngc_w6_l1_l2_candidates_t07",
    )
    output_dir = _resolve(ROOT, args.output_dir)
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh run requires a new output directory: {output_dir}")
    if args.arms == "full" and not candidates_dir.is_dir():
        raise SystemExit(f"candidate directory missing: {candidates_dir}")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact
    from rase.collect.forked_rollout import RolloutConfig, run_one_forked_rollout
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.prefix_ablation import (
        PrefixArm,
        action_prefix_sha256,
        build_decision_suffix_arms,
        build_decision_suffix_prefix_arms,
        build_prefix_arms,
        summarize_decision_suffix_prefix_state,
        summarize_decision_suffix_state,
        summarize_prefix_state,
    )
    from rase.collect.run_manifest import build_run_manifest, write_run_manifest
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.state_pool import StatePool
    from rase.collect.triage_report import write_json
    from rase.interventions.decision_context import strict_continue_suffix
    from rase.oracle.client import OracleClient

    adapter = _adapter_config(cfg)
    oracle_cfg = dict(cfg.get("oracle") or {})
    scheduler_cfg = dict(cfg.get("scheduler") or {})
    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get(
        "libero_plus_root"
    )
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()

    keys, keys_checksum = _load_keys(args.state_keys_json)
    pool = StatePool(pool_root)
    selected = []
    for key in keys:
        meta = pool.read_state(key, load_observations=False).metadata
        if _suite_from_task_id(meta.task_id) == args.suite:
            selected.append(key)
    if not selected:
        raise SystemExit(f"no selected states match suite {args.suite}")

    def arms_for_state(state_key: str) -> tuple[PrefixArm, ...] | list[PrefixArm]:
        if args.arms == "full":
            artifact = load_artifact(candidates_dir / f"{state_key}.npz")
            return build_prefix_arms(artifact.actions)
        if args.arms == "decision-suffix":
            loaded = pool.read_state(state_key, load_observations=False)
            suffix = strict_continue_suffix(loaded.controller_state)
            return build_decision_suffix_arms(suffix)
        if args.arms == "suffix-prefix-grid":
            loaded = pool.read_state(state_key, load_observations=False)
            suffix = strict_continue_suffix(loaded.controller_state)
            return build_decision_suffix_prefix_arms(suffix)
        return [
            PrefixArm("direct_oft", "direct", np.empty((0, 7), dtype=np.float32))
        ]

    endpoint = args.endpoint or oracle_cfg.get("endpoint") or "tcp://127.0.0.1:5555"
    client = OracleClient(
        endpoint, timeout_ms=int(oracle_cfg.get("request_timeout_ms", 60_000))
    )
    try:
        oracle_info = client.model_info()
        if oracle_info.get("suite") not in {None, args.suite}:
            raise SystemExit(
                f"oracle suite {oracle_info.get('suite')!r} != requested {args.suite!r}"
            )
        experiment_by_mode = {
            "full": "oft_prefix_ablation/v1",
            "direct": "oft_direct_escalation/v1",
            "decision-suffix": "oft_decision_suffix_switch/v1",
            "suffix-prefix-grid": "oft_decision_suffix_prefix_grid/v1",
        }
        protocol_by_mode = {
            "full": "oft-prefix-ablation/v1",
            "direct": "oft-direct-escalation/v1",
            "decision-suffix": "oft-decision-suffix-switch/v1",
            "suffix-prefix-grid": "oft-decision-suffix-prefix-grid/v1",
        }
        arms_by_mode = {
            "full": ["direct_oft", "zero_T", "candidate_0..K-1"],
            "direct": ["direct_oft"],
            "decision-suffix": ["direct_oft", "decision_suffix_oft"],
            "suffix-prefix-grid": ["suffix_prefix_0..T"],
        }
        resolved = {
            "experiment": experiment_by_mode[args.arms],
            "suite": args.suite,
            "state_keys": selected,
            "state_keys_sha256": keys_checksum,
            "arms": arms_by_mode[args.arms],
            "prefix_source": (
                "decision_context.active_action_suffix"
                if args.arms in {"decision-suffix", "suffix-prefix-grid"}
                else None
            ),
        }
        policy_path = _resolve(
            ROOT, adapter.get("policy_path") or "ckpts/smolvla_libero"
        )
        candidate_source = (
            candidates_dir if candidates_dir.exists() else args.state_keys_json.resolve()
        )
        write_run_manifest(
            output_dir,
            build_run_manifest(
                repo_root=ROOT,
                resolved_config=resolved,
                pool_root=pool_root,
                candidates_dir=candidate_source,
                policy_path=policy_path,
                policy_hash=str((cfg.get("candidates") or {}).get("policy_hash") or ""),
                protocol_version=protocol_by_mode[args.arms],
                oracle_model_info=oracle_info,
            ),
        )
        scheduler = DiskRolloutScheduler(
            output_dir / "scheduler",
            max_attempts=int(scheduler_cfg.get("max_attempts", 3)),
            lease_seconds=float(scheduler_cfg.get("lease_seconds", 3600)),
        )
        worker = str(scheduler_cfg.get("worker", "oft-prefix-ablation"))
        rollout_cfg = RolloutConfig(
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        started = time.perf_counter()
        for state_key in selected:
            meta = pool.read_state(state_key, load_observations=False).metadata
            arms = arms_for_state(state_key)
            for arm_index, arm in enumerate(arms):
                key = RolloutKey(state_key, arm_index, 0)
                if scheduler.is_complete(key):
                    continue
                claim = scheduler.claim(key, worker)
                if claim is None:
                    if scheduler.result(key) is None:
                        raise RuntimeError(
                            f"cannot claim {key}; retry budget may be exhausted"
                        )
                    continue
                try:
                    predict_before = client.metrics("predict")
                    continuation = OracleChunkContinuation(
                        client, instruction=meta.instruction
                    )
                    result = run_one_forked_rollout(
                        pool,
                        state_key,
                        arm.actions,
                        continuation,
                        libero_plus_root=libero_plus_root,
                        config=rollout_cfg,
                    )
                    if result.candidate_steps > len(arm.actions):
                        raise AssertionError(
                            f"prefix step overflow for {state_key}/{arm.label}: "
                            f"executed={result.candidate_steps} frozen={len(arm.actions)}"
                        )
                    predict_after = client.metrics("predict")
                    prefix = np.asarray(arm.actions, dtype=np.float32)
                    scheduler.complete(
                        key,
                        {
                            **result.to_dict(),
                            "arm_label": arm.label,
                            "arm_kind": arm.kind,
                            "candidate_index": arm.candidate_index,
                            "prefix_source": (
                                "decision_context.active_action_suffix"
                                if arm.kind
                                in {"decision_suffix", "decision_suffix_prefix"}
                                else arm.kind
                            ),
                            "prefix_steps": len(arm.actions),
                            "prefix_sha256": action_prefix_sha256(arm.actions),
                            "prefix_translation_l2_sum": round(
                                float(np.linalg.norm(prefix[:, :3], axis=1).sum()), 8
                            ),
                            "prefix_rotation_l2_sum": round(
                                float(np.linalg.norm(prefix[:, 3:6], axis=1).sum()), 8
                            ),
                            "prefix_gripper_abs_sum": round(
                                float(np.abs(prefix[:, 6]).sum()), 8
                            ),
                            "prefix_completed": result.candidate_steps
                            == len(arm.actions),
                            "terminal_during_prefix": bool(result.success)
                            and result.candidate_steps < len(arm.actions),
                            "oracle": "oft",
                            "oracle_predict_calls": int(
                                predict_after["calls"] - predict_before["calls"]
                            ),
                            "oracle_predict_elapsed_s": round(
                                float(predict_after["elapsed_s"])
                                - float(predict_before["elapsed_s"]),
                                6,
                            ),
                        },
                        worker=worker,
                    )
                    print(
                        f"PREFIX_ABLATION state={state_key} arm={arm.label} "
                        f"success={result.success} steps={result.env_steps}",
                        flush=True,
                    )
                except Exception as exc:
                    scheduler.fail(key, repr(exc), worker=worker)
                    raise

        per_state = []
        for state_key in selected:
            meta = pool.read_state(state_key, load_observations=False).metadata
            arms = arms_for_state(state_key)
            records = []
            for arm_index, arm in enumerate(arms):
                record = scheduler.result(RolloutKey(state_key, arm_index, 0))
                if record is None:
                    raise RuntimeError(
                        f"missing completed result for {state_key}/{arm.label}"
                    )
                records.append(dict(record["result"]))
            metadata = {
                "suite": meta.suite,
                "dim": meta.perturb_dim,
                "level": meta.level,
                "episode_id": meta.episode_id,
            }
            if args.arms == "full":
                per_state.append(
                    summarize_prefix_state(state_key, records, metadata=metadata)
                )
            elif args.arms == "decision-suffix":
                per_state.append(
                    summarize_decision_suffix_state(
                        state_key, records, metadata=metadata
                    )
                )
            elif args.arms == "suffix-prefix-grid":
                suffix_steps = max(int(record["prefix_steps"]) for record in records)
                per_state.append(
                    summarize_decision_suffix_prefix_state(
                        state_key,
                        records,
                        expected_suffix_steps=suffix_steps,
                        metadata=metadata,
                    )
                )
            else:
                direct = records[0]
                per_state.append(
                    {
                        "state_key": state_key,
                        **metadata,
                        "direct_oft_success": bool(direct["success"]),
                        "result": direct,
                    }
                )
        schema_by_mode = {
            "full": "rase-oft-prefix-ablation/v1",
            "direct": "rase-oft-direct-escalation/v1",
            "decision-suffix": "rase-oft-decision-suffix/v1",
            "suffix-prefix-grid": "rase-oft-decision-suffix-prefix-grid/v1",
        }
        summary = {
            "schema_version": schema_by_mode[args.arms],
            "status": "complete",
            "suite": args.suite,
            "state_keys_sha256": keys_checksum,
            "n_states": len(per_state),
            "per_state": per_state,
            "elapsed_wall_s": round(time.perf_counter() - started, 3),
        }
        if args.arms == "full":
            summary["classification_counts"] = {
                label: sum(row["classification"] == label for row in per_state)
                for label in sorted({row["classification"] for row in per_state})
            }
            summary["interpretation"] = (
                "Candidate-specific rescue requires direct_oft=false, zero-prefix=false, "
                "and at least one frozen candidate success."
            )
        elif args.arms == "decision-suffix":
            summary["classification_counts"] = {
                label: sum(row["classification"] == label for row in per_state)
                for label in ("neither", "direct_only", "deferred_only", "both")
            }
            summary["direct_oft"] = {
                "hits": sum(row["direct_oft_success"] for row in per_state),
                "trials": len(per_state),
            }
            summary["decision_suffix_oft"] = {
                "hits": sum(
                    row["decision_suffix_oft_success"] for row in per_state
                ),
                "trials": len(per_state),
            }
            summary["prefix_source"] = "decision_context.active_action_suffix"
            summary["interpretation"] = (
                "Immediate and active-suffix-preserving OFT switches from the same "
                "snapshot; calibration only, not independent confirmation."
            )
        elif args.arms == "suffix-prefix-grid":
            summary["success_patterns"] = {
                pattern: sum(row["success_pattern"] == pattern for row in per_state)
                for pattern in sorted({row["success_pattern"] for row in per_state})
            }
            summary["single_transition_states"] = sum(
                row["single_transition"] for row in per_state
            )
            summary["prefix_source"] = "decision_context.active_action_suffix"
            summary["interpretation"] = (
                "Exploratory selected-disagreement mechanism audit across every "
                "active-suffix prefix length; not a population or selector claim."
            )
        else:
            summary["direct_oft"] = {
                "hits": sum(row["direct_oft_success"] for row in per_state),
                "trials": len(per_state),
            }
            summary["interpretation"] = (
                "Direct OFT is a deployable escalation arm with no candidate or "
                "time-matched zero prefix."
            )
        write_json(output_dir / "summary.json", summary)
        print(
            json.dumps(
                summary.get("classification_counts")
                or summary.get("success_patterns")
                or summary.get("direct_oft"),
                sort_keys=True,
            ),
            flush=True,
        )
        print(f"SUMMARY {output_dir / 'summary.json'}", flush=True)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

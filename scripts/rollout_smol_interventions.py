#!/usr/bin/env python3
"""Strict-CONTINUE and optional paired REPLAN decision-state rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mcnemar_exact(continue_only: int, replan_only: int) -> float:
    disagreements = continue_only + replan_only
    if disagreements == 0:
        return 1.0
    tail = sum(
        math.comb(disagreements, k)
        for k in range(0, min(continue_only, replan_only) + 1)
    ) / (2**disagreements)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--continuation-seeds", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("paired", "continue-only"),
        default="paired",
        help="run the legacy paired comparison or only the preregistered CONTINUE arm",
    )
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--fresh-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.continuation_seeds < 1:
        raise SystemExit("--continuation-seeds must be positive")

    cfg = _load(args.config.resolve())
    collection = dict(cfg.get("collection") or {})
    adapter = dict(cfg.get("adapter_config") or {})
    if not adapter and isinstance(cfg.get("adapter"), dict):
        adapter = dict(cfg["adapter"])
    pool_root = _resolve(cfg.get("pool") or collection.get("output_dir") or "")
    output_dir = _resolve(args.output_dir)
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh run requires a new output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    key_payload = _load(args.state_keys_json.resolve())
    values = key_payload if isinstance(key_payload, list) else key_payload.get("state_keys") or []
    keys = [str(value) for value in values]
    if not keys or len(keys) != len(set(keys)):
        raise SystemExit("state-key artifact must contain unique non-empty keys")
    declared = key_payload.get("state_keys_sha256") if isinstance(key_payload, dict) else None
    key_checksum = _checksum(keys)
    if declared is not None and str(declared) != key_checksum:
        raise SystemExit("state-key checksum mismatch")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        RolloutConfig,
        load_smolvla_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
    )
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.smolvla_candidate_policy import checkpoint_sha256
    from rase.collect.state_pool import StatePool
    from rase.interventions.dataset import registry_payload
    from rase.interventions.decision_context import strict_continue_suffix
    from rase.interventions.schema import (
        CostVector,
        Feasibility,
        InterventionOutcome,
        InterventionSnapshot,
        OperatorFamily,
        OperatorSpec,
    )

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_root)
    loaded = {key: pool.read_state(key, load_observations=False) for key in keys}
    suffixes = {
        key: strict_continue_suffix(state.controller_state)
        for key, state in loaded.items()
    }

    policy_path = _resolve(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_raw = adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"
    tokenizer_path = _resolve(tokenizer_raw) if tokenizer_raw else None
    temperature = float(adapter.get("continuation_temperature", 0.5))
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=temperature,
    )
    bundle = load_smolvla_policy_bundle(
        policy_path,
        device=str(adapter.get("device", "cuda")),
        num_steps=rollout_cfg.num_steps,
        n_action_steps=rollout_cfg.n_action_steps,
        tokenizer_path=tokenizer_path,
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )

    all_specs = [
        OperatorSpec(
            "continue_smol_active_chunk",
            OperatorFamily.CONTINUE,
            "smolvla",
            "active_action_suffix",
            parameters={
                "active_suffix_preserved": True,
                "post_suffix_policy_reset": True,
                "then": "normal_source_policy",
            },
            requires=("active_action_suffix", "public_observation"),
        ),
        OperatorSpec(
            "replan_smol",
            OperatorFamily.REPLAN,
            "smolvla",
            "current_observation",
            parameters={"discard_active_chunk": True, "policy_reset": True},
            requires=("public_observation",),
        ),
    ]
    specs = all_specs if args.profile == "paired" else all_specs[:1]
    _write_json(output_dir / "operators.json", registry_payload(specs))
    snapshots = []
    for key in keys:
        state = loaded[key]
        context = state.controller_state["decision_context"]
        snapshots.append(
            InterventionSnapshot(
                snapshot_id=key,
                state_key=key,
                task_id=state.metadata.task_id,
                episode_id=state.metadata.episode_id,
                step=state.metadata.step,
                source_policy=str(context["source_policy"]),
                restore_state_ref=f"state_pool:{key}",
                public_history_ref=f"state_pool:{key}#decision_context.public_history",
                active_action_suffix_ref=f"state_pool:{key}#decision_context.active_action_suffix",
                suite=state.metadata.suite,
                perturbation={
                    "dimension": state.metadata.perturb_dim,
                    "subdimension": state.metadata.perturb_sub,
                    "level": state.metadata.level,
                },
            ).to_dict()
        )
    _write_jsonl(output_dir / "snapshots.jsonl", snapshots)

    manifest = {
        "schema_version": "rase-smol-intervention-run/v1",
        "pool": str(pool_root),
        "state_keys_sha256": key_checksum,
        "n_states": len(keys),
        "continuation_seeds": args.continuation_seeds,
        "policy_path": str(policy_path),
        "policy_sha256": checkpoint_sha256(policy_path),
        "temperature": temperature,
        "paired_noise": args.profile == "paired",
        "profile": args.profile,
    }
    existing_manifest = output_dir / "run_manifest.json"
    if args.resume and existing_manifest.is_file():
        if _load(existing_manifest) != manifest:
            raise SystemExit("resume manifest differs from requested experiment")
    else:
        _write_json(existing_manifest, manifest)

    scheduler = DiskRolloutScheduler(output_dir / "scheduler", max_attempts=3, lease_seconds=3600)
    all_operators = [
        ("continue_smol_active_chunk", 0),
        ("replan_smol", 1),
    ]
    operators = all_operators if args.profile == "paired" else all_operators[:1]
    started = time.perf_counter()
    for state_key in keys:
        for repeat in range(args.continuation_seeds):
            paired_seed = rollout_seed(state_key, 0, repeat, salt=0x52415345)
            for operator_id, candidate_id in operators:
                rollout_key = RolloutKey(state_key, candidate_id, repeat)
                if scheduler.is_complete(rollout_key):
                    continue
                worker = f"smol-intervention-{operator_id}"
                claim = scheduler.claim(rollout_key, worker)
                if claim is None:
                    if scheduler.result(rollout_key) is None:
                        raise RuntimeError(f"cannot claim {rollout_key}")
                    continue
                prefix = (
                    suffixes[state_key]
                    if operator_id == "continue_smol_active_chunk"
                    else np.empty((0, 7), dtype=np.float32)
                )
                try:
                    continuation = InProcessSmolVLAContinuation(
                        bundle, temperature=temperature, seed=paired_seed
                    )
                    result = run_one_forked_rollout(
                        pool,
                        state_key,
                        prefix,
                        continuation,
                        libero_plus_root=libero_plus_root,
                        config=rollout_cfg,
                    )
                    scheduler.complete(
                        rollout_key,
                        {
                            **result.to_dict(),
                            **continuation.metrics(),
                            "operator_id": operator_id,
                            "continuation_seed": paired_seed,
                            "prefix_steps": len(prefix),
                            "outcome_semantics": (
                                "strict_active_suffix_then_source_policy"
                                if prefix.size
                                else "discard_active_suffix_replan_source_policy"
                            ),
                        },
                        worker=worker,
                    )
                    print(
                        f"ARM state={state_key} repeat={repeat} operator={operator_id} "
                        f"success={result.success} steps={result.env_steps}",
                        flush=True,
                    )
                except Exception as exc:
                    scheduler.fail(rollout_key, repr(exc), worker=worker)
                    raise

    outcomes = []
    rows = []
    for state_key in keys:
        for repeat in range(args.continuation_seeds):
            paired_seed = rollout_seed(state_key, 0, repeat, salt=0x52415345)
            pair = {
                "state_key": state_key,
                "repeat": repeat,
                "continuation_seed": paired_seed,
            }
            for operator_id, candidate_id in operators:
                record = scheduler.result(RolloutKey(state_key, candidate_id, repeat))
                if record is None:
                    raise RuntimeError(f"missing result for {state_key}/{operator_id}/{repeat}")
                result = dict(record["result"])
                pair[operator_id] = bool(result["success"])
                pair[f"{operator_id}_env_steps"] = int(result["env_steps"])
                pair[f"{operator_id}_latency_seconds"] = float(result["elapsed_s"])
                if "action_select_elapsed_s" in result:
                    pair[f"{operator_id}_action_select_calls"] = int(
                        result["action_select_calls"]
                    )
                    pair[f"{operator_id}_action_select_elapsed_s"] = float(
                        result["action_select_elapsed_s"]
                    )
                outcome = InterventionOutcome(
                    snapshot_id=state_key,
                    operator_id=operator_id,
                    continuation_seed=paired_seed,
                    feasibility=Feasibility(feasible=True),
                    observed=True,
                    success=bool(result["success"]),
                    operator_completed=True,
                    stop_reason=str(result["stop_reason"]),
                    utility_cost=0.0,
                    cost_source="phase0_zero_utility_cost",
                    costs=CostVector(
                        compute_seconds=float(result["elapsed_s"]),
                        latency_seconds=float(result["elapsed_s"]),
                        env_steps=int(result["env_steps"]),
                    ),
                    outcome_semantics=str(result["outcome_semantics"]),
                )
                outcomes.append(outcome.to_dict())
            rows.append(pair)
    _write_jsonl(output_dir / "outcomes.jsonl", outcomes)
    continue_hits = sum(row["continue_smol_active_chunk"] for row in rows)

    def action_select_summary(operator_id: str) -> dict[str, Any]:
        measured = [
            row
            for row in rows
            if f"{operator_id}_action_select_elapsed_s" in row
        ]
        calls = sum(
            int(row[f"{operator_id}_action_select_calls"]) for row in measured
        )
        elapsed_s = sum(
            float(row[f"{operator_id}_action_select_elapsed_s"]) for row in measured
        )
        return {
            "measurement_scope": (
                "wall time inside SmolVLA select_env_action; includes cached action "
                "queue access and model forward passes, excludes environment stepping"
            ),
            "n_trials": len(rows),
            "n_measured_trials": len(measured),
            "coverage": len(measured) / len(rows) if rows else None,
            "action_select_calls": calls,
            "action_select_elapsed_s": elapsed_s,
            "mean_action_select_elapsed_s_per_measured_trial": (
                elapsed_s / len(measured) if measured else None
            ),
        }

    summary = {
        "schema_version": (
            "rase-smol-intervention-summary/v1"
            if args.profile == "paired"
            else "rase-smol-intervention-summary/v2"
        ),
        "status": "complete",
        "profile": args.profile,
        "n_states": len(keys),
        "n_trials": len(rows),
        "continue": {
            "hits": continue_hits,
            "rate": continue_hits / len(rows),
            "mean_env_steps": float(
                np.mean(
                    [row["continue_smol_active_chunk_env_steps"] for row in rows]
                )
            ),
        },
        "action_selection_metrics": {
            operator_id: action_select_summary(operator_id)
            for operator_id, _candidate_id in operators
        },
        "per_pair": rows,
        "elapsed_wall_s": round(time.perf_counter() - started, 3),
    }
    if args.profile == "paired":
        replan_hits = sum(row["replan_smol"] for row in rows)
        continue_only = sum(
            row["continue_smol_active_chunk"] and not row["replan_smol"]
            for row in rows
        )
        replan_only = sum(
            row["replan_smol"] and not row["continue_smol_active_chunk"]
            for row in rows
        )
        summary["n_paired_trials"] = len(rows)
        summary["replan"] = {
            "hits": replan_hits,
            "rate": replan_hits / len(rows),
            "mean_env_steps": float(
                np.mean([row["replan_smol_env_steps"] for row in rows])
            ),
        }
        summary["paired"] = {
            "continue_only": continue_only,
            "replan_only": replan_only,
            "both_success": sum(
                row["continue_smol_active_chunk"] and row["replan_smol"]
                for row in rows
            ),
            "both_failure": sum(
                not row["continue_smol_active_chunk"] and not row["replan_smol"]
                for row in rows
            ),
            "mcnemar_exact_p": _mcnemar_exact(continue_only, replan_only),
        }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

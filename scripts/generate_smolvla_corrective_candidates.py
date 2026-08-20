#!/usr/bin/env python3
"""Run PRE-C0 same-policy corrective arms from frozen deviation snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rase.collect.candidates import generate_candidates
from rase.collect.forked_rollout import (
    InProcessSmolVLAContinuation,
    RolloutConfig,
    load_smolvla_policy_bundle,
    restore_pool_state,
    rollout_seed,
    run_one_forked_rollout,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.smolvla_candidate_policy import (
    SmolVLACandidatePolicy,
    action_tensor_sha256,
    cache_initialization_fingerprint,
    checkpoint_sha256,
)
from rase.collect.state_pool import StatePool
from rase.interventions.decision_context import strict_continue_suffix

SCHEMA_VERSION = "rase-pre-c0-corrective-rollout/v1"


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha(value: Any) -> str:
    raw = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _observation_sha(observations: dict[str, bytes] | Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(dict(observations).items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(value)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _arm_record(
    *,
    family: str,
    arm_name: str,
    seed: int,
    result: Any,
    continuation: Any,
    candidate: np.ndarray,
    generation_context: str,
    execution_horizon: int | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "arm_name": arm_name,
        "seed": int(seed),
        "generation_context": generation_context,
        "execution_horizon": execution_horizon,
        "candidate_steps": len(candidate),
        "action_sha256": (
            action_tensor_sha256(candidate) if len(candidate) else None
        ),
        "action_tensor": candidate.tolist() if len(candidate) else [],
        **result.to_dict(),
        "policy_metrics": continuation.metrics(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["T1", "T3"])
    parser.add_argument("--limit", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    collection = dict(cfg.get("collection") or {})
    adapter = dict(cfg.get("adapter_config") or cfg.get("adapter") or {})
    pool_root = Path(cfg.get("pool") or collection["output_dir"]).resolve()
    output_dir = args.output_dir.resolve()
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_payload = _load(args.stage_keys.resolve())
    raw_records = (
        stage_payload.get("records")
        or stage_payload.get("selected_states")
        or []
    )
    records = [
        dict(row)
        for row in raw_records
        if str(row.get("stage")) in set(args.stages)
    ]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no requested PRE-C0 stage records")

    pool = StatePool(pool_root)
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero").resolve()
    tokenizer = Path(
        adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"
    ).resolve()
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
        tokenizer_path=tokenizer,
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )
    candidate_policy = SmolVLACandidatePolicy(
        policy=bundle["policy"],
        preprocessor=bundle["preprocessor"],
        env_preprocessor=bundle["env_preprocessor"],
        postprocessor=bundle["postprocessor"],
        env_postprocessor=bundle["env_postprocessor"],
        chunk_length=rollout_cfg.n_action_steps,
        checkpoint=str(policy_path),
    )
    policy_sha = checkpoint_sha256(policy_path)
    libero_plus_root = adapter.get("libero_plus_root")

    manifest = {
        "schema_version": "rase-pre-c0-corrective-run/v1",
        "pool": str(pool_root),
        "stage_keys": str(args.stage_keys.resolve()),
        "stages": list(args.stages),
        "n_states": len(records),
        "policy_sha256": policy_sha,
        "temperature": temperature,
        "strict_resample_k": 8,
        "fresh_replan_k": 4,
        "execution_horizons": [1, 2, 4],
        "runtime_policy": "smolvla_only",
    }
    manifest_path = output_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        if _load(manifest_path) != manifest:
            raise SystemExit("resume manifest differs from requested PRE-C0 run")
    else:
        _atomic_json(manifest_path, manifest)

    for ordinal, record in enumerate(records):
        state_key = str(record["state_key"])
        target = output_dir / f"{ordinal:03d}_{record['stage']}_{state_key}.json"
        if args.resume and target.exists():
            print(f"SKIP state={state_key} stage={record['stage']}", flush=True)
            continue
        loaded = pool.read_state(state_key)
        context = dict(loaded.controller_state["decision_context"])
        observation_sha = _observation_sha(dict(loaded.observations))
        history_sha = _sha(
            {
                "observations": context.get("public_observation_history"),
                "proprio": context.get("public_proprio_history"),
                "actions": context.get("public_action_history"),
            }
        )
        cache_sha = cache_initialization_fingerprint(
            state_key=state_key,
            observation_sha256=observation_sha,
            history_sha256=history_sha,
            policy_sha256=policy_sha,
            reset_policy=True,
        )

        restored = restore_pool_state(
            pool,
            state_key,
            libero_plus_root=libero_plus_root,
            observation_height=rollout_cfg.observation_height,
            observation_width=rollout_cfg.observation_width,
        )
        try:
            observation = observation_from_libero_env(restored.handle.vector_env.envs[0])
            candidate_seed = rollout_seed(state_key, 100, 0, salt=0xC0)
            artifact = generate_candidates(
                candidate_policy,
                observation,
                k=8,
                temperature=temperature,
                base_seed=candidate_seed % (2**32 - 8),
                policy_hash=policy_sha,
            )
        finally:
            restored.close()

        arms: list[dict[str, Any]] = []
        paired_seed = rollout_seed(state_key, 0, 0, salt=0xC0)
        suffix = strict_continue_suffix(loaded.controller_state)
        continuation = InProcessSmolVLAContinuation(
            bundle, temperature=temperature, seed=paired_seed
        )
        result = run_one_forked_rollout(
            pool,
            state_key,
            suffix,
            continuation,
            libero_plus_root=libero_plus_root,
            config=rollout_cfg,
        )
        arms.append(
            _arm_record(
                family="current_suffix",
                arm_name="current_suffix",
                seed=paired_seed,
                result=result,
                continuation=continuation,
                candidate=suffix,
                generation_context="stored_active_suffix",
            )
        )

        for candidate_index, candidate in enumerate(artifact.actions):
            seed = int(artifact.metadata.seeds[candidate_index])
            continuation = InProcessSmolVLAContinuation(
                bundle, temperature=temperature, seed=paired_seed
            )
            result = run_one_forked_rollout(
                pool,
                state_key,
                candidate,
                continuation,
                libero_plus_root=libero_plus_root,
                config=rollout_cfg,
            )
            arms.append(
                _arm_record(
                    family="strict_resample",
                    arm_name=f"strict_resample_{candidate_index}",
                    seed=seed,
                    result=result,
                    continuation=continuation,
                    candidate=candidate,
                    generation_context="exact_snapshot_fixed_profile_seed_only",
                )
            )

        empty = np.empty((0, 7), dtype=np.float32)
        for repeat in range(4):
            seed = rollout_seed(state_key, 200, repeat, salt=0xC0)
            continuation = InProcessSmolVLAContinuation(
                bundle, temperature=temperature, seed=seed
            )
            result = run_one_forked_rollout(
                pool,
                state_key,
                empty,
                continuation,
                libero_plus_root=libero_plus_root,
                config=rollout_cfg,
            )
            arms.append(
                _arm_record(
                    family="fresh_replan",
                    arm_name=f"fresh_replan_{repeat}",
                    seed=seed,
                    result=result,
                    continuation=continuation,
                    candidate=empty,
                    generation_context="latest_observation_discard_suffix_reset_cache",
                )
            )

        for horizon in (1, 2, 4):
            seed = rollout_seed(state_key, 300 + horizon, 0, salt=0xC0)
            continuation = RecedingHorizonSmolVLAContinuation(
                bundle,
                execution_horizon=horizon,
                temperature=temperature,
                seed=seed,
            )
            result = run_one_forked_rollout(
                pool,
                state_key,
                empty,
                continuation,
                libero_plus_root=libero_plus_root,
                config=rollout_cfg,
            )
            arms.append(
                _arm_record(
                    family="receding_horizon",
                    arm_name=f"receding_horizon_{horizon}",
                    seed=seed,
                    result=result,
                    continuation=continuation,
                    candidate=empty,
                    generation_context="latest_observation_reobserve_and_reset",
                    execution_horizon=horizon,
                )
            )

        family_success = {
            family: any(
                bool(arm["success"]) for arm in arms if arm["family"] == family
            )
            for family in (
                "current_suffix",
                "strict_resample",
                "fresh_replan",
                "receding_horizon",
            )
        }
        output = {
            "schema_version": SCHEMA_VERSION,
            "state_key": state_key,
            "episode_id": loaded.metadata.episode_id,
            "task_id": record.get("logical_task_id") or record.get("task_id"),
            "concrete_task_id": loaded.metadata.task_id,
            "suite": record.get("suite") or loaded.metadata.suite,
            "cell": record.get("cell"),
            "stage": record["stage"],
            "source_episode_outcome": loaded.metadata.episode_outcome,
            "snapshot_sha256": _sha(
                {
                    "state_key": state_key,
                    "sim_state": loaded.sim_state,
                    "controller_state": loaded.controller_state,
                }
            ),
            "observation_sha256": observation_sha,
            "history_sha256": history_sha,
            "cache_initialization_sha256": cache_sha,
            "policy_sha256": policy_sha,
            "family_success": family_success,
            "arms": arms,
        }
        _atomic_json(target, output)
        print(
            f"PRE_C0_STATE_DONE ordinal={ordinal} state={state_key} "
            f"stage={record['stage']} success={family_success}",
            flush=True,
        )
    print(f"PRE_C0_CORRECTIVE_DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one direct SmolVLA continuation from every frozen snapshot."""

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--fresh-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    pool_root = _resolve(cfg.get("pool") or "pool/ngc_w5_l1_l2_camera_robot")
    output_dir = _resolve(args.output_dir)
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh run requires a new output directory: {output_dir}")
    key_payload = _load(args.state_keys_json.resolve())
    values = key_payload if isinstance(key_payload, list) else key_payload.get("state_keys") or []
    keys = [str(value) for value in values]
    if not keys or len(keys) != len(set(keys)):
        raise SystemExit("state-key artifact must contain unique non-empty keys")
    key_checksum = _checksum(keys)
    declared = key_payload.get("state_keys_sha256") if isinstance(key_payload, dict) else None
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
    from rase.collect.run_manifest import build_run_manifest, write_run_manifest
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.smolvla_candidate_policy import checkpoint_sha256
    from rase.collect.state_pool import StatePool
    from rase.collect.triage_report import write_json

    adapter = dict(cfg.get("adapter") or {})
    scheduler_cfg = dict(cfg.get("scheduler") or {})
    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_root)
    for key in keys:
        pool.read_state(key, load_observations=False)

    policy_path = _resolve(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_raw = adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"
    tokenizer_path = _resolve(tokenizer_raw) if tokenizer_raw else None
    policy_hash = str((cfg.get("candidates") or {}).get("policy_hash") or checkpoint_sha256(policy_path))
    temperature = float(adapter.get("continuation_temperature", 0.5))
    resolved = {
        "experiment": "smol_direct_continuation/v1",
        "state_keys": keys,
        "state_keys_sha256": key_checksum,
        "temperature": temperature,
        "one_shot": True,
        "prefix_steps": 0,
    }
    write_run_manifest(
        output_dir,
        build_run_manifest(
            repo_root=ROOT,
            resolved_config=resolved,
            pool_root=pool_root,
            candidates_dir=args.state_keys_json.resolve(),
            policy_path=policy_path,
            policy_hash=policy_hash,
            protocol_version="smol-direct-continuation/v1",
        ),
    )
    scheduler = DiskRolloutScheduler(
        output_dir / "scheduler",
        max_attempts=int(scheduler_cfg.get("max_attempts", 3)),
        lease_seconds=float(scheduler_cfg.get("lease_seconds", 3600)),
    )
    worker = str(scheduler_cfg.get("worker", "smol-direct")) + "-direct"
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
    started = time.perf_counter()
    for state_key in keys:
        rollout_key = RolloutKey(state_key, 0, 0)
        if scheduler.is_complete(rollout_key):
            continue
        claim = scheduler.claim(rollout_key, worker)
        if claim is None:
            if scheduler.result(rollout_key) is None:
                raise RuntimeError(f"cannot claim {rollout_key}")
            continue
        seed = rollout_seed(state_key, 0, 0, salt=0x534D4F4C)
        try:
            continuation = InProcessSmolVLAContinuation(
                bundle, temperature=temperature, seed=seed
            )
            result = run_one_forked_rollout(
                pool,
                state_key,
                np.empty((0, 7), dtype=np.float32),
                continuation,
                libero_plus_root=libero_plus_root,
                config=rollout_cfg,
            )
            scheduler.complete(
                rollout_key,
                {
                    **result.to_dict(),
                    "oracle": "smolvla",
                    "rollout_seed": seed,
                    "continuation_temperature": temperature,
                    "outcome_semantics": "direct_smol_from_snapshot",
                },
                worker=worker,
            )
            print(
                f"DIRECT_SMOL state={state_key} success={result.success} steps={result.env_steps}",
                flush=True,
            )
        except Exception as exc:
            scheduler.fail(rollout_key, repr(exc), worker=worker)
            raise

    per_state = []
    for state_key in keys:
        record = scheduler.result(RolloutKey(state_key, 0, 0))
        if record is None:
            raise RuntimeError(f"missing direct Smol result for {state_key}")
        result = dict(record["result"])
        meta = pool.read_state(state_key, load_observations=False).metadata
        per_state.append(
            {
                "state_key": state_key,
                "suite": meta.suite,
                "dim": meta.perturb_dim,
                "level": meta.level,
                "episode_id": meta.episode_id,
                "direct_smol_success": bool(result["success"]),
                "result": result,
            }
        )
    summary = {
        "schema_version": "rase-smol-direct-continuation/v1",
        "status": "complete",
        "state_keys_sha256": key_checksum,
        "n_states": len(per_state),
        "direct_smol": {
            "hits": sum(row["direct_smol_success"] for row in per_state),
            "trials": len(per_state),
        },
        "per_state": per_state,
        "elapsed_wall_s": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["direct_smol"], sort_keys=True), flush=True)
    print(f"SUMMARY {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

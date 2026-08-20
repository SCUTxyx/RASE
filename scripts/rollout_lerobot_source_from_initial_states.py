#!/usr/bin/env python3
"""Evaluate one frozen LeRobot source VLA from exact pre-action reset snapshots."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    weights = sorted(path.glob("model*.safetensors"))
    if not config.is_file() or not weights:
        raise ValueError(f"incomplete LeRobot checkpoint: {path}")
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256(config),
        "weight_files": [
            {"name": weight.name, "size": weight.stat().st_size}
            for weight in weights
        ],
    }


def tokenizer_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    files = sorted(
        file for file in resolved.rglob("*")
        if file.is_file() and ".cache" not in file.parts
    )
    if not files:
        raise ValueError(f"empty tokenizer directory: {resolved}")
    return {
        "path": str(resolved),
        "files": [
            {
                "name": str(file.relative_to(resolved)),
                "size": file.stat().st_size,
                "sha256": sha256(file),
            }
            for file in files
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument(
        "--max-states", type=int, default=0,
        help="Diagnostic smoke-test limit; zero evaluates the frozen 48-state cohort.",
    )
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()

    keys_payload = read_json(args.initial_keys.resolve())
    if keys_payload.get("status") != "frozen":
        raise ValueError("initial state-key artifact is not frozen")
    keys = [str(value) for value in keys_payload["state_keys"]]
    if len(keys) != 48 or len(keys) != len(set(keys)):
        raise ValueError("R6-A requires exactly 48 unique reset states")
    if any(int(row["snapshot_policy_step"]) != 0 for row in keys_payload["records"]):
        raise ValueError("R6-A source states must precede every source action")
    if args.max_states < 0 or args.max_states > len(keys):
        raise ValueError("--max-states must be in [0, 48]")
    evaluation_keys = keys[: args.max_states] if args.max_states else keys

    policy_path = args.policy_path.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if args.fresh_run and output_dir.exists():
        raise ValueError(f"fresh output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessLeRobotContinuation,
        RolloutConfig,
        load_lerobot_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
    )
    from rase.collect.state_pool import StatePool

    pool_root = Path(str(keys_payload["pool"])).resolve()
    pool = StatePool(pool_root)
    for state_key in evaluation_keys:
        pool.read_state(state_key, load_observations=False)
    ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
    _patch_lerobot_init_states()

    bundle = load_lerobot_policy_bundle(
        policy_path,
        device=args.device,
        num_steps=10,
        n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360,
        observation_width=360,
    )
    rollout_cfg = RolloutConfig(
        n_action_steps=10,
        num_steps=10,
        observation_height=360,
        observation_width=360,
    )
    policy_salt = int.from_bytes(hashlib.sha256(args.policy_id.encode()).digest()[:4], "big")
    per_state = []
    started = time.perf_counter()
    for index, state_key in enumerate(evaluation_keys):
        seed = rollout_seed(
            state_key, args.seed_index, 0,
            salt=policy_salt ^ (0xA16A0000 + args.seed_index),
        )
        continuation = InProcessLeRobotContinuation(bundle, seed=seed)
        result = run_one_forked_rollout(
            pool,
            state_key,
            np.empty((0, 7), dtype=np.float32),
            continuation,
            libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
            config=rollout_cfg,
        )
        meta = pool.read_state(state_key, load_observations=False).metadata
        row = {
            "state_key": state_key,
            "task_id": meta.task_id,
            "episode_id": meta.episode_id,
            "suite": meta.suite,
            "dimension": meta.perturb_dim,
            "level": meta.level,
            "policy_id": args.policy_id,
            "seed_index": args.seed_index,
            "rollout_seed": seed,
            "source_success": bool(result.success),
            "result": result.to_dict(),
            "policy_metrics": continuation.metrics(),
        }
        per_state.append(row)
        print(
            f"R6A_SOURCE policy={args.policy_id} seed={args.seed_index} "
            f"state={index + 1}/{len(evaluation_keys)} "
            f"success={result.success} steps={result.env_steps}",
            flush=True,
        )

    summary = {
        "schema_version": "rase-r6a-source-reset-rollout/v1",
        "status": "complete" if len(evaluation_keys) == 48 else "diagnostic_smoke",
        "scientific_scope": "development-only initial-state opportunity screen",
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys.resolve()),
        "state_keys_sha256": keys_payload["state_keys_sha256"],
        "policy_id": args.policy_id,
        "policy_identity": checkpoint_identity(policy_path),
        "text_tokenizer_identity": tokenizer_identity(args.tokenizer_path),
        "action_tokenizer_identity": tokenizer_identity(args.action_tokenizer_path),
        "seed_index": args.seed_index,
        "n_states": len(per_state),
        "n_tasks": len({row["task_id"] for row in per_state}),
        "source_successes": sum(row["source_success"] for row in per_state),
        "source_success_rate": sum(row["source_success"] for row in per_state) / len(per_state),
        "elapsed_wall_s": time.perf_counter() - started,
        "per_state": per_state,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("policy_id", "seed_index", "source_successes", "source_success_rate", "elapsed_wall_s")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

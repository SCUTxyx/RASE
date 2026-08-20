#!/usr/bin/env python3
"""Resume-safe direct takeover by a frozen LeRobot policy from pool snapshots."""

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def checksum(keys: list[str]) -> str:
    raw = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = read_json(args.state_keys_json.resolve())
    keys = [str(value) for value in payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state-key artifact must contain unique non-empty keys")
    if payload.get("state_keys_sha256") != checksum(keys):
        raise ValueError("state-key checksum mismatch")

    from rase.collect.forked_rollout import (
        InProcessLeRobotContinuation,
        RolloutConfig,
        load_lerobot_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
    )
    from rase.collect.state_pool import StatePool

    pool = StatePool(args.pool.resolve())
    for key in keys:
        pool.read_state(key, load_observations=False)
    bundle = load_lerobot_policy_bundle(
        args.policy_path.resolve(),
        device=args.device,
        num_steps=10,
        n_action_steps=10,
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360,
        observation_width=360,
    )
    cfg = RolloutConfig(n_action_steps=10, num_steps=10)
    output = args.output_dir.resolve()
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    policy_salt = int.from_bytes(hashlib.sha256(args.policy_id.encode()).digest()[:4], "big")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, key in enumerate(keys):
        target = episode_dir / f"{key}.json"
        if target.is_file():
            row = read_json(target)
            if row.get("state_keys_sha256") != payload["state_keys_sha256"]:
                raise ValueError(f"stale direct result {target}")
            skipped = True
        else:
            loaded = pool.read_state(key, load_observations=False)
            seed = rollout_seed(key, 0, 0, salt=policy_salt ^ 0x47324231)
            continuation = InProcessLeRobotContinuation(bundle, seed=seed, capture=False)
            result = run_one_forked_rollout(
                pool,
                key,
                np.empty((0, 7), dtype=np.float32),
                continuation,
                libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
                config=cfg,
            )
            direct = {
                **result.to_dict(),
                "prefix_source": "direct",
                "prefix_steps": 0,
                "outcome_semantics": "direct_frozen_lerobot_takeover_to_true_terminal",
            }
            row = {
                "state_key": key,
                "task_id": loaded.metadata.task_id,
                "episode_id": loaded.metadata.episode_id,
                "suite": loaded.metadata.suite,
                "policy_id": args.policy_id,
                "rollout_seed": seed,
                "result": direct,
                "policy_metrics": continuation.metrics(),
                "state_keys_sha256": payload["state_keys_sha256"],
            }
            write_json(target, row)
            skipped = False
        rows.append(row)
        print(
            f"DIRECT state={index + 1}/{len(keys)} key={key} "
            f"success={row['result']['success']} steps={row['result']['env_steps']} skipped={skipped}",
            flush=True,
        )

    summary = {
        "schema_version": "rase-lerobot-direct-fallback/v1",
        "status": "complete",
        "scientific_scope": "development same-root cross-policy eligibility screen",
        "policy_id": args.policy_id,
        "pool": str(args.pool.resolve()),
        "state_keys_json": str(args.state_keys_json.resolve()),
        "state_keys_sha256": payload["state_keys_sha256"],
        "n_states": len(rows),
        "successes": sum(bool(row["result"]["success"]) for row in rows),
        "success_rate": sum(bool(row["result"]["success"]) for row in rows) / len(rows),
        "elapsed_wall_s": time.perf_counter() - started,
        "per_state": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("n_states", "successes", "success_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

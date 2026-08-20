#!/usr/bin/env python3
"""Collect resume-safe multi-seed reference traces from exact pool roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_SCHEMA = "rase-e3v-reference-roots/v1"
RESULT_SCHEMA = "rase-e3v-reference-trace/v1"


def canonical_sha256(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_protocol(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != PROTOCOL_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("unsupported or unfrozen E3-V protocol")
    if payload.get("selection_uses_outcomes") is not True:
        raise ValueError("E3-V must explicitly declare outcome-selected development roots")
    records = [dict(row) for row in payload.get("records") or []]
    if not records or len(records) != int(payload.get("n_states", -1)):
        raise ValueError("invalid E3-V records")
    if len({str(row["state_key"]) for row in records}) != len(records):
        raise ValueError("duplicate E3-V state keys")
    if canonical_sha256(records) != payload.get("records_sha256"):
        raise ValueError("E3-V records checksum mismatch")
    no_hash = dict(payload)
    claimed = no_hash.pop("protocol_sha256", None)
    if canonical_sha256(no_hash) != claimed:
        raise ValueError("E3-V protocol checksum mismatch")
    return records


class RecordingContinuation:
    """Record exactly the env-space actions returned to the rollout loop."""

    def __init__(self, continuation: Any) -> None:
        self.continuation = continuation
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.actions.clear()
        self.continuation.reset()

    def act(self, observation: Mapping[str, Any], *, task: str) -> np.ndarray:
        action = np.asarray(self.continuation.act(observation, task=task), dtype=np.float32).reshape(7)
        if not np.isfinite(action).all():
            raise ValueError("reference emitted a non-finite env action")
        self.actions.append(action.copy())
        return action


def checkpoint_identity(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    weights = sorted(path.glob("model*.safetensors"))
    return {
        "path": str(path.resolve()),
        "config_sha256": file_sha256(config),
        "weights": [{"name": item.name, "size": item.stat().st_size} for item in weights],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--action-tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    records = validate_protocol(protocol)
    start = max(0, args.start_index)
    end = args.end_index if args.end_index > 0 else len(records)
    records = records[start:end]
    if not records:
        raise ValueError("empty E3-V shard")

    from rase.collect.forked_rollout import (
        InProcessLeRobotContinuation,
        RolloutConfig,
        load_lerobot_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
    )
    from rase.collect.state_pool import StatePool

    pool = StatePool(Path(str(protocol["pool"])))
    policy_path = args.policy_path.resolve()
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
    cfg = RolloutConfig(n_action_steps=10, num_steps=10)
    output = args.output_dir.resolve()
    episode_dir = output / "episodes"
    trace_dir = output / "traces"
    episode_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy_id = str(protocol["policy_id"])
    policy_salt = int.from_bytes(hashlib.sha256(policy_id.encode()).digest()[:4], "big")
    rollouts_per_state = int(protocol["rollouts_per_state"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for state_index, record in enumerate(records, start=start):
        key = str(record["state_key"])
        loaded = pool.read_state(key, load_observations=False)
        root_checksums = pool.verify_state(key)
        root_sha = canonical_sha256(root_checksums)
        for rollout_index in range(rollouts_per_state):
            result_path = episode_dir / f"{key}__r{rollout_index:02d}.json"
            if result_path.is_file():
                row = read_json(result_path)
                if row.get("protocol_sha256") != protocol["protocol_sha256"]:
                    raise ValueError(f"stale E3-V result {result_path}")
                skipped = True
            else:
                seed = rollout_seed(key, 0, rollout_index, salt=policy_salt ^ 0x45335631)
                native = InProcessLeRobotContinuation(
                    bundle,
                    seed=seed,
                    capture=True,
                    capture_horizon=10,
                )
                recording = RecordingContinuation(native)
                result = run_one_forked_rollout(
                    pool,
                    key,
                    np.empty((0, 7), dtype=np.float32),
                    recording,
                    libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"),
                    config=cfg,
                )
                executed = np.asarray(recording.actions, dtype=np.float32).reshape(-1, 7)
                if len(executed) != result.continuation_steps:
                    raise RuntimeError("recorded action count does not match rollout result")
                events = list(native._capture.events)
                if not events:
                    raise RuntimeError("reference rollout captured no native inference events")
                event_chunks = np.stack([event.env_chunk for event in events]).astype(np.float32)
                initial_chunk = event_chunks[0]
                trace_path = trace_dir / f"{key}__r{rollout_index:02d}.npz"
                np.savez_compressed(
                    trace_path,
                    executed_actions=executed,
                    initial_env_chunk=initial_chunk,
                    inference_env_chunks=event_chunks,
                    root_proprio=np.asarray(loaded.proprio, dtype=np.float32),
                )
                row = {
                    "schema_version": RESULT_SCHEMA,
                    **record,
                    "rollout_index": rollout_index,
                    "rollout_seed": seed,
                    "policy_id": policy_id,
                    "reference_id": policy_id,
                    "result": {
                        **result.to_dict(),
                        "prefix_source": "direct_exact_root",
                        "prefix_steps": 0,
                        "outcome_semantics": "exact_root_reference_rollout_to_true_terminal",
                    },
                    "policy_metrics": native.metrics(),
                    "trace_path": str(trace_path),
                    "trace_sha256": file_sha256(trace_path),
                    "executed_action_steps": len(executed),
                    "inference_events": len(events),
                    "initial_chunk_sha256": hashlib.sha256(initial_chunk.tobytes()).hexdigest(),
                    "event_ids": [event.inference_event_id for event in events],
                    "root_bundle_sha256": root_sha,
                    "protocol_sha256": protocol["protocol_sha256"],
                }
                write_json(result_path, row)
                skipped = False
            rows.append(row)
            print(
                f"E3V state={state_index + 1}/{start + len(records)} rollout={rollout_index + 1}/{rollouts_per_state} "
                f"success={row['result']['success']} skipped={skipped}",
                flush=True,
            )

    summary = {
        "schema_version": "rase-e3v-reference-collection/v1",
        "status": "complete",
        "scientific_scope": protocol["scientific_scope"],
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "policy": checkpoint_identity(policy_path),
        "n_states": len(records),
        "n_rollouts": len(rows),
        "successes": sum(bool(row["result"]["success"]) for row in rows),
        "elapsed_wall_s": time.perf_counter() - started,
        "per_rollout": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("n_states", "n_rollouts", "successes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

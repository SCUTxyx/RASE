#!/usr/bin/env python3
"""Resume-safe direct evaluation for the frozen G2a Pi0Fast clean protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_SCHEMA = "rase-g2a-pi0fast-clean-direct-protocol/v1"
RESULT_SCHEMA = "rase-g2a-pi0fast-clean-direct-result/v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_protocol(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported G2a protocol schema")
    if protocol.get("status") != "frozen" or protocol.get("selection_uses_outcomes") is not False:
        raise ValueError("G2a protocol must be frozen and outcome-blind")
    if protocol.get("libero_flavor") != "clean" or protocol.get("suite") != "libero_10":
        raise ValueError("G2a requires official clean libero_10")
    records = [dict(row) for row in protocol.get("records", [])]
    expected = int(protocol.get("n_tasks", -1)) * int(protocol.get("episodes_per_task", -1))
    if expected != 80 or len(records) != expected or int(protocol.get("n_episodes", -1)) != expected:
        raise ValueError("G2a requires exactly 10 tasks x 8 episodes = 80")
    episode_ids = [str(row.get("episode_id")) for row in records]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate G2a episode_id")
    expected_pairs = {(task, init) for task in range(1, 11) for init in range(8)}
    observed_pairs = {
        (int(row.get("clean_task_index", -1)), int(row.get("init_state_id", -1)))
        for row in records
    }
    if observed_pairs != expected_pairs:
        raise ValueError("G2a task/init-state grid is incomplete")
    if canonical_sha256(records) != protocol.get("records_sha256"):
        raise ValueError("G2a records hash mismatch")
    no_hash = dict(protocol)
    claimed = no_hash.pop("protocol_sha256", None)
    if canonical_sha256(no_hash) != claimed:
        raise ValueError("G2a protocol hash mismatch")
    return records


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def task_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 2026082002, replicates: int = 10000
) -> list[float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(float(bool(row["success"])))
    task_ids = sorted(by_task)
    if not task_ids:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        values[index] = float(np.mean([value for task in sampled for value in by_task[str(task)]]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def gate_decision(rate: float, gate: Mapping[str, Any]) -> str:
    lower, upper = (float(value) for value in gate["long_pair_eligible_interval"])
    if rate < lower:
        return str(gate["below_interval_decision"])
    if rate > upper:
        return str(gate["above_interval_decision"])
    return str(gate["inside_interval_decision"])


def summarize(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], *, formal: bool
) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    n = len(rows)
    rate = successes / n if n else float("nan")
    task_rows: list[dict[str, Any]] = []
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        subset = [row for row in rows if str(row["task_id"]) == task_id]
        task_successes = sum(bool(row["success"]) for row in subset)
        task_rows.append(
            {
                "task_id": task_id,
                "episodes": len(subset),
                "successes": task_successes,
                "success_rate": task_successes / len(subset),
            }
        )
    summary: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if formal else "diagnostic_smoke",
        "scientific_scope": protocol["scientific_scope"],
        "protocol_sha256": protocol["protocol_sha256"],
        "episodes": n,
        "tasks": len(task_rows),
        "successes": successes,
        "success_rate": rate,
        "wilson_95_interval": wilson_interval(successes, n),
        "task_cluster_bootstrap_95_interval": task_cluster_bootstrap(rows),
        "task_results": task_rows,
        "policy_errors": sum(bool(row.get("policy_error")) for row in rows),
        "cost": {
            "environment_steps": sum(int(row["environment_steps"]) for row in rows),
            "wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "model_forward_calls": sum(int(row["policy_metrics"]["model_forward_calls"]) for row in rows),
            "action_select_seconds": sum(float(row["policy_metrics"]["action_select_elapsed_s"]) for row in rows),
        },
    }
    summary["gate_decision"] = gate_decision(rate, protocol["gate"]) if formal else "NOT_EVALUATED_SMOKE"
    return summary


def is_action_grammar_error(exc: BaseException) -> bool:
    return isinstance(exc, AssertionError) and "Token sequence does not start" in str(exc)


def evaluate_episode(bundle: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    from rase.collect.forked_rollout import InProcessLeRobotContinuation
    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.policy_step import as_batched_action, current_timestep, success_from_info
    from rase.collect.pool_candidates import observation_from_libero_env

    started = time.perf_counter()
    handle = make_libero_env_for_task(
        str(row["task_id"]),
        init_state_id=int(row["init_state_id"]),
        seed=int(row["environment_seed"]),
        observation_height=360,
        observation_width=360,
        libero_clean_root=os.environ.get("LIBERO_CLEAN_ROOT"),
        libero_flavor="clean",
    )
    continuation = InProcessLeRobotContinuation(bundle, seed=int(row["policy_seed"]))
    environment_steps = 0
    success = False
    stop_reason = "horizon"
    policy_error: dict[str, str] | None = None
    try:
        single = handle.vector_env.envs[0]
        observation = observation_from_libero_env(single)
        task = str(single.task_description)
        horizon = int(getattr(single, "_max_episode_steps", 600))
        continuation.reset_metrics()
        continuation.reset()
        while current_timestep(handle.control_env) < horizon:
            try:
                action = continuation.act(observation, task=task)
            except BaseException as exc:
                if not is_action_grammar_error(exc):
                    raise
                policy_error = {"type": "invalid_action_token_sequence", "message": str(exc)}
                stop_reason = "policy_inference_error"
                break
            observation, _, term, trunc, info = handle.vector_env.step(as_batched_action(action))
            environment_steps += 1
            terminal = bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0])
            if terminal:
                success = bool(success_from_info(info))
                stop_reason = "success" if success else "terminal_failure"
                break
        return {
            **dict(row),
            "schema_version": RESULT_SCHEMA,
            "success": success,
            "stop_reason": stop_reason,
            "policy_error": policy_error,
            "environment_steps": environment_steps,
            "final_timestep": current_timestep(handle.control_env),
            "wall_seconds": time.perf_counter() - started,
            "policy_metrics": continuation.metrics(),
        }
    finally:
        handle.close()


def checkpoint_identity(path: Path) -> dict[str, Any]:
    weights = sorted(path.glob("model*.safetensors"))
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256(path / "config.json"),
        "weights": [{"name": item.name, "size": item.stat().st_size} for item in weights],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    records = validate_protocol(protocol)
    if args.max_episodes < 0 or args.max_episodes > len(records):
        raise ValueError("--max-episodes must be in [0, 80]")
    selected = records[: args.max_episodes] if args.max_episodes else records
    formal = args.max_episodes == 0
    output_dir = args.output_dir.resolve()
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path.cwd().resolve()
    policy_path = (repo_root / str(protocol["policy_path"])).resolve()
    from rase.collect.forked_rollout import load_lerobot_policy_bundle

    bundle = load_lerobot_policy_bundle(
        policy_path,
        device=args.device,
        num_steps=int(protocol["num_steps"]),
        n_action_steps=int(protocol["n_action_steps"]),
        tokenizer_path=repo_root / str(protocol["tokenizer_path"]),
        action_tokenizer_path=repo_root / str(protocol["action_tokenizer_path"]),
        observation_height=int(protocol["observation_height"]),
        observation_width=int(protocol["observation_width"]),
    )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        target = episode_dir / f"{row['episode_id']}.json"
        if target.is_file():
            result = read_json(target)
            if result.get("protocol_sha256") != protocol["protocol_sha256"]:
                raise ValueError(f"stale episode result: {target}")
            skipped = True
        else:
            result = evaluate_episode(bundle, row)
            result["protocol_sha256"] = protocol["protocol_sha256"]
            result["checkpoint"] = checkpoint_identity(policy_path)
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            skipped = False
        rows.append(result)
        print(
            f"G2A episode={index + 1}/{len(selected)} id={row['episode_id']} "
            f"success={result['success']} steps={result['environment_steps']} skipped={skipped}",
            flush=True,
        )

    summary = summarize(rows, protocol, formal=formal)
    summary.update(
        {
            "protocol": str(protocol_path),
            "protocol_file_sha256": sha256(protocol_path),
            "checkpoint": checkpoint_identity(policy_path),
            "episode_result_hashes": {
                str(row["episode_id"]): sha256(episode_dir / f"{row['episode_id']}.json")
                for row in rows
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "successes", "episodes", "success_rate", "gate_decision")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

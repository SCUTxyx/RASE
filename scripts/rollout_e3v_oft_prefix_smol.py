#!/usr/bin/env python3
"""Evaluate short exact-root OFT prefixes followed by frozen SmolVLA.

This is the E3-Prefix mechanism gate: it tests whether a short correction
chunk can move a source-failure root into SmolVLA's basin of success before we
spend evidence or compute fitting a residual model to imitate that chunk.
"""

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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def teacher_prefix(actions: np.ndarray, horizon: int) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 7:
        raise ValueError(f"expected OFT artifact [1,T,7], got {array.shape}")
    if horizon < 1:
        raise ValueError("teacher horizon must be positive")
    if array.shape[1] < horizon:
        raise ValueError(f"trajectory has {array.shape[1]} steps, needs {horizon}")
    prefix = array[0, :horizon].copy()
    if not np.isfinite(prefix).all():
        raise ValueError("teacher prefix contains non-finite actions")
    return prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-horizon", type=int, action="append", required=True)
    parser.add_argument("--include-source-baseline", action="store_true")
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    horizons = sorted(set(args.teacher_horizon))
    if not horizons or horizons[0] < 1:
        raise ValueError("all teacher horizons must be positive")

    cfg = read_json(args.config.resolve())
    protocol = read_json(args.protocol.resolve())
    records = [dict(row) for row in protocol.get("records") or []]
    if not records or any(bool(row.get("source_success")) for row in records):
        raise ValueError("E3-Prefix protocol must be a non-empty source-failure cohort")
    explicit_adapter = cfg.get("adapter_config")
    if explicit_adapter is not None:
        if not isinstance(explicit_adapter, Mapping):
            raise TypeError("adapter_config must be a mapping")
        adapter = dict(explicit_adapter)
    else:
        legacy_adapter = cfg.get("adapter")
        adapter = dict(legacy_adapter) if isinstance(legacy_adapter, Mapping) else {}
    pool_path = Path(cfg.get("pool") or protocol["pool"])
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        RolloutConfig,
        load_smolvla_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
    )
    from rase.collect.state_pool import StatePool
    from rase.interventions.decision_context import strict_continue_suffix

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    trajectory_dir = args.trajectory_dir.resolve()
    temperature = float(adapter.get("continuation_temperature", 0.5))
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=temperature,
    )
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=rollout_cfg.num_steps,
        n_action_steps=rollout_cfg.n_action_steps,
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )

    output = args.output_dir.resolve()
    episode_dir = output / "episodes"
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output already exists: {output}")
    episode_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, frozen in enumerate(records):
        state_key = str(frozen["state_key"])
        loaded = pool.read_state(state_key, load_observations=False)
        artifact = load_artifact(trajectory_dir / f"{state_key}.npz")
        arms: list[tuple[str, np.ndarray]] = []
        if args.include_source_baseline:
            arms.append(
                (
                    "source_active_suffix",
                    np.asarray(strict_continue_suffix(loaded.controller_state), dtype=np.float32),
                )
            )
        for horizon in horizons:
            arms.append((f"oft_prefix_{horizon}", teacher_prefix(artifact.actions, horizon)))

        for arm_index, (arm, prefix) in enumerate(arms):
            target = episode_dir / f"{state_key}__{arm}.json"
            if target.is_file():
                row = read_json(target)
                skipped = True
            else:
                # Match the frozen Phase0G source continuation seed exactly.
                seed = rollout_seed(state_key, 0, 0, salt=0x52415345)
                continuation = InProcessSmolVLAContinuation(
                    bundle, temperature=temperature, seed=seed
                )
                result = run_one_forked_rollout(
                    pool,
                    state_key,
                    prefix,
                    continuation,
                    libero_plus_root=libero_plus_root,
                    config=rollout_cfg,
                )
                row = {
                    "schema_version": "rase-e3v-oft-prefix-smol-rollout/v1",
                    "protocol_sha256": protocol.get("protocol_sha256"),
                    "state_key": state_key,
                    "task_id": str(frozen["task_id"]),
                    "suite": str(frozen["suite"]),
                    "arm": arm,
                    "prefix_steps": int(len(prefix)),
                    "prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                    "continuation_seed": seed,
                    "continuation_temperature": temperature,
                    "result": result.to_dict(),
                    "policy_metrics": continuation.metrics(),
                }
                write_json(target, row)
                skipped = False
            rows.append(row)
            print(
                f"E3_PREFIX state={index + 1}/{len(records)} arm={arm} "
                f"success={row['result']['success']} skipped={skipped}",
                flush=True,
            )

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        successes = [row for row in selected if bool(row["result"]["success"])]
        by_arm[arm] = {
            "n": len(selected),
            "successes": len(successes),
            "success_rate": len(successes) / len(selected),
            "successful_tasks": len({row["task_id"] for row in successes}),
            "successful_suites": sorted({row["suite"] for row in successes}),
        }
    baseline = by_arm.get("source_active_suffix")
    checks: dict[str, bool] = {}
    if baseline is not None:
        checks["source_baseline_reproduces_frozen_failures"] = baseline["successes"] == 0
    best_teacher = max((value for key, value in by_arm.items() if key.startswith("oft_prefix_")), key=lambda x: x["successes"])
    checks.update(
        {
            "short_prefix_rescues_at_least_4_roots": best_teacher["successes"] >= 4,
            "short_prefix_rescues_at_least_2_suites": len(best_teacher["successful_suites"]) >= 2,
        }
    )
    summary = {
        "schema_version": "rase-e3v-oft-prefix-smol-summary/v1",
        "status": "complete",
        "scientific_scope": "development_only_short_correction_mechanism_gate",
        "protocol_sha256": protocol.get("protocol_sha256"),
        "n_states": len(records),
        "arms": by_arm,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "elapsed_wall_s": time.perf_counter() - started,
        "per_state_arm": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "arms": by_arm}, sort_keys=True))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

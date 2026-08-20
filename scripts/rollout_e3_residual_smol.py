#!/usr/bin/env python3
"""Run paired source vs frozen residual-chunk exact-root SmolVLA rollouts."""

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

from scripts.build_e3_residual_dataset import language_hash, load_rgb  # noqa: E402
from scripts.train_e3_residual_ridge import build_features, predict  # noqa: E402


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


def corrected_chunk(source: np.ndarray, delta: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float32)
    delta_array = np.asarray(delta, dtype=np.float32).reshape(source_array.shape)
    if source_array.ndim != 2 or source_array.shape[1] != 7:
        raise ValueError(f"source chunk must be [H,7], got {source_array.shape}")
    if not np.isfinite(delta_array).all():
        raise ValueError("predicted residual contains non-finite values")
    return np.clip(source_array + delta_array, -1.0, 1.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-chunk-mode",
        choices=("active-suffix", "replan"),
        default="active-suffix",
        help="use saved continue suffix or generate a fresh frozen-Smol chunk at the root",
    )
    parser.add_argument("--include-source-baseline", action="store_true")
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()

    cfg = read_json(args.config.resolve())
    keys_payload = read_json(args.state_keys_json.resolve())
    keys = [str(key) for key in keys_payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state-key artifact must contain unique non-empty keys")
    explicit = cfg.get("adapter_config")
    adapter = dict(explicit) if isinstance(explicit, Mapping) else dict(cfg.get("adapter") or {}) if isinstance(cfg.get("adapter"), Mapping) else {}
    pool_path = Path(cfg.get("pool") or keys_payload.get("pool") or "")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path

    with np.load(args.model.resolve(), allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    horizon = int(model["horizon"])
    action_dim = int(model["action_dim"])
    if action_dim != 7:
        raise ValueError("only 7D LIBERO actions are supported")
    variant = str(model["feature_variant"])

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        RolloutConfig,
        load_smolvla_policy_bundle,
        rollout_seed,
        run_one_forked_rollout,
        restore_pool_state,
    )
    from rase.collect.candidates import seed_everything
    from rase.collect.policy_step import capture_inference_event, clear_policy_queues
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool
    from rase.interventions.decision_context import strict_continue_suffix

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    manifest = pool.manifest()
    temperature = float(adapter.get("continuation_temperature", 0.5))
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=temperature,
    )
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(), device=str(adapter.get("device", "cuda")),
        num_steps=rollout_cfg.num_steps, n_action_steps=rollout_cfg.n_action_steps,
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )

    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output already exists: {output}")
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, state_key in enumerate(keys):
        loaded = pool.read_state(state_key, load_observations=False)
        seed = rollout_seed(state_key, 0, 0, salt=0x52415345)
        if args.source_chunk_mode == "active-suffix":
            source = np.asarray(strict_continue_suffix(loaded.controller_state), dtype=np.float32)
        else:
            restored = restore_pool_state(
                pool, state_key, libero_plus_root=libero_plus_root,
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                single = restored.handle.vector_env.envs[0]
                observation = observation_from_libero_env(single)
                seed_everything(seed)
                bundle["policy"].reset()
                clear_policy_queues(bundle["policy"])
                _first, event = capture_inference_event(
                    bundle, observation,
                    task=str(getattr(single, "task_description", "") or loaded.metadata.instruction),
                    boundary_step=int(loaded.metadata.step),
                    generation_seed=seed,
                    horizon=horizon,
                )
                source = np.asarray(event.env_chunk, dtype=np.float32)
            finally:
                restored.close()
        if source.shape != (horizon, 7):
            raise ValueError(f"{state_key}: source suffix {source.shape} != {(horizon, 7)}")
        state_path = pool.root / manifest["states"][state_key]["path"]
        one: dict[str, np.ndarray] = {
            "proprio": np.asarray(loaded.proprio, dtype=np.float32)[None, ...],
            "source_action": source[None, ...],
            "language_hash": language_hash(loaded.metadata.instruction)[None, ...],
        }
        if variant == "state_vision":
            one["agentview"] = load_rgb(state_path / "obs_agentview.png", 24)[None, ...]
            one["wrist"] = load_rgb(state_path / "obs_wrist.png", 24)[None, ...]
        delta = predict(model, build_features(one, variant))[0].reshape(horizon, 7)
        residual = corrected_chunk(source, delta)
        arms = [("residual_ridge", residual)]
        if args.include_source_baseline:
            arms.insert(0, ("source_active_suffix", source))
        for arm, chunk in arms:
            target = episode_dir / f"{state_key}__{arm}.json"
            if target.is_file():
                row = read_json(target)
                skipped = True
            else:
                continuation = InProcessSmolVLAContinuation(bundle, temperature=temperature, seed=seed)
                result = run_one_forked_rollout(
                    pool, state_key, chunk, continuation,
                    libero_plus_root=libero_plus_root, config=rollout_cfg,
                )
                row = {
                    "schema_version": "rase-e3-residual-smol-rollout/v1",
                    "state_key": state_key,
                    "task_id": loaded.metadata.task_id,
                    "suite": loaded.metadata.suite,
                    "arm": arm,
                    "source_chunk_mode": args.source_chunk_mode,
                    "continuation_seed": seed,
                    "chunk_sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                    "predicted_delta_abs_mean": float(np.mean(np.abs(delta))) if arm == "residual_ridge" else 0.0,
                    "result": result.to_dict(),
                    "policy_metrics": continuation.metrics(),
                }
                write_json(target, row)
                skipped = False
            rows.append(row)
            print(f"E3_RESIDUAL state={index+1}/{len(keys)} arm={arm} success={row['result']['success']} skipped={skipped}", flush=True)

    paired = {key: {} for key in keys}
    for row in rows:
        paired[row["state_key"]][row["arm"]] = bool(row["result"]["success"])
    counts = Counter()
    gain_tasks = set()
    for key, result in paired.items():
        source_ok = result.get("source_active_suffix", False)
        residual_ok = result.get("residual_ridge", False)
        label = "both" if source_ok and residual_ok else "source_only" if source_ok else "residual_only" if residual_ok else "neither"
        counts[label] += 1
        if label == "residual_only":
            gain_tasks.add(pool.read_state(key, load_observations=False).metadata.task_id)
    n = len(keys)
    source_rate = (counts["both"] + counts["source_only"]) / n
    residual_rate = (counts["both"] + counts["residual_only"]) / n
    oracle_rate = (counts["both"] + counts["source_only"] + counts["residual_only"]) / n
    checks = {
        "both_one_sided_wins_exist": counts["source_only"] > 0 and counts["residual_only"] > 0,
        "H_within_at_least_5pct": (counts["source_only"] + counts["residual_only"]) / n >= 0.05,
        "oracle_gain_at_least_5pp": oracle_rate - max(source_rate, residual_rate) >= 0.05,
        "gain_spans_at_least_2_tasks": len(gain_tasks) >= 2,
    }
    summary = {
        "schema_version": "rase-e3-residual-smol-summary/v1",
        "status": "complete",
        "scientific_scope": "eligibility_only; scope inherited from the supplied cohort",
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.resolve().read_bytes()).hexdigest(),
        "n_states": n,
        "classification_counts": dict(counts),
        "source_success_rate": source_rate,
        "residual_success_rate": residual_rate,
        "oracle_success_rate": oracle_rate,
        "oracle_gain_over_best_fixed": oracle_rate - max(source_rate, residual_rate),
        "H_within": (counts["source_only"] + counts["residual_only"]) / n,
        "residual_only_tasks": len(gain_tasks),
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "elapsed_wall_s": time.perf_counter() - started,
        "per_state_arm": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("decision", "classification_counts", "source_success_rate", "residual_success_rate", "oracle_gain_over_best_fixed", "H_within")}, sort_keys=True))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

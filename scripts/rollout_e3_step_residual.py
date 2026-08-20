#!/usr/bin/env python3
"""Paired exact-root closed-loop source-replan vs stepwise residual rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_e3_residual_dataset import language_hash  # noqa: E402
from scripts.collect_e3_step_demos import canonical_action, resize_rgb  # noqa: E402
from scripts.train_e3_step_residual import build_features, predict  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def corrected_action(source: np.ndarray, delta: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    if source.shape != (7,) or delta.shape != (7,):
        raise ValueError(f"expected source and delta shape (7,), got {source.shape} and {delta.shape}")
    if not 0.0 < scale <= 1.0:
        raise ValueError("residual scale must be in (0, 1]")
    return np.clip(source + float(scale) * delta, -1.0, 1.0).astype(np.float32)


def route_c_history(entries: list[Mapping[str, Any]], window: int = 8) -> np.ndarray:
    """Build Route-C history as proprio8, source7, progress1, executed7."""
    result = np.zeros((window, 23), dtype=np.float32)
    recent = entries[-window:]
    offset = window - len(recent)
    for index, entry in enumerate(recent):
        proprio = np.asarray(entry["proprio"], dtype=np.float32).reshape(-1)
        source = np.asarray(entry["source_action"], dtype=np.float32).reshape(-1)
        executed = np.asarray(entry["executed_action"], dtype=np.float32).reshape(-1)
        if proprio.shape != (8,) or source.shape != (7,) or executed.shape != (7,):
            raise ValueError("invalid Route-C history entry shape")
        result[offset + index] = np.concatenate(
            [proprio, source, [float(entry["progress"])], executed]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--model", type=Path, help="E3 ridge residual model")
    candidate.add_argument("--route-c-plugin", type=Path, help="legacy Route-C F0 residual plugin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-results-from",
        type=Path,
        help="reuse exact source-arm JSON files from a compatible completed paired run",
    )
    parser.add_argument(
        "--source-action-mode",
        choices=("chunked", "step-requery"),
        default="chunked",
    )
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.residual_scale <= 1.0:
        raise ValueError("--residual-scale must be in (0, 1]")
    cfg = read_json(args.config.resolve())
    key_payload = read_json(args.state_keys_json.resolve())
    keys = [str(key) for key in key_payload.get("state_keys") or []]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("state keys must be non-empty and unique")
    explicit = cfg.get("adapter_config")
    adapter = dict(explicit) if isinstance(explicit, Mapping) else {}
    pool_path = Path(cfg.get("pool") or key_payload.get("pool") or "")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero")
    tokenizer_path = Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    model = None
    plugin = None
    if args.model:
        with np.load(args.model.resolve(), allow_pickle=False) as archive:
            model = {key: archive[key] for key in archive.files}
        variant = str(model["feature_variant"])
        image_size = int(model["image_size"])
        language_dim = int(model["language_dim"])
        candidate_path = args.model.resolve()
        candidate_mode = "e3_ridge"
    else:
        from rase.recovery.residual_plugin import load_plugin
        plugin = load_plugin(str(args.route_c_plugin.resolve()))
        plugin.eval()
        variant = "route_c_f0"
        image_size = 0
        language_dim = 0
        candidate_path = args.route_c_plugin.resolve()
        candidate_mode = "route_c_f0_plugin"

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import seed_everything
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.oracle_continuation import raw_libero_to_oracle_arrays
    from rase.collect.policy_step import (
        as_batched_action, capture_inference_event, clear_policy_queues, success_from_info,
    )
    from rase.collect.pool_candidates import observation_from_libero_env
    from rase.collect.state_pool import StatePool

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT") or adapter.get("libero_plus_root")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(), device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)), n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path.resolve(),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    output = args.output_dir.resolve()
    if args.fresh_run and output.exists():
        raise SystemExit(f"fresh output already exists: {output}")
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    source_reuse_dir = args.source_results_from.resolve() if args.source_results_from else None
    source_reuse_summary_sha256 = None
    if source_reuse_dir:
        source_summary_path = source_reuse_dir / "summary.json"
        source_summary = read_json(source_summary_path)
        if source_summary.get("status") != "complete":
            raise ValueError("source reuse run must be complete")
        if source_summary.get("source_action_mode") != args.source_action_mode:
            raise ValueError("source reuse action mode mismatch")
        source_reuse_summary_sha256 = hashlib.sha256(source_summary_path.read_bytes()).hexdigest()
    started = time.perf_counter()
    for root_index, state_key in enumerate(keys):
        for arm in ("source_step_replan", "residual_stepwise"):
            target = episode_dir / f"{state_key}__{arm}.json"
            if target.is_file():
                row = read_json(target)
                skipped = True
            elif arm == "source_step_replan" and source_reuse_dir:
                reused_path = source_reuse_dir / "episodes" / target.name
                row = read_json(reused_path)
                if row.get("state_key") != state_key or row.get("arm") != arm:
                    raise ValueError(f"incompatible reused source row: {reused_path}")
                row = dict(row)
                row["reused_from"] = str(reused_path)
                write_json(target, row)
                skipped = True
            else:
                restored = restore_pool_state(
                    pool, state_key, libero_plus_root=libero_plus_root,
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                deltas = []
                plugin_history: list[dict[str, Any]] = []
                action_digest = hashlib.sha256()
                success = False
                stop_reason = "horizon"
                steps = 0
                try:
                    vector_env = restored.handle.vector_env
                    single = vector_env.envs[0]
                    task = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
                    horizon = min(args.max_steps, int(getattr(single, "_max_episode_steps", args.max_steps)))
                    instruction_feature = (
                        language_hash(task, language_dim)[None, ...] if model is not None else None
                    )
                    source_policy = InProcessSmolVLAContinuation(
                        bundle,
                        temperature=float(adapter.get("continuation_temperature", 0.5)),
                        seed=rollout_seed(state_key, 0, 0, salt=0x45335354),
                    )
                    source_policy.reset()
                    for step in range(horizon):
                        observation = observation_from_libero_env(single)
                        agentview, wrist, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)
                        if args.source_action_mode == "chunked":
                            source_action = source_policy.act(observation, task=task)
                        else:
                            seed = rollout_seed(state_key, 0, step, salt=0x45335354)
                            seed_everything(seed)
                            bundle["policy"].reset()
                            clear_policy_queues(bundle["policy"])
                            source_action, _event = capture_inference_event(
                                bundle, observation, task=task,
                                boundary_step=int(restored.loaded.metadata.step) + step,
                                generation_seed=seed, horizon=1,
                            )
                        source_action = canonical_action(source_action)
                        if arm == "residual_stepwise":
                            if model is not None:
                                one = {
                                    "proprio": np.asarray(proprio, dtype=np.float32)[None, ...],
                                    "source_action": source_action[None, ...],
                                    "language_hash": instruction_feature,
                                }
                                if variant == "state_vision":
                                    one["agentview"] = resize_rgb(agentview, image_size)[None, ...]
                                    one["wrist"] = resize_rgb(wrist, image_size)[None, ...]
                                delta = predict(model, build_features(one, variant))[0]
                            else:
                                delta = plugin.predict_delta(
                                    route_c_history(plugin_history, int(plugin.history_window)),
                                    np.zeros(int(plugin.obs_feature_dim), dtype=np.float32),
                                    source_action,
                                )
                            action = corrected_action(source_action, delta, args.residual_scale)
                            delta = float(args.residual_scale) * delta
                        else:
                            delta = np.zeros(7, dtype=np.float32)
                            action = np.asarray(source_action, dtype=np.float32)
                        deltas.append(delta)
                        proprio_array = np.asarray(proprio, dtype=np.float32).reshape(-1)
                        if proprio_array.shape != (8,):
                            raise ValueError(f"expected proprio shape (8,), got {proprio_array.shape}")
                        plugin_history.append(
                            {
                                "proprio": proprio_array,
                                "source_action": source_action,
                                "progress": float(np.linalg.norm(proprio_array[:3])),
                                "executed_action": action,
                            }
                        )
                        action_digest.update(np.ascontiguousarray(action).tobytes())
                        _obs, _reward, term, trunc, info = vector_env.step(as_batched_action(action))
                        steps += 1
                        terminated = bool(np.asarray(term).reshape(-1)[0])
                        truncated = bool(np.asarray(trunc).reshape(-1)[0])
                        if terminated or truncated:
                            success = success_from_info(info)
                            stop_reason = "success" if success else "terminal_failure"
                            break
                finally:
                    restored.close()
                row = {
                    "schema_version": "rase-e3-step-residual-rollout/v1",
                    "state_key": state_key,
                    "task_id": restored.loaded.metadata.task_id,
                    "suite": restored.loaded.metadata.suite,
                    "arm": arm,
                    "source_action_mode": args.source_action_mode,
                    "success": success,
                    "steps": steps,
                    "stop_reason": stop_reason,
                    "action_trace_sha256": action_digest.hexdigest(),
                    "predicted_delta_abs_mean": float(np.mean(np.abs(deltas))) if deltas else 0.0,
                }
                write_json(target, row)
                skipped = False
            rows.append(row)
            print(f"E3_STEP_ROLLOUT root={root_index+1}/{len(keys)} arm={arm} success={row['success']} steps={row['steps']} skipped={skipped}", flush=True)

    by_key = {key: {} for key in keys}
    task_by_key = {}
    for row in rows:
        by_key[row["state_key"]][row["arm"]] = bool(row["success"])
        task_by_key[row["state_key"]] = row["task_id"]
    counts = Counter()
    residual_only_tasks = set()
    for key, values in by_key.items():
        source_ok = values["source_step_replan"]
        residual_ok = values["residual_stepwise"]
        label = "both" if source_ok and residual_ok else "source_only" if source_ok else "residual_only" if residual_ok else "neither"
        counts[label] += 1
        if label == "residual_only":
            residual_only_tasks.add(task_by_key[key])
    n = len(keys)
    source_rate = (counts["both"] + counts["source_only"]) / n
    residual_rate = (counts["both"] + counts["residual_only"]) / n
    oracle_rate = (counts["both"] + counts["source_only"] + counts["residual_only"]) / n
    checks = {
        "continue_and_residual_only_both_exist": counts["source_only"] > 0 and counts["residual_only"] > 0,
        "H_within_at_least_5pct": (counts["source_only"] + counts["residual_only"]) / n >= 0.05,
        "oracle_gain_at_least_5pp": oracle_rate - max(source_rate, residual_rate) >= 0.05,
        "residual_gain_spans_2_tasks": len(residual_only_tasks) >= 2,
    }
    summary = {
        "schema_version": "rase-e3-step-residual-summary/v1",
        "status": "complete",
        "scientific_scope": str(key_payload.get("scientific_scope") or "inherited cohort scope"),
        "candidate_mode": candidate_mode,
        "candidate": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "source_action_mode": args.source_action_mode,
        "residual_scale": args.residual_scale,
        "source_reuse_summary_sha256": source_reuse_summary_sha256,
        "n_states": n,
        "classification_counts": dict(counts),
        "source_success_rate": source_rate,
        "residual_success_rate": residual_rate,
        "oracle_success_rate": oracle_rate,
        "oracle_gain_over_best_fixed": oracle_rate - max(source_rate, residual_rate),
        "H_within": (counts["source_only"] + counts["residual_only"]) / n,
        "residual_only_tasks": len(residual_only_tasks),
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

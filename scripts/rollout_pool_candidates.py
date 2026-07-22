#!/usr/bin/env python3
"""W3: fork-execute candidates, SmolVLA stochastic continuation, optional OFT verify."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    if value is None or value == "":
        return os.environ.get(env_name) if env_name else None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def _resolve_state_keys(cfg: dict[str, Any], pool, args) -> list[str]:
    if args.state_key:
        return list(args.state_key)
    sample = dict(cfg.get("sample") or {})
    explicit = list(sample.get("state_keys") or [])
    if explicit:
        return explicit
    strategy = str(sample.get("strategy", "explicit_or_w2"))
    if strategy == "stratified":
        from rase.collect.stratified_sample import sample_stratified_keys

        suite_horizons = sample.get("suite_horizons")
        return sample_stratified_keys(
            pool,
            per_cell=int(sample.get("per_cell", 2)),
            seed=int(sample.get("sample_seed", 0)),
            dims=tuple(sample.get("dims") or ("camera", "robot")),
            suites=tuple(sample.get("suites") or ("Spatial", "Object", "Goal", "Long")),
            levels=tuple(int(x) for x in (sample.get("levels") or (3, 4, 5))),
            min_remaining_steps=(
                int(sample["min_remaining_steps"])
                if sample.get("min_remaining_steps") is not None
                else None
            ),
            max_t0=(
                int(sample["max_t0"]) if sample.get("max_t0") is not None else None
            ),
            suite_horizons=(
                {str(k): int(v) for k, v in dict(suite_horizons).items()}
                if suite_horizons
                else None
            ),
        )
    # Default: reuse W2 pilot keys if summary exists.
    w2_summary = ROOT / "runs/ngc_w2_candidates_pilot/summary.json"
    if w2_summary.is_file():
        payload = json.loads(w2_summary.read_text(encoding="utf-8"))
        keys = list(payload.get("state_keys") or [])
        if keys:
            return keys
    from rase.collect.pool_candidates import sample_pool_keys

    return sample_pool_keys(pool, 2, int(sample.get("sample_seed", 0)))


def _suite_from_task_id(task_id: str) -> str:
    if task_id.startswith("libero_spatial"):
        return "libero_spatial"
    if task_id.startswith("libero_object"):
        return "libero_object"
    if task_id.startswith("libero_goal"):
        return "libero_goal"
    if task_id.startswith("libero_10"):
        return "libero_10"
    raise ValueError(f"cannot map task_id to suite: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "smolvla-primary", "oft-verify"),
        default=None,
    )
    parser.add_argument("--state-key", action="append", default=[])
    parser.add_argument("--suite", default=None, help="Filter states for oft-verify")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--force-new-run", action="store_true")
    parser.add_argument(
        "--continuation-temperature",
        type=float,
        default=None,
        help="Override adapter.continuation_temperature for SmolVLA continuation",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    mode = args.mode or str(cfg.get("mode", "smolvla-primary"))
    pool_root = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool/ngc_step1_scale200")
    if not pool_root.is_absolute():
        pool_root = (ROOT / pool_root).resolve()
    candidates_dir = Path(cfg.get("candidates_dir") or "runs/ngc_w2_candidates_pilot/candidates")
    if not candidates_dir.is_absolute():
        candidates_dir = (ROOT / candidates_dir).resolve()
    output_dir = Path(args.output_dir or cfg.get("output_dir") or f"runs/ngc_w3_{mode}")
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    if args.force_new_run and output_dir.exists():
        raise SystemExit(f"refusing --force-new-run on existing {output_dir}")

    adaptive = dict(cfg.get("adaptive") or {})
    adapter = dict(cfg.get("adapter") or {})
    candidates_cfg = dict(cfg.get("candidates") or {})
    scheduler_cfg = dict(cfg.get("scheduler") or {})
    oracle_cfg = dict(cfg.get("oracle") or {})
    k = int(candidates_cfg.get("k", 8))
    n_first = int(adaptive.get("first_stage_rollouts", 6))
    n_total = int(adaptive.get("total_rollouts", 20))
    threshold = float(adaptive.get("threshold", 0.5))
    alpha_first = float(adaptive.get("alpha_first", 0.01))
    alpha_final = float(adaptive.get("alpha_final", 0.04))
    sidedness = str(adaptive.get("sidedness", "one-sided"))
    protocol_version = str(
        adaptive.get("protocol_version", "wilson-onesided-alpha-spend-v1")
    )
    set_a_min = int(adaptive.get("set_a_min_good_candidates", 3))
    worker = args.worker or str(scheduler_cfg.get("worker", "w3-worker"))
    max_rollouts = args.max_rollouts
    if max_rollouts is None and mode == "smoke":
        max_rollouts = int((cfg.get("smoke") or {}).get("max_rollouts", 20))
    continuation_temperature = (
        float(args.continuation_temperature)
        if args.continuation_temperature is not None
        else float(adapter.get("continuation_temperature", 0.5))
    )
    adapter["continuation_temperature"] = continuation_temperature

    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.collect.candidates import load_artifact
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        RolloutConfig,
        evaluate_candidate,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.resumable_sampling import adaptive_sample_resumable
    from rase.collect.run_manifest import build_run_manifest, write_run_manifest
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.smolvla_candidate_policy import checkpoint_sha256
    from rase.collect.state_pool import StatePool
    from rase.collect.triage_report import summarize_run, write_json

    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()

    pool = StatePool(pool_root)
    state_keys = _resolve_state_keys(cfg, pool, args)
    if args.suite:
        filtered = []
        for key in state_keys:
            meta = pool.read_state(key, load_observations=False).metadata
            if _suite_from_task_id(meta.task_id) == args.suite:
                filtered.append(key)
        state_keys = filtered
        if not state_keys:
            raise SystemExit(f"no states match suite {args.suite}")

    policy_path = Path(
        _expand(adapter.get("policy_path"), "RASE_POLICY_PATH") or "ckpts/smolvla_libero"
    )
    if not policy_path.is_absolute():
        policy_path = (ROOT / policy_path).resolve()
    policy_hash = candidates_cfg.get("policy_hash") or checkpoint_sha256(policy_path)

    # OFT verify must never share the SmolVLA adaptive scheduler: both use
    # rollout_index=0 and would skip real oracle calls / overwrite records.
    if mode == "oft-verify":
        scheduler_root = output_dir / "scheduler"
    else:
        scheduler_root = Path(scheduler_cfg.get("root") or (output_dir / "scheduler"))
    if not scheduler_root.is_absolute():
        scheduler_root = (ROOT / scheduler_root).resolve()
    scheduler = DiskRolloutScheduler(
        scheduler_root,
        max_attempts=int(scheduler_cfg.get("max_attempts", 3)),
        lease_seconds=float(scheduler_cfg.get("lease_seconds", 3600)),
    )

    tokenizer_path = _expand(adapter.get("tokenizer_path"), "RASE_TOKENIZER_PATH")
    if tokenizer_path is None:
        default_tok = ROOT / "ckpts" / "SmolVLM2-500M-Instruct"
        if default_tok.is_dir():
            tokenizer_path = str(default_tok)

    oracle_info = {}
    oracle_client = None
    if mode == "oft-verify":
        from rase.oracle.client import OracleClient

        endpoint = args.endpoint or oracle_cfg.get("endpoint") or "tcp://127.0.0.1:5555"
        oracle_client = OracleClient(
            endpoint, timeout_ms=int(oracle_cfg.get("request_timeout_ms", 60_000))
        )
        oracle_info = oracle_client.model_info()
        if args.suite and oracle_info.get("suite") not in {None, args.suite}:
            raise SystemExit(
                f"oracle suite {oracle_info.get('suite')!r} != --suite {args.suite!r}"
            )

    resolved = {
        "mode": mode,
        "pool": str(pool_root),
        "candidates_dir": str(candidates_dir),
        "output_dir": str(output_dir),
        "state_keys": state_keys,
        "adaptive": adaptive,
        "adapter": {
            "continuation_temperature": continuation_temperature,
            "tokenizer_path": tokenizer_path,
        },
        "scheduler_root": str(scheduler_root),
        "policy_path": str(policy_path),
        "policy_hash": policy_hash,
        "oracle": oracle_info,
        "sample": dict(cfg.get("sample") or {}),
    }
    write_run_manifest(
        output_dir,
        build_run_manifest(
            repo_root=ROOT,
            resolved_config=resolved,
            pool_root=pool_root,
            candidates_dir=candidates_dir,
            policy_path=policy_path,
            policy_hash=str(policy_hash),
            protocol_version=protocol_version,
            oracle_model_info=oracle_info,
        ),
    )

    print(
        f"ROLLOUT_START mode={mode} n_states={len(state_keys)} "
        f"k={k} n_first={n_first} n_total={n_total} "
        f"cont_temp={continuation_temperature} out={output_dir}",
        flush=True,
    )

    policy_bundle = None
    if mode in {"smoke", "smolvla-primary"}:
        policy_bundle = load_smolvla_policy_bundle(
            policy_path,
            device=str(adapter.get("device", "cuda")),
            num_steps=int(adapter.get("num_steps", 10)),
            n_action_steps=int(adapter.get("n_action_steps", 10)),
            tokenizer_path=tokenizer_path,
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )

    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=continuation_temperature,
    )

    completed_budget = 0
    t0 = time.perf_counter()
    cross_oracle: dict[str, dict[str, Any]] = {}

    for state_key in state_keys:
        artifact_path = candidates_dir / f"{state_key}.npz"
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        artifact = load_artifact(artifact_path)
        if artifact.actions.shape[0] != k:
            raise ValueError(f"{artifact_path} expected K={k}, got {artifact.actions.shape}")
        meta = pool.read_state(state_key, load_observations=False).metadata
        print(
            f"STATE key={state_key} task={meta.task_id} dim={meta.perturb_dim} "
            f"level={meta.level}",
            flush=True,
        )

        if mode == "oft-verify":
            from rase.collect.oracle_continuation import OracleChunkContinuation

            assert oracle_client is not None
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=libero_plus_root,
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                outcomes = []
                for candidate in range(k):
                    key = RolloutKey(state_key, candidate, 0)
                    existing = scheduler.result(key)
                    if existing is not None:
                        outcomes.append(bool(existing["result"]["success"]))
                        continue
                    claim = scheduler.claim(key, worker)
                    if claim is None:
                        existing = scheduler.result(key)
                        if existing is None:
                            raise RuntimeError(f"cannot claim {key}")
                        outcomes.append(bool(existing["result"]["success"]))
                        continue
                    try:
                        continuation = OracleChunkContinuation(
                            oracle_client, instruction=meta.instruction
                        )
                        result = evaluate_candidate(
                            restored,
                            artifact.actions[candidate],
                            continuation,
                        )
                        payload = {
                            **result.to_dict(),
                            "oracle": "oft",
                            "model_info": oracle_info,
                        }
                        scheduler.complete(key, payload, worker=worker)
                        outcomes.append(bool(result.success))
                        completed_budget += 1
                        print(
                            f"OFT_VERIFY c={candidate} success={result.success} "
                            f"steps={result.env_steps}",
                            flush=True,
                        )
                    except Exception as exc:
                        scheduler.fail(key, repr(exc), worker=worker)
                        raise
                cross_oracle[state_key] = {
                    "successes": sum(outcomes),
                    "trials": len(outcomes),
                    "per_candidate": outcomes,
                }
            finally:
                restored.close()
            continue

        # SmolVLA primary / smoke
        assert policy_bundle is not None

        def run_one(cand_index: int, rollout_index: int) -> dict[str, Any]:
            nonlocal completed_budget
            seed = rollout_seed(state_key, cand_index, rollout_index)
            continuation = InProcessSmolVLAContinuation(
                policy_bundle,
                temperature=rollout_cfg.continuation_temperature,
                seed=seed,
            )
            restored = restore_pool_state(
                pool,
                state_key,
                libero_plus_root=libero_plus_root,
                observation_height=rollout_cfg.observation_height,
                observation_width=rollout_cfg.observation_width,
            )
            try:
                result = evaluate_candidate(
                    restored,
                    artifact.actions[cand_index],
                    continuation,
                )
            finally:
                restored.close()
            completed_budget += 1
            print(
                f"ROLLOUT state={state_key} c={cand_index} r={rollout_index} "
                f"success={result.success} steps={result.env_steps} "
                f"elapsed={result.elapsed_s:.2f}",
                flush=True,
            )
            return {
                **result.to_dict(),
                "oracle": "smolvla",
                "rollout_seed": seed,
                "continuation_temperature": rollout_cfg.continuation_temperature,
            }

        if mode == "smoke":
            # Timing gate: execute up to max_rollouts durable units, no full triage claim.
            for candidate in range(k):
                for rollout_index in range(n_total):
                    if max_rollouts is not None and completed_budget >= max_rollouts:
                        print("SMOKE_BUDGET_REACHED", flush=True)
                        break
                    key = RolloutKey(state_key, candidate, rollout_index)
                    if scheduler.is_complete(key):
                        continue
                    claim = scheduler.claim(key, worker)
                    if claim is None:
                        continue
                    try:
                        payload = run_one(candidate, rollout_index)
                        scheduler.complete(key, payload, worker=worker)
                    except Exception as exc:
                        scheduler.fail(key, repr(exc), worker=worker)
                        raise
                else:
                    continue
                break
            if max_rollouts is not None and completed_budget >= max_rollouts:
                break
            continue

        def make_rollout_fn(cand_index: int):
            def rollout_fn(rollout_index: int) -> dict[str, Any]:
                return run_one(cand_index, rollout_index)

            return rollout_fn

        for candidate in range(k):
            estimate = adaptive_sample_resumable(
                scheduler,
                state_key,
                candidate,
                worker,
                make_rollout_fn(candidate),
                threshold=threshold,
                n_first=n_first,
                n_total=n_total,
                protocol_version=protocol_version,
                alpha_first=alpha_first,
                alpha_final=alpha_final,
                sidedness=sidedness,
            )
            print(
                f"CANDIDATE_DONE state={state_key} c={candidate} "
                f"s={estimate.successes}/{estimate.trials} "
                f"L={estimate.lower:.3f} U={estimate.upper:.3f} "
                f"early={estimate.stopped_early}",
                flush=True,
            )

    summary = summarize_run(
        scheduler,
        state_keys,
        k=k,
        n_first=n_first,
        n_total=n_total,
        threshold=threshold,
        set_a_min_good=set_a_min,
        alpha_first=alpha_first,
        alpha_final=alpha_final,
        sidedness=sidedness,
        protocol_version=protocol_version,
        cross_oracle=cross_oracle or None,
    )
    summary["mode"] = mode
    summary["elapsed_wall_s"] = round(time.perf_counter() - t0, 3)
    summary["rollouts_this_process"] = completed_budget
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"SUMMARY {summary_path}", flush=True)
    print(json.dumps(summary.get("label_counts", {}), sort_keys=True), flush=True)
    if oracle_client is not None:
        oracle_client.close()
    print(f"ROLLOUT_DONE mode={mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

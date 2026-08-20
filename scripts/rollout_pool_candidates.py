#!/usr/bin/env python3
"""W3: fork-execute candidates, SmolVLA stochastic continuation, optional OFT verify."""

from __future__ import annotations

import argparse
import hashlib
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


def _state_keys_checksum(keys: list[str]) -> str:
    payload = json.dumps(
        keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_state_keys_json(path: Path) -> tuple[list[str], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if isinstance(payload, list):
        keys = [str(item) for item in payload]
        declared_checksum = None
    elif isinstance(payload, dict):
        keys = [str(item) for item in (payload.get("state_keys") or [])]
        declared_checksum = payload.get("state_keys_sha256")
    else:
        raise ValueError(f"{resolved} must contain a list or object")
    if not keys:
        raise ValueError(f"no state_keys in {resolved}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate state_keys in {resolved}")
    checksum = _state_keys_checksum(keys)
    if declared_checksum is not None and str(declared_checksum) != checksum:
        raise ValueError(
            f"state_keys_sha256 mismatch in {resolved}: "
            f"declared={declared_checksum} computed={checksum}"
        )
    return keys, {
        "source": str(resolved),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "state_keys_sha256": checksum,
        "n_states": len(keys),
    }


def _resolve_state_keys(
    cfg: dict[str, Any], pool, args
) -> tuple[list[str], dict[str, Any]]:
    if args.state_key:
        keys = list(args.state_key)
        return keys, {
            "source": "cli:--state-key",
            "artifact_sha256": None,
            "state_keys_sha256": _state_keys_checksum(keys),
            "n_states": len(keys),
        }
    if args.state_keys_json is not None:
        return _load_state_keys_json(args.state_keys_json)
    sample = dict(cfg.get("sample") or {})
    explicit = list(sample.get("state_keys") or [])
    if explicit:
        return explicit, {
            "source": "config:sample.state_keys",
            "artifact_sha256": None,
            "state_keys_sha256": _state_keys_checksum(explicit),
            "n_states": len(explicit),
        }
    strategy = str(sample.get("strategy", "explicit_or_w2"))
    if strategy == "stratified":
        from rase.collect.stratified_sample import sample_stratified_keys

        suite_horizons = sample.get("suite_horizons")
        outcomes = sample.get("episode_outcomes", sample.get("episode_outcome"))
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        excluded_keys = {str(key) for key in sample.get("excluded_keys") or []}
        excluded_paths = sample.get("excluded_keys_json") or []
        if isinstance(excluded_paths, (str, Path)):
            excluded_paths = [excluded_paths]
        for raw_path in excluded_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = ROOT / path
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload if isinstance(payload, list) else payload.get("state_keys") or []
            excluded_keys.update(str(key) for key in values)
        excluded_episode_keys = {
            str(key) for key in sample.get("excluded_episode_keys") or []
        }
        excluded_episode_paths = sample.get("excluded_episode_keys_json") or []
        if isinstance(excluded_episode_paths, (str, Path)):
            excluded_episode_paths = [excluded_episode_paths]
        for raw_path in excluded_episode_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = ROOT / path
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = (
                payload if isinstance(payload, list) else payload.get("state_keys") or []
            )
            excluded_episode_keys.update(str(key) for key in values)
        keys = sample_stratified_keys(
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
            strata=tuple(sample.get("strata") or ("suite", "dim")),
            t0_bins=sample.get("t0_bins"),
            selection=str(sample.get("selection", "earliest")),
            episode_outcomes=(
                tuple(str(value) for value in outcomes) if outcomes is not None else None
            ),
            excluded_keys=excluded_keys,
            excluded_episode_keys=excluded_episode_keys,
            distinct_episodes=bool(sample.get("distinct_episodes", False)),
        )
        return keys, {
            "source": "config:sample.stratified",
            "artifact_sha256": None,
            "state_keys_sha256": _state_keys_checksum(keys),
            "n_states": len(keys),
        }
    # Default: reuse W2 pilot keys if summary exists.
    w2_summary = ROOT / "runs/ngc_w2_candidates_pilot/summary.json"
    if w2_summary.is_file():
        payload = json.loads(w2_summary.read_text(encoding="utf-8"))
        keys = list(payload.get("state_keys") or [])
        if keys:
            return keys, {
                "source": str(w2_summary.resolve()),
                "artifact_sha256": hashlib.sha256(w2_summary.read_bytes()).hexdigest(),
                "state_keys_sha256": _state_keys_checksum(keys),
                "n_states": len(keys),
            }
    from rase.collect.pool_candidates import sample_pool_keys

    keys = sample_pool_keys(pool, 2, int(sample.get("sample_seed", 0)))
    return keys, {
        "source": "config:sample.fallback",
        "artifact_sha256": None,
        "state_keys_sha256": _state_keys_checksum(keys),
        "n_states": len(keys),
    }


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


def _continuation_seed(
    rollout_seed_fn,
    state_key: str,
    candidate_index: int,
    rollout_index: int,
    *,
    mode: str,
) -> int:
    """Derive the continuation seed under the preregistered pairing mode.

    ``common_root_rollout`` is a common-random-number design: candidates at the
    same root and repeat see the same continuation RNG stream.  Their first
    chunks still use independent generation seeds stored in the candidate
    artifact.  The legacy behavior remains the default for old configs.
    """
    if mode == "candidate_specific":
        seed_candidate = candidate_index
    elif mode == "common_root_rollout":
        seed_candidate = 0
    else:
        raise ValueError(
            "protocol.continuation_seed_mode must be candidate_specific or "
            f"common_root_rollout, got {mode!r}"
        )
    return int(rollout_seed_fn(state_key, seed_candidate, rollout_index))


def apply_screen_semantics(summary: dict[str, Any]) -> None:
    """Remove formal labels and expose only one-shot screen hit counts."""
    summary["formal_set_labels"] = False
    summary["screen_semantics"] = "one_rollout_per_candidate_hit_screen"
    summary["screen_warning"] = (
        "Screen outcomes locate non-zero candidate hits only. They are not "
        "Wilson A/B/C ground truth; freeze hits and run smolvla-primary confirm."
    )
    summary["diagnostic_label_counts"] = summary.pop("label_counts", {})
    state_hits = 0
    candidate_hits = 0
    for state in summary.get("per_state") or []:
        state["diagnostic_set_label"] = state.get("set_label")
        state["set_label"] = None
        hits = sum(
            int(candidate.get("successes", 0)) > 0
            for candidate in state.get("candidates") or []
        )
        state["screen_candidate_hits"] = hits
        candidate_hits += hits
        state_hits += hits > 0
    summary["screen_candidate_hits"] = candidate_hits
    summary["screen_state_hits"] = state_hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "smolvla-screen", "smolvla-primary", "oft-verify"),
        default=None,
    )
    parser.add_argument("--state-key", action="append", default=[])
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=None,
        help="Frozen JSON key artifact; preferred for formal runs",
    )
    parser.add_argument("--suite", default=None, help="Filter states for oft-verify")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="Override config candidates_dir",
    )
    parser.add_argument("--worker", default=None)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--fresh-run",
        action="store_true",
        help="Require that output-dir does not exist; default is safe resume",
    )
    run_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume the durable scheduler (the default)",
    )
    parser.add_argument(
        "--force-new-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--continuation-temperature",
        type=float,
        default=None,
        help="Override adapter.continuation_temperature for SmolVLA continuation",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-state-key", action="append", default=[])
    parser.add_argument(
        "--trace-all",
        action="store_true",
        help="Trace every selected state (otherwise repeat --trace-state-key)",
    )
    parser.add_argument(
        "--trace-outcomes",
        choices=("all", "success", "failure"),
        default="success",
    )
    parser.add_argument(
        "--trace-format",
        choices=("archive", "mp4", "both"),
        default="archive",
    )
    parser.add_argument("--trace-stride", type=int, default=5)
    parser.add_argument("--trace-max-frames", type=int, default=256)
    parser.add_argument("--trace-fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.trace_dir is not None and not (args.trace_all or args.trace_state_key):
        parser.error("--trace-dir requires --trace-all or at least one --trace-state-key")
    if args.state_key and args.state_keys_json is not None:
        parser.error("--state-key and --state-keys-json are mutually exclusive")
    if args.force_new_run and (args.fresh_run or args.resume):
        parser.error("--force-new-run cannot be combined with --fresh-run/--resume")

    cfg = _load_config(args.config.resolve())
    mode = args.mode or str(cfg.get("mode", "smolvla-primary"))
    pool_root = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool/ngc_step1_scale200")
    if not pool_root.is_absolute():
        pool_root = (ROOT / pool_root).resolve()
    candidates_dir = Path(
        args.candidates_dir
        or cfg.get("candidates_dir")
        or "runs/ngc_w2_candidates_pilot/candidates"
    )
    if not candidates_dir.is_absolute():
        candidates_dir = (ROOT / candidates_dir).resolve()
    if not candidates_dir.is_dir():
        raise SystemExit(
            f"candidates_dir does not exist: {candidates_dir}\n"
            "Run scripts/generate_pool_candidates.py first (it must write to "
            "config candidates_dir, not the W2 default)."
        )
    output_dir = Path(args.output_dir or cfg.get("output_dir") or f"runs/ngc_w3_{mode}")
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    fresh_run = bool(args.fresh_run or args.force_new_run)
    if fresh_run and output_dir.exists():
        raise SystemExit(f"fresh run requires a new output directory: {output_dir}")

    adaptive = dict(cfg.get("adaptive") or {})
    adapter = dict(cfg.get("adapter") or {})
    candidates_cfg = dict(cfg.get("candidates") or {})
    protocol_cfg = dict(cfg.get("protocol") or {})
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
    continuation_seed_mode = str(
        protocol_cfg.get("continuation_seed_mode", "candidate_specific")
    )
    if continuation_seed_mode not in {"candidate_specific", "common_root_rollout"}:
        parser.error(
            "protocol.continuation_seed_mode must be candidate_specific or "
            "common_root_rollout"
        )

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        RolloutConfig,
        evaluate_candidate,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
        run_one_forked_rollout,
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
    state_keys, state_keys_provenance = _resolve_state_keys(cfg, pool, args)
    if args.suite:
        filtered = []
        for key in state_keys:
            meta = pool.read_state(key, load_observations=False).metadata
            if _suite_from_task_id(meta.task_id) == args.suite:
                filtered.append(key)
        state_keys = filtered
        if not state_keys:
            raise SystemExit(f"no states match suite {args.suite}")
    selected_keys_checksum = _state_keys_checksum(state_keys)

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
        "protocol": {
            **protocol_cfg,
            "continuation_seed_mode": continuation_seed_mode,
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
        f"cont_temp={continuation_temperature} "
        f"cont_seed_mode={continuation_seed_mode} out={output_dir} "
        f"run_behavior={'fresh' if fresh_run else 'resume'} "
        f"keys_sha256={selected_keys_checksum}",
        flush=True,
    )

    policy_bundle = None
    if mode in {"smoke", "smolvla-screen", "smolvla-primary"}:
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
    trace_dir = args.trace_dir.resolve() if args.trace_dir is not None else None
    trace_state_keys = set(args.trace_state_key)

    def new_trace(state_key: str):
        if trace_dir is None:
            return None
        if not args.trace_all and state_key not in trace_state_keys:
            return None
        from rase.collect.rollout_trace import RolloutTraceRecorder

        return RolloutTraceRecorder(
            stride=args.trace_stride,
            max_frames=args.trace_max_frames,
        )

    def persist_trace(
        recorder,
        result,
        *,
        state_key: str,
        candidate: int,
        rollout: int,
        oracle: str,
    ) -> None:
        if recorder is None or trace_dir is None:
            return
        if args.trace_outcomes == "success" and not result.success:
            return
        if args.trace_outcomes == "failure" and result.success:
            return
        target = trace_dir / state_key / f"c{candidate}" / f"r{rollout}"
        metadata = {
            "state_key": state_key,
            "candidate_id": candidate,
            "rollout_index": rollout,
            "oracle": oracle,
            **result.to_dict(),
        }
        if args.trace_format in {"archive", "both"}:
            recorder.write_frame_archive(target, metadata=metadata)
        if args.trace_format in {"mp4", "both"}:
            recorder.write_mp4(target / "rollout.mp4", fps=args.trace_fps)

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
                    recorder = new_trace(state_key)
                    # Each candidate gets a fresh environment. A terminal
                    # rollout can mutate model state used by task fingerprinting.
                    result = run_one_forked_rollout(
                        pool,
                        state_key,
                        artifact.actions[candidate],
                        continuation,
                        libero_plus_root=libero_plus_root,
                        config=rollout_cfg,
                        trace_callback=recorder,
                    )
                    persist_trace(
                        recorder,
                        result,
                        state_key=state_key,
                        candidate=candidate,
                        rollout=0,
                        oracle="oft",
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
            continue

        # SmolVLA primary / smoke
        assert policy_bundle is not None

        def run_one(cand_index: int, rollout_index: int) -> dict[str, Any]:
            nonlocal completed_budget
            seed = _continuation_seed(
                rollout_seed,
                state_key,
                cand_index,
                rollout_index,
                mode=continuation_seed_mode,
            )
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
                recorder = new_trace(state_key)
                result = evaluate_candidate(
                    restored,
                    artifact.actions[cand_index],
                    continuation,
                    trace_callback=recorder,
                )
                persist_trace(
                    recorder,
                    result,
                    state_key=state_key,
                    candidate=cand_index,
                    rollout=rollout_index,
                    oracle="smolvla",
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
                "continuation_seed_mode": continuation_seed_mode,
                "continuation_temperature": rollout_cfg.continuation_temperature,
            }

        if mode == "smolvla-screen":
            for candidate in range(k):
                key = RolloutKey(state_key, candidate, 0)
                if scheduler.is_complete(key):
                    continue
                claim = scheduler.claim(key, worker)
                if claim is None:
                    continue
                try:
                    payload = run_one(candidate, 0)
                    scheduler.complete(key, payload, worker=worker)
                except Exception as exc:
                    scheduler.fail(key, repr(exc), worker=worker)
                    raise
            continue

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
    summary["run_behavior"] = "fresh" if fresh_run else "resume"
    summary["continuation_seed_mode"] = continuation_seed_mode
    summary["state_keys_provenance"] = {
        **state_keys_provenance,
        "selected_state_keys_sha256": selected_keys_checksum,
        "selected_n_states": len(state_keys),
        "suite_filter": args.suite,
    }
    if mode == "oft-verify":
        summary["verification_semantics"] = "deterministic_one_shot"
        summary["verification_warning"] = (
            "One deterministic trial per candidate; Wilson Set A/B/C labels "
            "do not apply. Use cross_oracle candidate hits and portfolio coverage."
        )
    elif mode == "smolvla-screen":
        apply_screen_semantics(summary)
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

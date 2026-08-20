#!/usr/bin/env python3
"""Restore pool states and generate configurable-K SmolVLA candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    if value is None or value == "":
        if env_name:
            return os.environ.get(env_name)
        return None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def _state_keys_checksum(keys: list[str]) -> str:
    encoded = json.dumps(
        keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/candidates_w2_pilot.json",
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--state-key",
        action="append",
        default=[],
        help="State key to process (repeatable). If omitted, use --sample.",
    )
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=None,
        help="JSON with state_keys list (or {state_keys:[...]})",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=None,
        help="W3 YAML/JSON; use sample.state_keys when present",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--policy-hash", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--libero-plus-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        help="Require full model XML fingerprint (fails on some Plus initstate robots)",
    )
    parser.add_argument(
        "--min-endpoint-l2",
        type=float,
        default=None,
        help="Fail if mean pairwise endpoint L2 across states is below this",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if cfg_path.suffix in {".yaml", ".yml"}:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = _load_json(cfg_path)
    candidates = cfg.get("candidates", {})
    adapter = cfg.get("adapter", {})

    pilot_keys: list[str] = []
    if args.pilot_config is not None:
        pilot_path = args.pilot_config.resolve()
        if pilot_path.suffix in {".yaml", ".yml"}:
            import yaml

            pilot_cfg = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))
        else:
            pilot_cfg = _load_json(pilot_path)
        pilot_keys = list((pilot_cfg.get("sample") or {}).get("state_keys") or [])
        if not pilot_keys:
            raise SystemExit(f"no sample.state_keys in {pilot_path}")
        if args.pool is None and pilot_cfg.get("pool"):
            args.pool = Path(str(pilot_cfg["pool"]))
        if args.output_dir is None and pilot_cfg.get("candidates_dir"):
            args.output_dir = Path(str(pilot_cfg["candidates_dir"]))

    pool_root = Path(
        _expand(args.pool or cfg.get("pool"), "RASE_POOL_ROOT")
        or "pool/ngc_step1_scale200"
    ).resolve()
    # W3+ YAML uses top-level candidates_dir; W2 JSON used candidates.output_dir.
    raw_out = (
        args.output_dir
        or cfg.get("candidates_dir")
        or candidates.get("output_dir")
        or "runs/ngc_w2_candidates_pilot/candidates"
    )
    output_dir = Path(str(raw_out)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    policy_path = Path(
        _expand(args.policy_path or adapter.get("policy_path"), "RASE_POLICY_PATH")
        or "ckpts/smolvla_libero"
    ).resolve()
    tokenizer_path = _expand(adapter.get("tokenizer_path"), "RASE_TOKENIZER_PATH")
    device = args.device or adapter.get("device") or "cuda"
    temperature = float(
        args.temperature if args.temperature is not None else candidates.get("temperature", 0.7)
    )
    base_seed = int(
        args.base_seed if args.base_seed is not None else candidates.get("base_seed", 17072026)
    )
    chunk_length = int(candidates.get("chunk_length", 10))
    k = int(candidates.get("k", 8))
    if k < 2:
        raise SystemExit("candidates.k must be >= 2")
    raw_sample = cfg.get("sample", 2)
    default_sample_n = int(raw_sample) if isinstance(raw_sample, int) else 2
    sample_n = int(args.sample if args.sample is not None else default_sample_n)
    default_sample_seed = 0
    if isinstance(raw_sample, dict) and raw_sample.get("sample_seed") is not None:
        default_sample_seed = int(raw_sample["sample_seed"])
    elif cfg.get("sample_seed") is not None:
        default_sample_seed = int(cfg["sample_seed"])
    sample_seed = int(
        args.sample_seed if args.sample_seed is not None else default_sample_seed
    )
    min_endpoint = (
        args.min_endpoint_l2
        if args.min_endpoint_l2 is not None
        else candidates.get("min_mean_endpoint_l2")
    )
    libero_plus_root = _expand(
        args.libero_plus_root or adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT"
    )

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.pool_candidates import (
        diversity_summary,
        generate_for_state,
        sample_pool_keys,
        write_summary,
    )
    from rase.collect.smolvla_candidate_policy import (
        checkpoint_sha256,
        load_smolvla_candidate_policy,
    )
    from rase.collect.state_pool import StatePool

    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()

    policy_hash = args.policy_hash or candidates.get("policy_hash")
    if not policy_hash or str(policy_hash).startswith("REPLACE_"):
        policy_hash = checkpoint_sha256(policy_path)

    pool = StatePool(pool_root)
    keys = list(args.state_key)
    keys_provenance: dict[str, object] = {
        "source": "cli:--state-key",
        "artifact_sha256": None,
    }
    if not keys and args.state_keys_json is not None:
        keys_path = args.state_keys_json.resolve()
        if not keys_path.is_file():
            raise SystemExit(
                f"frozen state-key artifact missing: {keys_path}\n"
                "Sample or export keys first, e.g.\n"
                "  python scripts/sample_state_keys.py "
                "--config configs/ngc_w5_failure_frontier_screen.yaml "
                "--output runs/ngc_w5_failure_frontier_state_keys.json\n"
                "or for OFT-recovered smoke:\n"
                "  python scripts/export_dual_oracle_split_keys.py "
                "--dual-oracle runs/ngc_w4_adequate_dual_oracle_summary.json "
                "--split oft_only "
                "--output runs/ngc_w5_oft_recovered_state_keys.json"
            )
        raw_keys = keys_path.read_bytes()
        payload = json.loads(raw_keys)
        if isinstance(payload, list):
            keys = [str(x) for x in payload]
            declared_checksum = None
        else:
            keys = [str(x) for x in (payload.get("state_keys") or [])]
            declared_checksum = payload.get("state_keys_sha256")
        if not keys:
            raise SystemExit(f"no state_keys in {args.state_keys_json}")
        if len(set(keys)) != len(keys):
            raise SystemExit(f"duplicate state_keys in {args.state_keys_json}")
        computed_checksum = _state_keys_checksum(keys)
        if declared_checksum is not None and str(declared_checksum) != computed_checksum:
            raise SystemExit(
                f"state_keys_sha256 mismatch in {keys_path}: "
                f"declared={declared_checksum} computed={computed_checksum}"
            )
        keys_provenance = {
            "source": str(keys_path),
            "artifact_sha256": hashlib.sha256(raw_keys).hexdigest(),
        }
    if not keys and pilot_keys:
        keys = list(pilot_keys)
        keys_provenance = {
            "source": str(args.pilot_config.resolve()),
            "artifact_sha256": hashlib.sha256(
                args.pilot_config.resolve().read_bytes()
            ).hexdigest(),
        }
    # Prefer explicit / stratified sample block from W3+ YAML configs.
    if not keys:
        sample_block = cfg.get("sample")
        if isinstance(sample_block, dict):
            explicit = list(sample_block.get("state_keys") or [])
            if explicit:
                keys = [str(x) for x in explicit]
                keys_provenance = {
                    "source": "config:sample.state_keys",
                    "artifact_sha256": None,
                }
            elif str(sample_block.get("strategy", "")) == "stratified":
                from rase.collect.stratified_sample import sample_stratified_keys

                suite_horizons = sample_block.get("suite_horizons")
                outcomes = sample_block.get(
                    "episode_outcomes", sample_block.get("episode_outcome")
                )
                if isinstance(outcomes, str):
                    outcomes = [outcomes]
                excluded_keys = {
                    str(key) for key in sample_block.get("excluded_keys") or []
                }
                excluded_paths = sample_block.get("excluded_keys_json") or []
                if isinstance(excluded_paths, (str, Path)):
                    excluded_paths = [excluded_paths]
                for raw_path in excluded_paths:
                    path = Path(raw_path)
                    if not path.is_absolute():
                        path = ROOT / path
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    values = (
                        payload
                        if isinstance(payload, list)
                        else payload.get("state_keys") or []
                    )
                    excluded_keys.update(str(key) for key in values)
                excluded_episode_keys = {
                    str(key)
                    for key in sample_block.get("excluded_episode_keys") or []
                }
                excluded_episode_paths = (
                    sample_block.get("excluded_episode_keys_json") or []
                )
                if isinstance(excluded_episode_paths, (str, Path)):
                    excluded_episode_paths = [excluded_episode_paths]
                for raw_path in excluded_episode_paths:
                    path = Path(raw_path)
                    if not path.is_absolute():
                        path = ROOT / path
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    values = (
                        payload
                        if isinstance(payload, list)
                        else payload.get("state_keys") or []
                    )
                    excluded_episode_keys.update(str(key) for key in values)
                keys = sample_stratified_keys(
                    pool,
                    per_cell=int(sample_block.get("per_cell", 2)),
                    seed=int(sample_block.get("sample_seed", sample_seed)),
                    dims=tuple(sample_block.get("dims") or ("camera", "robot")),
                    suites=tuple(
                        sample_block.get("suites")
                        or ("Spatial", "Object", "Goal", "Long")
                    ),
                    levels=tuple(
                        int(x) for x in (sample_block.get("levels") or (3, 4, 5))
                    ),
                    min_remaining_steps=(
                        int(sample_block["min_remaining_steps"])
                        if sample_block.get("min_remaining_steps") is not None
                        else None
                    ),
                    max_t0=(
                        int(sample_block["max_t0"])
                        if sample_block.get("max_t0") is not None
                        else None
                    ),
                    suite_horizons=(
                        {str(k): int(v) for k, v in dict(suite_horizons).items()}
                        if suite_horizons
                        else None
                    ),
                    strata=tuple(sample_block.get("strata") or ("suite", "dim")),
                    t0_bins=sample_block.get("t0_bins"),
                    selection=str(sample_block.get("selection", "earliest")),
                    episode_outcomes=(
                        tuple(str(value) for value in outcomes)
                        if outcomes is not None
                        else None
                    ),
                    excluded_keys=excluded_keys,
                    excluded_episode_keys=excluded_episode_keys,
                    distinct_episodes=bool(
                        sample_block.get("distinct_episodes", False)
                    ),
                )
                keys_provenance = {
                    "source": "config:sample.stratified",
                    "artifact_sha256": None,
                }
    if not keys:
        # Legacy: cfg["sample"] may be an int count for W2 JSON configs.
        if isinstance(cfg.get("sample"), int):
            sample_n = int(cfg["sample"])
        keys = sample_pool_keys(pool, sample_n, sample_seed)
        keys_provenance = {
            "source": "config:sample.fallback",
            "artifact_sha256": None,
        }
    keys_checksum = _state_keys_checksum(keys)
    keys_provenance.update(
        {
            "state_keys_sha256": keys_checksum,
            "n_states": len(keys),
        }
    )

    print(
        f"CANDIDATES_START n={len(keys)} pool={pool.root} out={output_dir} "
        f"K={k} T={chunk_length} temp={temperature} device={device}",
        flush=True,
    )
    print(f"POLICY path={policy_path} hash={policy_hash}", flush=True)

    policy = load_smolvla_candidate_policy(
        policy_path,
        device=device,
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        chunk_length=chunk_length,
        tokenizer_path=tokenizer_path,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )

    results = []
    t0 = time.perf_counter()
    for key in keys:
        print(f"CANDIDATE_STATE key={key}", flush=True)
        result = generate_for_state(
            pool,
            key,
            policy,
            output_dir=output_dir,
            temperature=temperature,
            base_seed=base_seed,
            policy_hash=str(policy_hash),
            k=k,
            libero_plus_root=libero_plus_root,
            strict_fingerprint=args.strict_fingerprint,
            force=args.force,
            observation_height=int(adapter.get("observation_height", 360)),
            observation_width=int(adapter.get("observation_width", 360)),
        )
        div = result.artifact.metadata.diversity
        print(
            f"{'SKIP' if result.skipped else 'WRITE'} key={key} "
            f"shape={result.artifact.actions.shape} "
            f"endpoint_l2={div.mean_pairwise_endpoint_l2:.4f} "
            f"chunk_l2={div.mean_pairwise_chunk_l2:.4f} "
            f"path={result.path}",
            flush=True,
        )
        results.append(result)

    artifacts = [item.artifact for item in results]
    summary = {
        "pool": str(pool.root),
        "output_dir": str(output_dir),
        "policy_path": str(policy_path),
        "policy_hash": str(policy_hash),
        "temperature": temperature,
        "base_seed": base_seed,
        "chunk_length": chunk_length,
        "k": k,
        "state_keys": keys,
        "state_keys_provenance": keys_provenance,
        "state_keys_sha256": keys_checksum,
        "n_written": sum(1 for item in results if not item.skipped),
        "n_skipped": sum(1 for item in results if item.skipped),
        "diversity": diversity_summary(artifacts),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    # Write beside artifacts (not parent/) so runs/<name>/summary.json is not polluted.
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    write_summary(summary_path, summary)
    print(f"SUMMARY {summary_path}", flush=True)
    print(json.dumps(summary["diversity"], sort_keys=True), flush=True)

    if min_endpoint is not None and summary["diversity"]["mean_endpoint_l2"] < float(min_endpoint):
        print(
            f"CANDIDATES_FAIL diversity collapsed: mean_endpoint_l2="
            f"{summary['diversity']['mean_endpoint_l2']:.6f} < {min_endpoint}",
            flush=True,
        )
        return 2

    print(f"CANDIDATES_DONE n={len(keys)} written={summary['n_written']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

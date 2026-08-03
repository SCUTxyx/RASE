#!/usr/bin/env python3
"""Execute OFT prefix lengths, then hand control back to frozen SmolVLA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    if value in {None, ""}:
        return os.environ.get(env_name) if env_name else None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, action="append", default=[])
    parser.add_argument("--fresh-run", action="store_true")
    args = parser.parse_args()
    lengths = args.prefix_length or [0, 1, 4, 8]
    if lengths[0] != 0 or sorted(set(lengths)) != lengths:
        raise ValueError("prefix lengths must be sorted, unique, and start at zero")
    if args.fresh_run and args.output_dir.exists():
        raise SystemExit(f"fresh output exists: {args.output_dir}")

    cfg = _load(args.config.resolve())
    key_payload = _load(args.state_keys_json.resolve())
    keys = [str(value) for value in key_payload.get("state_keys") or []]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("frozen state keys must be non-empty and unique")
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    policy_path = Path(
        _expand((cfg.get("adapter") or {}).get("policy_path"), "RASE_POLICY_PATH")
        or "ckpts/smolvla_libero"
    )
    tokenizer_path = _expand(
        (cfg.get("adapter") or {}).get("tokenizer_path"), "RASE_TOKENIZER_PATH"
    )
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.candidates import load_artifact
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        evaluate_candidate,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.state_pool import StatePool

    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=tokenizer_path,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    scheduler = DiskRolloutScheduler(
        args.output_dir / "scheduler", max_attempts=3, lease_seconds=3600
    )
    worker = "oft-prefix-to-smol"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for state_key in keys:
        artifact = load_artifact(args.candidates_dir / f"{state_key}.npz")
        if artifact.actions.shape[0] != 1 or artifact.actions.shape[2] != 7:
            raise ValueError(f"expected one OFT chunk for {state_key}")
        if max(lengths) > artifact.actions.shape[1]:
            raise ValueError(
                f"OFT chunk for {state_key} has T={artifact.actions.shape[1]}, "
                f"need {max(lengths)}"
            )
        arm_results = []
        shared_seed = rollout_seed(state_key, 0, 0, salt=2_026_080_303)
        for arm_index, prefix_length in enumerate(lengths):
            key = RolloutKey(state_key, arm_index, 0)
            existing = scheduler.result(key)
            if existing is None:
                claim = scheduler.claim(key, worker)
                if claim is None:
                    raise RuntimeError(f"could not claim {key}")
                restored = restore_pool_state(
                    pool,
                    state_key,
                    libero_plus_root=libero_plus_root,
                    observation_height=int(adapter.get("observation_height", 360)),
                    observation_width=int(adapter.get("observation_width", 360)),
                )
                try:
                    continuation = InProcessSmolVLAContinuation(
                        bundle,
                        temperature=float(adapter.get("continuation_temperature", 0.5)),
                        seed=shared_seed,
                    )
                    result = evaluate_candidate(
                        restored,
                        artifact.actions[0, :prefix_length],
                        continuation,
                    )
                    payload = {
                        **result.to_dict(),
                        "prefix_length": prefix_length,
                        "continuation": "smolvla",
                        "continuation_seed": shared_seed,
                        "prefix_policy_hash": artifact.metadata.policy_hash,
                    }
                    scheduler.complete(key, payload, worker=worker)
                    existing = scheduler.result(key)
                    print(
                        f"PREFIX_TRANSFER state={state_key} h={prefix_length} "
                        f"success={result.success} steps={result.env_steps}",
                        flush=True,
                    )
                except Exception as exc:
                    scheduler.fail(key, repr(exc), worker=worker)
                    raise
                finally:
                    restored.close()
            assert existing is not None
            arm_results.append(dict(existing["result"]))
        meta = pool.read_state(state_key, load_observations=False).metadata
        records.append(
            {
                "state_key": state_key,
                "task_id": meta.task_id,
                "suite": meta.suite,
                "perturbation_dimension": meta.perturb_dim,
                "perturbation_level": meta.level,
                "arms": arm_results,
            }
        )

    successes = {
        str(length): sum(bool(row["arms"][index]["success"]) for row in records)
        for index, length in enumerate(lengths)
    }
    base = [bool(row["arms"][0]["success"]) for row in records]
    prefix_oracle = [
        any(bool(arm["success"]) for arm in row["arms"][1:]) for row in records
    ]
    summary = {
        "schema_version": "rase-oft-prefix-to-smol/v1",
        "status": "complete",
        "n_states": len(records),
        "prefix_lengths": lengths,
        "successes_by_prefix_length": successes,
        "base_successes": sum(base),
        "prefix_oracle_successes": sum(prefix_oracle),
        "prefix_rescues": sum((not b) and p for b, p in zip(base, prefix_oracle, strict=True)),
        "per_state": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    compact = {
        key: summary[key]
        for key in ("base_successes", "prefix_oracle_successes", "prefix_rescues")
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

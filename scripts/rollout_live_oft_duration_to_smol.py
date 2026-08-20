#!/usr/bin/env python3
"""Live closed-loop OFT recovery for h steps, then same-seed SmolVLA handback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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


class LiveOFTThenSmolContinuation:
    """Query OFT live for ``prefix_length`` env steps, then hand back to SmolVLA."""

    def __init__(
        self,
        oft: Any,
        smol: Any,
        *,
        prefix_length: int,
    ) -> None:
        self.oft = oft
        self.smol = smol
        self.prefix_length = int(prefix_length)
        self.steps = 0
        self.oft_action_select_calls = 0
        self.oft_action_select_elapsed_s = 0.0

    def bind_control_env(self, control_env: Any) -> None:
        if hasattr(self.oft, "bind_control_env"):
            self.oft.bind_control_env(control_env)
        if hasattr(self.smol, "bind_control_env"):
            self.smol.bind_control_env(control_env)

    def reset(self) -> None:
        self.steps = 0
        self.oft_action_select_calls = 0
        self.oft_action_select_elapsed_s = 0.0
        self.oft.reset()
        self.smol.reset()

    def act(self, observation: Any, *, task: str) -> Any:
        if self.steps < self.prefix_length:
            started = time.perf_counter()
            action = self.oft.act(observation, task=task)
            self.oft_action_select_elapsed_s += time.perf_counter() - started
            self.oft_action_select_calls += 1
            self.steps += 1
            return action
        self.steps += 1
        return self.smol.act(observation, task=task)

    def metrics(self) -> dict[str, float | int | str]:
        smol_metrics = self.smol.metrics() if hasattr(self.smol, "metrics") else {}
        return {
            "execution_mode": "live_closed_loop_oft_prefix",
            "prefix_length": self.prefix_length,
            "oft_action_select_calls": self.oft_action_select_calls,
            "oft_action_select_elapsed_s": self.oft_action_select_elapsed_s,
            **{f"smol_{key}": value for key, value in smol_metrics.items()},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, action="append", default=[])
    parser.add_argument("--include-persistent-oft", action="store_true")
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--split", default=None, help="Optional train/val/test filter")
    args = parser.parse_args()

    lengths = args.prefix_length or [0, 8, 16, 32, 64, 96, 128]
    if lengths[0] != 0 or sorted(set(lengths)) != lengths:
        raise ValueError("prefix lengths must be sorted, unique, and start at zero")
    if args.fresh_run and args.output_dir.exists():
        raise SystemExit(f"fresh output exists: {args.output_dir}")

    cfg = _load(args.config.resolve())
    key_payload = _load(args.state_keys_json.resolve())
    records = list(key_payload.get("records") or [])
    keys = [str(value) for value in key_payload.get("state_keys") or []]
    meta_by_key = {str(row["state_key"]): row for row in records}
    if not keys:
        raise ValueError("frozen state keys must be non-empty")

    from scripts.generate_oft_pool_candidates import _suite  # local helper

    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    policy_path = Path(
        _expand((cfg.get("adapter") or {}).get("policy_path"), "RASE_POLICY_PATH")
        or "ckpts/smolvla_libero"
    )
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    tokenizer_path = _expand(
        (cfg.get("adapter") or {}).get("tokenizer_path"), "RASE_TOKENIZER_PATH"
    )

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import (
        InProcessSmolVLAContinuation,
        evaluate_candidate,
        load_smolvla_policy_bundle,
        restore_pool_state,
        rollout_seed,
    )
    from rase.collect.oracle_continuation import OracleChunkContinuation
    from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool = StatePool(pool_path.resolve())

    selected = []
    for state_key in keys:
        meta = pool.read_state(state_key, load_observations=False).metadata
        if _suite(meta.task_id) != args.suite:
            continue
        row = meta_by_key.get(state_key, {})
        if args.split and str(row.get("split", "")) != args.split:
            continue
        selected.append(state_key)
    if not selected:
        print(json.dumps({"suite": args.suite, "n_states": 0, "skipped": True}))
        return 0

    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    if model_info.get("suite") not in {None, args.suite}:
        raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")

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
    worker = f"live-oft-duration-{args.suite}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arm_specs: list[tuple[str, int | None]] = [
        (f"h{length}", length) for length in lengths
    ]
    if args.include_persistent_oft:
        arm_specs.append(("persistent_oft", None))

    out_records: list[dict[str, Any]] = []
    for state_key in selected:
        loaded = pool.read_state(state_key, load_observations=False)
        meta = loaded.metadata
        shared_seed = rollout_seed(state_key, 0, 0, salt=2_026_080_403)
        arm_results = []
        persistent_success = None
        for arm_index, (arm_name, prefix_length) in enumerate(arm_specs):
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
                    oft = OracleChunkContinuation(
                        client, instruction=meta.instruction
                    )
                    if prefix_length is None:
                        continuation: Any = oft
                    else:
                        smol = InProcessSmolVLAContinuation(
                            bundle,
                            temperature=float(
                                adapter.get("continuation_temperature", 0.5)
                            ),
                            seed=shared_seed,
                        )
                        continuation = LiveOFTThenSmolContinuation(
                            oft, smol, prefix_length=prefix_length
                        )
                    result = evaluate_candidate(
                        restored,
                        np.zeros((0, 7), dtype=np.float32),
                        continuation,
                    )
                    payload = {
                        **result.to_dict(),
                        "arm_name": arm_name,
                        "prefix_length": prefix_length,
                        "continuation": (
                            "persistent_oft"
                            if prefix_length is None
                            else "live_oft_then_smol"
                        ),
                        "continuation_seed": shared_seed,
                        "execution_mode": "live_closed_loop",
                        "metrics": (
                            continuation.metrics()
                            if hasattr(continuation, "metrics")
                            else {}
                        ),
                    }
                    scheduler.complete(key, payload, worker=worker)
                    existing = scheduler.result(key)
                    print(
                        f"LIVE_DURATION state={state_key} arm={arm_name} "
                        f"success={result.success} steps={result.env_steps}",
                        flush=True,
                    )
                except Exception as exc:
                    scheduler.fail(key, repr(exc), worker=worker)
                    raise
                finally:
                    restored.close()
            assert existing is not None
            result_payload = dict(existing["result"])
            if prefix_length is None:
                persistent_success = bool(result_payload["success"])
            else:
                arm_results.append(result_payload)

        out_records.append(
            {
                "state_key": state_key,
                "task_id": meta.task_id,
                "suite": meta.suite,
                "perturbation_dimension": meta.perturb_dim,
                "perturbation_level": meta.level,
                "split": meta_by_key.get(state_key, {}).get("split"),
                "arms": arm_results,
                "persistent_oft_success": persistent_success,
                "direct_oft_success": persistent_success,
            }
        )

    summary = {
        "schema_version": "rase-live-oft-duration-to-smol/v1",
        "status": "complete",
        "suite": args.suite,
        "n_states": len(out_records),
        "prefix_lengths": lengths,
        "execution_mode": "live_closed_loop",
        "include_persistent_oft": bool(args.include_persistent_oft),
        "per_state": out_records,
    }
    (args.output_dir / f"summary_{args.suite}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"suite": args.suite, "n_states": len(out_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

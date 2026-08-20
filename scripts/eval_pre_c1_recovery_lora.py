#!/usr/bin/env python3
"""Evaluate PRE-C1 recovery LoRA: recovery gain vs clean retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    InProcessSmolVLAContinuation,
    RolloutConfig,
    load_smolvla_policy_bundle,
    run_one_forked_rollout,
)
from rase.collect.state_pool import StatePool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-recovery", type=int, default=0)
    parser.add_argument("--limit-retention", type=int, default=0)
    parser.add_argument(
        "--failure-rollout-dir",
        type=Path,
        default=None,
        help="If set, recovery eval only uses current_suffix failures from this PRE-C0 rollout dir.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    rows = _load_jsonl(args.dataset_jsonl.resolve())
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    val_eps = set(splits["val_episodes"])
    val_rows = [row for row in rows if str(row["episode_id"]) in val_eps]
    recovery_rows = [row for row in val_rows if not bool(row.get("clean_flag"))]
    retention_rows = [row for row in val_rows if bool(row.get("clean_flag"))]

    def _unique_by_state(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Eval is per restore state; multi-chunk rows must not multiply rollouts."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in items:
            key = str(row["state_key"])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    failure_keys: set[str] | None = None
    if args.failure_rollout_dir is not None:
        failure_keys = set()
        for path in sorted(args.failure_rollout_dir.resolve().glob("*.json")):
            if path.name == "run_manifest.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
                continue
            if bool(payload.get("family_success", {}).get("current_suffix")):
                continue
            failure_keys.add(str(payload["state_key"]))
        recovery_rows = [row for row in recovery_rows if str(row["state_key"]) in failure_keys]
        # If val episodes lack enough PRE-C0 failures, fall back to all failure keys in dataset.
        if len(_unique_by_state(recovery_rows)) < 4:
            recovery_rows = [
                row
                for row in rows
                if (not bool(row.get("clean_flag"))) and str(row["state_key"]) in failure_keys
            ]

    recovery_rows = _unique_by_state(recovery_rows)
    retention_rows = _unique_by_state(retention_rows)
    # If val has too few clean rows, fall back to any clean in dataset held by val episodes only.
    if args.limit_recovery:
        recovery_rows = recovery_rows[: args.limit_recovery]
    if args.limit_retention:
        retention_rows = retention_rows[: args.limit_retention]
    if not recovery_rows:
        recovery_rows = _unique_by_state(
            [row for row in rows if not row.get("clean_flag")]
        )[: max(1, args.limit_recovery or 4)]
    if not retention_rows:
        retention_rows = _unique_by_state(
            [row for row in rows if row.get("clean_flag")]
        )[: max(1, args.limit_retention or 4)]

    bundle = load_smolvla_policy_bundle(
        Path(adapter.get("policy_path") or "ckpts/smolvla_libero"),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=Path(adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    handle = load_lora_onto_policy(bundle["policy"], str(args.adapter_dir.resolve()))
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=float(adapter.get("continuation_temperature", 0.5)),
    )
    libero_plus_root = adapter.get("libero_plus_root")
    empty = np.empty((0, 7), dtype=np.float32)

    def eval_state(row: dict[str, Any], *, adapter_on: bool) -> bool:
        set_adapter_enabled(handle, adapter_on)
        bundle["policy"] = handle.policy
        continuation = InProcessSmolVLAContinuation(
            bundle,
            temperature=float(adapter.get("continuation_temperature", 0.5)),
            seed=2026080405,
        )
        result = run_one_forked_rollout(
            pool,
            str(row["state_key"]),
            empty,
            continuation,
            libero_plus_root=libero_plus_root,
            config=rollout_cfg,
        )
        return bool(result.success)

    recovery_out = []
    for row in recovery_rows:
        base = eval_state(row, adapter_on=False)
        adapted = eval_state(row, adapter_on=True)
        recovery_out.append(
            {
                "state_key": row["state_key"],
                "episode_id": row["episode_id"],
                "suite": row.get("suite"),
                "stage": row.get("stage"),
                "base_success": base,
                "adapted_success": adapted,
                "arm": "recovery",
            }
        )
        print(
            f"REC state={row['state_key']} base={base} adapted={adapted}",
            flush=True,
        )

    retention_out = []
    for row in retention_rows:
        base = eval_state(row, adapter_on=False)
        # Retention protocol: adapter_off at clean inference.
        adapted_off = eval_state(row, adapter_on=False)
        retention_out.append(
            {
                "state_key": row["state_key"],
                "episode_id": row["episode_id"],
                "suite": row.get("suite"),
                "stage": row.get("stage"),
                "base_success": base,
                "adapted_success": adapted_off,
                "arm": "retention_adapter_off",
            }
        )
        print(
            f"RET state={row['state_key']} base={base} adapted_off={adapted_off}",
            flush=True,
        )

    payload = {
        "schema_version": "rase-pre-c1-eval/v1",
        "adapter_dir": str(args.adapter_dir),
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "not_smolvla_flow_api_guidance": True,
        "recovery": recovery_out,
        "retention": retention_out,
    }
    _write(args.output.resolve(), payload)
    print(f"PRE_C1_EVAL_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

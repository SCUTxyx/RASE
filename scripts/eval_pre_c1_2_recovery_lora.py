#!/usr/bin/env python3
"""PRE-C1.2 eval: recovery at frozen same H; retention at n_action_steps=10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rase.adapt.pre_c1_2 import load_protocol_lock
from rase.adapt.pre_c1_2_eval import load_pre_c0_failure_keys
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    InProcessSmolVLAContinuation,
    RolloutConfig,
    load_smolvla_policy_bundle,
    run_one_forked_rollout,
)
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unique_by_state(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in items:
        key = str(row["state_key"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--failure-rollout-dir",
        type=Path,
        default=Path("runs/rase_pre_c0_same_policy_pilot48_v1"),
    )
    parser.add_argument("--limit-recovery", type=int, default=0)
    parser.add_argument("--limit-retention", type=int, default=0)
    parser.add_argument(
        "--secondary-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Optional secondary seeds (does not affect primary gate).",
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    recovery_eval = dict(lock["evaluation"]["recovery"])
    h = recovery_eval.get("selected_horizon") or lock["horizon_selection"]["fallback_horizon"]
    h = int(h)
    if recovery_eval.get("base_execution_horizon") not in (None, h):
        raise SystemExit("protocol base_execution_horizon must match selected_horizon")
    if recovery_eval.get("adapted_execution_horizon") not in (None, h):
        raise SystemExit("protocol adapted_execution_horizon must match selected_horizon")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    rows = _load_jsonl(args.dataset_jsonl.resolve())
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    val_eps = set(splits["val_episodes"])
    failure_keys = {
        str(r["state_key"]) for r in load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    }
    recovery_rows = _unique_by_state(
        [
            row
            for row in rows
            if (not bool(row.get("clean_flag"))) and str(row.get("state_key")) in failure_keys
        ]
    )
    retention_rows = _unique_by_state(
        [row for row in rows if bool(row.get("clean_flag")) and str(row["episode_id"]) in val_eps]
    )
    if not retention_rows:
        retention_rows = _unique_by_state([row for row in rows if bool(row.get("clean_flag"))])
    if args.limit_recovery:
        recovery_rows = recovery_rows[: args.limit_recovery]
    if args.limit_retention:
        retention_rows = retention_rows[: args.limit_retention]

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
    empty = np.empty((0, 7), dtype=np.float32)
    seed_primary = 2026080405

    def eval_recovery(state_key: str, *, adapter_on: bool, seed: int) -> bool:
        set_adapter_enabled(handle, adapter_on)
        bundle["policy"] = handle.policy
        continuation = RecedingHorizonSmolVLAContinuation(
            bundle,
            execution_horizon=h,
            temperature=float(adapter.get("continuation_temperature", 0.5)),
            seed=seed,
        )
        result = run_one_forked_rollout(
            pool,
            state_key,
            empty,
            continuation,
            libero_plus_root=adapter.get("libero_plus_root"),
            config=rollout_cfg,
        )
        return bool(result.success)

    def eval_retention(state_key: str, *, seed: int) -> bool:
        # Retention: locked n_action_steps=10 default continuation, adapter off.
        set_adapter_enabled(handle, False)
        bundle["policy"] = handle.policy
        continuation = InProcessSmolVLAContinuation(
            bundle,
            temperature=float(adapter.get("continuation_temperature", 0.5)),
            seed=seed,
        )
        result = run_one_forked_rollout(
            pool,
            state_key,
            empty,
            continuation,
            libero_plus_root=adapter.get("libero_plus_root"),
            config=rollout_cfg,
        )
        return bool(result.success)

    recovery_out = []
    for row in recovery_rows:
        base = eval_recovery(str(row["state_key"]), adapter_on=False, seed=seed_primary)
        adapted = eval_recovery(str(row["state_key"]), adapter_on=True, seed=seed_primary)
        recovery_out.append(
            {
                "state_key": row["state_key"],
                "episode_id": row.get("episode_id"),
                "suite": row.get("suite"),
                "stage": row.get("stage"),
                "base_success": base,
                "adapted_success": adapted,
                "execution_horizon": h,
                "comparator": "adapted_minus_base_same_horizon",
                "arm": "recovery",
                "seed": seed_primary,
            }
        )
        print(
            f"REC H={h} state={row['state_key']} base={base} adapted={adapted}",
            flush=True,
        )

    retention_out = []
    for row in retention_rows:
        base = eval_retention(str(row["state_key"]), seed=seed_primary)
        adapted_off = eval_retention(str(row["state_key"]), seed=seed_primary)
        retention_out.append(
            {
                "state_key": row["state_key"],
                "episode_id": row.get("episode_id"),
                "suite": row.get("suite"),
                "stage": row.get("stage"),
                "base_success": base,
                "adapted_success": adapted_off,
                "n_action_steps": 10,
                "arm": "retention_adapter_off",
                "seed": seed_primary,
            }
        )
        print(
            f"RET nas10 state={row['state_key']} base={base} adapted_off={adapted_off}",
            flush=True,
        )

    secondary = []
    for seed in args.secondary_seeds:
        for row in recovery_rows:
            base = eval_recovery(str(row["state_key"]), adapter_on=False, seed=int(seed))
            adapted = eval_recovery(str(row["state_key"]), adapter_on=True, seed=int(seed))
            secondary.append(
                {
                    "state_key": row["state_key"],
                    "seed": int(seed),
                    "base_success": base,
                    "adapted_success": adapted,
                    "execution_horizon": h,
                    "affects_gate_decision": False,
                }
            )

    payload = {
        "schema_version": "rase-pre-c1-2-eval/v1",
        "adapter_dir": str(args.adapter_dir),
        "recovery_execution_horizon": h,
        "retention_n_action_steps": 10,
        "comparator_recovery": "adapted_minus_base_same_horizon",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "gate_note": "9 trials → 8pp ≈ ≥1 extra success; discrete",
        "recovery": recovery_out,
        "retention": retention_out,
        "secondary_recovery": secondary,
    }
    _write(args.output.resolve(), payload)
    print(f"PRE_C1_2_EVAL_DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

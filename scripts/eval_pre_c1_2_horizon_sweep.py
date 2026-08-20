#!/usr/bin/env python3
"""PRE-C1.2 Phase 1: same-H base vs adapted receding-horizon sweep."""

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

from rase.adapt.pre_c1_2 import (
    check_receding_invariants,
    freeze_selected_horizon,
    load_protocol_lock,
    select_recovery_horizon,
)
from rase.adapt.pre_c1_2_eval import load_pre_c0_failure_keys, summarize_horizon_sweep
from rase.adapt.recovery_lora import load_lora_onto_policy, set_adapter_enabled
from rase.collect.forked_rollout import (
    RolloutConfig,
    load_smolvla_policy_bundle,
    run_one_forked_rollout,
)
from rase.collect.same_policy_corrective import RecedingHorizonSmolVLAContinuation
from rase.collect.state_pool import StatePool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _best_progress_from_info(info: dict[str, Any] | None) -> float:
    """Locked progress definition: env native task score when present."""

    if not info:
        return 0.0
    for key in ("task_score", "reward", "progress", "is_success"):
        if key in info:
            value = info[key]
            if isinstance(value, (bool, np.bool_)):
                return 1.0 if bool(value) else 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    final = info.get("final_info")
    if isinstance(final, dict):
        return _best_progress_from_info(final)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_protocol_lock.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_pre_c0_deviation_pilot24.json"),
    )
    parser.add_argument(
        "--failure-rollout-dir",
        type=Path,
        default=Path("runs/rase_pre_c0_same_policy_pilot48_v1"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("runs/rase_pre_c1_1_lora_train_v1/adapter_final"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/rase_pre_c1_2_horizon_sweep_v1.json"))
    parser.add_argument(
        "--freeze-protocol",
        action="store_true",
        help="Write selected_horizon into protocol lock after sweep.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--state-keys-json",
        type=Path,
        default=Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json"),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Override candidate horizons (default from protocol).",
    )
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    horizons = list(args.horizons or lock["evaluation"]["recovery"]["candidate_horizons"])
    inv_tol = int(lock["receding_horizon_invariants"].get("forward_call_tolerance", 1))
    sel_cfg = dict(lock["horizon_selection"])
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    adapter = dict(cfg.get("adapter_config") or {})
    pool = StatePool(Path(cfg.get("pool") or cfg["collection"]["output_dir"]).resolve())
    failures = load_pre_c0_failure_keys(args.failure_rollout_dir.resolve())
    if args.state_keys_json and args.state_keys_json.is_file():
        allowed = set(json.loads(args.state_keys_json.read_text(encoding="utf-8"))["state_keys"])
        failures = [row for row in failures if str(row["state_key"]) in allowed]
    if args.limit:
        failures = failures[: args.limit]
    if not failures:
        raise SystemExit("no PRE-C0 failure keys")

    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths

    ensure_libero_plus_paths(adapter.get("libero_plus_root"))
    _patch_lerobot_init_states()

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
    bundle["policy"] = handle.policy
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=float(adapter.get("continuation_temperature", 0.5)),
    )
    empty = np.empty((0, 7), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    invariant_failures = 0

    for h in horizons:
        for failure in failures:
            state_key = str(failure["state_key"])
            row: dict[str, Any] = {
                "horizon": int(h),
                "state_key": state_key,
                "episode_id": failure.get("episode_id"),
                "suite": failure.get("suite"),
                "stage": failure.get("stage"),
            }
            for arm_name, adapter_on in (("base", False), ("adapted", True)):
                set_adapter_enabled(handle, adapter_on)
                bundle["policy"] = handle.policy
                continuation = RecedingHorizonSmolVLAContinuation(
                    bundle,
                    execution_horizon=int(h),
                    temperature=float(adapter.get("continuation_temperature", 0.5)),
                    seed=2026080405,
                )
                result = run_one_forked_rollout(
                    pool,
                    state_key,
                    empty,
                    continuation,
                    libero_plus_root=adapter.get("libero_plus_root"),
                    config=rollout_cfg,
                )
                metrics = continuation.metrics()
                inv = check_receding_invariants(
                    env_steps=int(result.env_steps),
                    execution_horizon=int(h),
                    model_forward_calls=int(metrics.get("model_forward_calls", 0)),
                    cache_resets=int(metrics.get("cache_resets", 0)),
                    tolerance=inv_tol,
                )
                if not inv["passed"]:
                    invariant_failures += 1
                row[f"{arm_name}_success"] = bool(result.success)
                row[f"{arm_name}_env_steps"] = int(result.env_steps)
                row[f"{arm_name}_metrics"] = metrics
                row[f"{arm_name}_invariants"] = inv
                # Progress proxy: success → 1 else 0 (native score unavailable mid-rollout here).
                row[f"{arm_name}_best_progress"] = 1.0 if result.success else 0.0
            row["base_success"] = bool(row["base_success"])
            row["adapted_success"] = bool(row["adapted_success"])
            row["best_progress"] = float(row["adapted_best_progress"])
            row["first_divergence_step"] = None
            row["invariant_passed"] = bool(
                row["base_invariants"]["passed"] and row["adapted_invariants"]["passed"]
            )
            rows.append(row)
            print(
                f"H={h} state={state_key} base={row['base_success']} "
                f"adapted={row['adapted_success']} inv={row['invariant_passed']}",
                flush=True,
            )

    if invariant_failures:
        print(
            f"PRE_C1_2_HORIZON_INVARIANT_FAIL count={invariant_failures}",
            flush=True,
        )

    selection = select_recovery_horizon(
        rows,
        candidate_horizons=horizons,
        minimum_adapted_successes=int(sel_cfg.get("minimum_adapted_successes", 2)),
        require_positive_adapter_delta_vs_base=bool(
            sel_cfg.get("require_positive_adapter_delta_vs_base", True)
        ),
        fallback_horizon=int(sel_cfg.get("fallback_horizon", 2)),
    )
    payload = {
        "schema_version": "rase-pre-c1-2-horizon-sweep/v1",
        "adapter_dir": str(args.adapter_dir),
        "horizons": horizons,
        "n_failure_states": len(failures),
        "invariant_failures": invariant_failures,
        "sweep_valid": invariant_failures == 0,
        "summary": summarize_horizon_sweep(rows),
        "selection": selection,
        "rows": rows,
        "comparator": "adapted_minus_base_same_horizon",
        "not_runtime_oft": True,
        "progress_definition": "env_native_task_score_proxy_success",
    }
    _write(args.output.resolve(), payload)

    if args.freeze_protocol:
        if not payload["sweep_valid"]:
            raise SystemExit("refusing to freeze horizon: receding invariants failed")
        frozen = freeze_selected_horizon(
            args.protocol_lock,
            selected_horizon=int(selection["selected_horizon"]),
        )
        payload["frozen"] = {
            "selected_horizon": frozen["selected_horizon"],
            "protocol_sha256": frozen["protocol_sha256"],
            "protocol_path": frozen["protocol_path"],
        }
        _write(args.output.resolve(), payload)
        print(
            f"PRE_C1_2_HORIZON_FROZEN H={frozen['selected_horizon']} "
            f"sha256={frozen['protocol_sha256']}",
            flush=True,
        )

    print(
        f"PRE_C1_2_HORIZON_SWEEP_DONE selected={selection['selected_horizon']} "
        f"mode={selection['selection_mode']} output={args.output}",
        flush=True,
    )
    return 0 if payload["sweep_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Closed-loop privileged trust-region action refinement audit (Gate B2).

Refines natural action chunks with a numerical trust-region update, then executes
the refined chunk via FixedActionContinuation on the same snapshot.

This is NOT SmolVLA flow-API guidance. Naming:
``privileged trust-region action refinement``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rase.collect.forked_rollout import (
    FixedActionContinuation,
    RolloutConfig,
    load_smolvla_policy_bundle,
    run_one_forked_rollout,
)
from rase.collect.pre_c0 import NATURAL_FAMILIES, analyze_guided_headroom
from rase.collect.state_pool import StatePool
from rase.guidance import apply_guidance_update


SCHEMA_VERSION = "rase-pre-c0-privileged-refine-rollout/v1"


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _refine(actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    direction = np.zeros_like(actions)
    direction[:, :3] = 1.0
    result = apply_guidance_update(
        actions,
        direction,
        step_size=0.05,
        action_low=np.full(7, -1.0),
        action_high=np.full(7, 1.0),
        trust_region_radius=0.25,
        max_guidance_norm=0.2,
    )
    meta = {
        "used_fallback": result.used_fallback,
        "reason": result.reason,
        "raw_guidance_norm": result.raw_guidance_norm,
        "applied_guidance_norm": result.applied_guidance_norm,
        "update_norm": result.update_norm,
        "naming": "privileged trust-region action refinement",
        "not_smolvla_flow_api_guidance": True,
    }
    return np.asarray(result.actions, dtype=np.float32), meta


def _pick_source_arm(row: dict[str, Any], family: str) -> dict[str, Any]:
    arms = [arm for arm in row.get("arms") or [] if str(arm.get("family")) == family]
    if not arms:
        raise SystemExit(f"state {row.get('state_key')} missing family={family}")
    # Prefer a failing natural arm to refine; else first arm.
    failing = [arm for arm in arms if not bool(arm.get("success"))]
    return failing[0] if failing else arms[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--natural-rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--family", default="strict_resample")
    parser.add_argument("--limit", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _load(args.config.resolve())
    collection = dict(cfg.get("collection") or {})
    adapter = dict(cfg.get("adapter_config") or cfg.get("adapter") or {})
    pool_root = Path(cfg.get("pool") or collection["output_dir"]).resolve()
    output_dir = args.output_dir.resolve()
    if args.fresh_run and output_dir.exists():
        raise SystemExit(f"fresh output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    natural_rows: list[dict[str, Any]] = []
    for path in sorted(args.natural_rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        natural_rows.append(payload)
    if args.limit:
        natural_rows = natural_rows[: args.limit]
    if not natural_rows:
        raise SystemExit("no natural rollout rows")

    pool = StatePool(pool_root)
    policy_path = Path(adapter.get("policy_path") or "ckpts/smolvla_libero").resolve()
    tokenizer = Path(
        adapter.get("tokenizer_path") or "ckpts/SmolVLM2-500M-Instruct"
    ).resolve()
    temperature = float(adapter.get("continuation_temperature", 0.5))
    rollout_cfg = RolloutConfig(
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        num_steps=int(adapter.get("num_steps", 10)),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
        continuation_temperature=temperature,
    )
    # Bundle load keeps env restore parity with natural corrective generator.
    _ = load_smolvla_policy_bundle(
        policy_path,
        device=str(adapter.get("device", "cuda")),
        num_steps=rollout_cfg.num_steps,
        n_action_steps=rollout_cfg.n_action_steps,
        tokenizer_path=tokenizer,
        observation_height=rollout_cfg.observation_height,
        observation_width=rollout_cfg.observation_width,
    )
    libero_plus_root = adapter.get("libero_plus_root")

    guided_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(natural_rows):
        state_key = str(row["state_key"])
        target = output_dir / f"{ordinal:03d}_{row['stage']}_{state_key}.json"
        if args.resume and target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
            guided_rows.append(payload["guided_row"])
            print(f"SKIP state={state_key}", flush=True)
            continue

        source = _pick_source_arm(row, str(args.family))
        actions = np.asarray(source.get("action_tensor") or [], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
            raise SystemExit(f"invalid action tensor for {state_key}")
        refined, refine_meta = _refine(actions)
        # Execute refined chunk as the candidate prefix; zero continuation after.
        continuation = FixedActionContinuation(np.zeros((1, 7), dtype=np.float32))
        result = run_one_forked_rollout(
            pool,
            state_key,
            refined,
            continuation,
            libero_plus_root=libero_plus_root,
            config=rollout_cfg,
        )
        guided_row = {
            "state_key": state_key,
            "episode_id": row.get("episode_id"),
            "task_id": row.get("task_id"),
            "suite": row.get("suite"),
            "cell": row.get("cell"),
            "stage": row.get("stage"),
            "family_success": dict(row["family_success"]),
            "privileged_guidance": bool(result.success),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state_key": state_key,
            "source_arm": source.get("arm_name"),
            "source_family": args.family,
            "refine_meta": refine_meta,
            "success": bool(result.success),
            "result": result.to_dict(),
            "guided_row": guided_row,
            "naming": "privileged trust-region action refinement",
            "not_smolvla_flow_api_guidance": True,
        }
        _atomic_json(target, payload)
        guided_rows.append(guided_row)
        print(
            f"PRE_C0_B2_STATE_DONE ordinal={ordinal} state={state_key} "
            f"success={bool(result.success)}",
            flush=True,
        )

    audit = analyze_guided_headroom(guided_rows)
    audit["mode"] = "closed_loop_privileged_trust_region_refinement"
    audit["naming"] = "privileged trust-region action refinement"
    audit["not_smolvla_flow_api_guidance"] = True
    audit["selection_family"] = str(args.family)
    audit["per_state"] = guided_rows
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = {
        "schema_version": "rase-pre-c0-decision/v1",
        "decision": audit["decision"],
        "natural_same_policy_gate": "closed",
        "candidate_critic_gate": "closed",
        "guided_generation_gate": audit["guided_generation_gate"],
        "learned_recovery_critic_gate": audit["learned_recovery_critic_gate"],
        "frozen_same_policy_recovery": audit["frozen_same_policy_recovery"],
        "pre_a3_method_gate": "closed",
        "world_model_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "audit": str(args.audit_output),
        "guided_gain_pp": audit["guided_gain_pp"],
        "naming": "privileged trust-region action refinement",
        "not_smolvla_flow_api_guidance": True,
        "natural_families": list(NATURAL_FAMILIES),
    }
    args.decision_output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True))
    print(f"PRE_C0_B2_DONE audit={args.audit_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

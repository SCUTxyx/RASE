#!/usr/bin/env python3
"""Privileged upper-bound audit for PRE-C0 Gate B.

This script has two modes:

1. ``--from-natural-rollouts`` (default): offline matched-compute Best-of-K
   selection over already-executed natural candidates. Terminal success is used
   only as an absolute ranking ceiling for the same K candidates; it is not a
   deployable critic.

2. ``--signals-json``: score externally supplied privileged transition signals
   with ``rase.guidance`` and report Best-of-K selection / Gate B metrics.

Numerical flow guidance utilities are exercised against action tensors when
present, but this script does not claim SmolVLA internal-API gradient injection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rase.collect.pre_c0 import NATURAL_FAMILIES, analyze_guided_headroom
from rase.guidance import (
    TransitionSignals,
    apply_guidance_update,
    select_best_of_k,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_natural_rows(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        rows.append(payload)
    if not rows:
        raise SystemExit(f"no PRE-C0 natural rollout JSON under {rollout_dir}")
    return rows


def _signals_from_arm(arm: dict[str, Any]) -> TransitionSignals:
    """Map executed-arm diagnostics into privileged scoring inputs.

    Prefer explicit privileged fields when present; otherwise fall back to a
    coarse terminal-outcome ceiling that answers whether any of the K samples
    recover under perfect ranking.
    """

    if "privileged_signals" in arm:
        raw = dict(arm["privileged_signals"])
        return TransitionSignals(
            progress_delta=float(raw.get("progress_delta", 0.0)),
            grasp_stability=float(raw.get("grasp_stability", 0.0)),
            collision_harm=float(raw.get("collision_harm", 0.0)),
            irreversible=bool(raw.get("irreversible", False)),
        )
    success = bool(arm.get("success"))
    return TransitionSignals(
        progress_delta=1.0 if success else 0.0,
        grasp_stability=1.0 if success else 0.0,
        collision_harm=0.0,
        irreversible=False,
    )


def _best_of_k_success(arms: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    if len(arms) < k:
        raise ValueError(f"need at least k={k} arms, got {len(arms)}")
    selected_arms = arms[:k]
    transitions = [_signals_from_arm(arm) for arm in selected_arms]
    ids = [str(arm.get("arm_name") or f"cand_{index}") for index, arm in enumerate(selected_arms)]
    choice = select_best_of_k(transitions, k=k, candidate_ids=ids)
    chosen = selected_arms[choice.index]
    return {
        "k": k,
        "selected_index": choice.index,
        "selected_arm": choice.candidate_id,
        "selected_success": bool(chosen.get("success")),
        "selected_score_total": choice.score.total,
        "evaluated_count": choice.evaluated_count,
        "random_first_success": bool(selected_arms[0].get("success")),
        "any_success": any(bool(arm.get("success")) for arm in selected_arms),
    }


def _maybe_guidance_probe(arm: dict[str, Any]) -> dict[str, Any] | None:
    tensor = arm.get("action_tensor")
    if not tensor:
        return None
    actions = np.asarray(tensor, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
        return None
    if not np.all(np.isfinite(actions)):
        return {"used_fallback": True, "reason": "non_finite_action_tensor"}
    # Synthetic progress-seeking direction: small push on translation dims.
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
    return {
        "used_fallback": result.used_fallback,
        "reason": result.reason,
        "raw_guidance_norm": result.raw_guidance_norm,
        "applied_guidance_norm": result.applied_guidance_norm,
        "update_norm": result.update_norm,
        "note": "numerical trust-region probe only; not SmolVLA API injection",
    }


def _summarize_trust_region_probes(probes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(probes)
    fallback = sum(1 for probe in probes if probe.get("used_fallback"))
    update_norms = [
        float(probe["update_norm"])
        for probe in probes
        if probe.get("update_norm") is not None and not probe.get("used_fallback")
    ]
    applied_norms = [
        float(probe["applied_guidance_norm"])
        for probe in probes
        if probe.get("applied_guidance_norm") is not None and not probe.get("used_fallback")
    ]
    return {
        "schema_version": "rase-pre-c0-guidance-trust-region-audit/v1",
        "naming": "privileged trust-region action refinement",
        "not_smolvla_flow_api_guidance": True,
        "n_probes": n,
        "fallback_count": fallback,
        "fallback_rate": fallback / max(1, n),
        "update_norm_mean": float(np.mean(update_norms)) if update_norms else None,
        "update_norm_max": float(np.max(update_norms)) if update_norms else None,
        "applied_guidance_norm_mean": (
            float(np.mean(applied_norms)) if applied_norms else None
        ),
        "applied_guidance_norm_max": (
            float(np.max(applied_norms)) if applied_norms else None
        ),
        "note": (
            "Numerical trust-region probe only; closed-loop privileged refinement "
            "is a separate Gate B stage when the offline ceiling shows signal."
        ),
        "probes": probes,
    }


def audit_from_natural_rollouts(
    rows: list[dict[str, Any]],
    *,
    family: str,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    guided_rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for row in rows:
        arms = [dict(arm) for arm in row.get("arms") or [] if str(arm.get("family")) == family]
        if not arms:
            raise SystemExit(
                f"state {row.get('state_key')} missing family={family} arms for Gate B"
            )
        selection = _best_of_k_success(arms, k=k)
        probe = _maybe_guidance_probe(arms[selection["selected_index"]])
        if probe is not None:
            probes.append({"state_key": row["state_key"], **probe})
        guided_rows.append(
            {
                "state_key": row["state_key"],
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id"),
                "suite": row.get("suite"),
                "cell": row.get("cell"),
                "stage": row.get("stage"),
                "family_success": dict(row["family_success"]),
                # Absolute ranking ceiling on the same K candidates.
                "privileged_guidance": bool(selection["any_success"]),
                "matched_compute_selection": selection,
            }
        )
    audit = analyze_guided_headroom(guided_rows)
    audit["selection_family"] = family
    audit["selection_k"] = k
    audit["selection_note"] = (
        "Best-of-K ceiling over executed natural candidates using privileged "
        "signals when present, else terminal-success ranking ceiling."
    )
    audit["guidance_probe_count"] = len(probes)
    audit["guidance_probe_fallback_rate"] = (
        sum(1 for probe in probes if probe.get("used_fallback")) / max(1, len(probes))
    )
    audit["per_state"] = guided_rows
    trust_region = _summarize_trust_region_probes(probes)
    return guided_rows, audit, trust_region


def audit_from_signals(payload: dict[str, Any]) -> dict[str, Any]:
    states = list(payload.get("states") or [])
    if not states:
        raise SystemExit("signals JSON must contain non-empty 'states'")
    guided_rows: list[dict[str, Any]] = []
    for state in states:
        candidates = list(state.get("candidates") or [])
        k = int(state.get("k") or len(candidates))
        transitions = [
            TransitionSignals(
                progress_delta=float(item["progress_delta"]),
                grasp_stability=float(item["grasp_stability"]),
                collision_harm=float(item.get("collision_harm", 0.0)),
                irreversible=bool(item.get("irreversible", False)),
            )
            for item in candidates
        ]
        ids = [str(item.get("candidate_id") or f"{index:04d}") for index, item in enumerate(candidates)]
        choice = select_best_of_k(transitions, k=k, candidate_ids=ids)
        natural = dict(state.get("family_success") or {})
        if not set(NATURAL_FAMILIES).issubset(natural):
            raise SystemExit(f"state {state.get('state_key')} missing natural family_success")
        guided_rows.append(
            {
                "state_key": state["state_key"],
                "episode_id": state.get("episode_id"),
                "task_id": state.get("task_id"),
                "suite": state.get("suite"),
                "cell": state.get("cell"),
                "stage": state.get("stage"),
                "family_success": natural,
                "privileged_guidance": bool(candidates[choice.index].get("success", True)),
                "matched_compute_selection": {
                    "k": k,
                    "selected_index": choice.index,
                    "selected_arm": choice.candidate_id,
                    "selected_score_total": choice.score.total,
                },
            }
        )
    audit = analyze_guided_headroom(guided_rows)
    audit["per_state"] = guided_rows
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-rollout-dir", type=Path, default=None)
    parser.add_argument("--signals-json", type=Path, default=None)
    parser.add_argument("--family", default="strict_resample")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, default=None)
    parser.add_argument(
        "--trust-region-output",
        type=Path,
        default=None,
        help="Optional path for numerical trust-region probe audit JSON.",
    )
    args = parser.parse_args()

    trust_region: dict[str, Any] | None = None
    if args.signals_json is not None:
        payload = json.loads(args.signals_json.read_text(encoding="utf-8"))
        audit = audit_from_signals(payload)
    elif args.natural_rollout_dir is not None:
        rows = _load_natural_rows(args.natural_rollout_dir.resolve())
        _, audit, trust_region = audit_from_natural_rollouts(
            rows, family=str(args.family), k=int(args.k)
        )
    else:
        raise SystemExit("provide --natural-rollout-dir or --signals-json")

    _write(args.output, audit)
    if trust_region is not None:
        trust_path = args.trust_region_output or args.output.with_name(
            "guidance_trust_region_audit.json"
        )
        _write(trust_path, trust_region)
        audit["trust_region_audit"] = str(trust_path)
    decision = {
        "schema_version": "rase-pre-c0-decision/v1",
        "decision": audit["decision"],
        "natural_same_policy_gate": "closed",
        "candidate_critic_gate": "closed",
        "guided_generation_gate": audit["guided_generation_gate"],
        "learned_recovery_critic_gate": audit["learned_recovery_critic_gate"],
        "frozen_same_policy_recovery": audit["frozen_same_policy_recovery"],
        "pre_a3_method_gate": "closed",
        "pre_b_allowed": False,
        "world_model_gate": "closed",
        "hidden_test": "sealed_not_unblinded",
        "audit": str(args.output),
        "guided_gain_pp": audit["guided_gain_pp"],
        "naming": "privileged trust-region action refinement",
        "not_smolvla_flow_api_guidance": True,
    }
    decision_path = args.decision_output or args.output.with_name("guided_decision.json")
    _write(decision_path, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

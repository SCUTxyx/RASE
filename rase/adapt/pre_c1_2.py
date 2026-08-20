"""PRE-C1.2 protocol helpers: horizon selection, batch schedule, flow weighting."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROTOCOL_LOCK_VERSION = "rase-pre-c1-2-protocol-lock/v1"
ALLOWED_PHASES = {"PRE-C1.2"}
REQUIRED_LOCK_KEYS = (
    "schema_version",
    "phase",
    "method",
    "lora",
    "train",
    "gate",
    "dataset",
    "sealed",
    "evaluation",
    "receding_horizon_invariants",
    "horizon_selection",
    "batch_schedule",
    "recovery_source_weights",
    "loss",
    "dagger",
    "dagger_sources",
    "capacity",
)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_protocol_lock(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PRE-C1.2 protocol lock must be a mapping")
    errors = validate_protocol_lock(payload)
    if errors:
        raise ValueError(f"PRE-C1.2 protocol lock invalid: {errors}")
    return payload


def validate_protocol_lock(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_LOCK_KEYS:
        if key not in payload:
            errors.append(f"missing:{key}")
    if payload.get("schema_version") != PROTOCOL_LOCK_VERSION:
        errors.append("bad_schema_version")
    if payload.get("phase") not in ALLOWED_PHASES:
        errors.append("bad_phase")
    if payload.get("not_runtime_oft") is not True:
        errors.append("runtime_oft_must_be_forbidden")
    sealed = dict(payload.get("sealed") or {})
    if sealed.get("world_model_gate") != "closed":
        errors.append("world_model_must_stay_closed")
    if sealed.get("hidden_test24") != "sealed":
        errors.append("hidden_test_must_stay_sealed")
    gate = dict(payload.get("gate") or {})
    if float(gate.get("recovery_gain_pp", -1)) != 8.0:
        errors.append("recovery_threshold_must_stay_8")
    if float(gate.get("clean_retention_drop_pp", -1)) != 2.0:
        errors.append("retention_threshold_must_stay_2")
    eval_cfg = dict(payload.get("evaluation") or {})
    recovery = dict(eval_cfg.get("recovery") or {})
    if recovery.get("comparator") != "adapted_minus_base_same_horizon":
        errors.append("recovery_comparator_must_be_same_horizon")
    retention = dict(eval_cfg.get("clean_retention") or {})
    if int(retention.get("base_n_action_steps", -1)) != 10:
        errors.append("retention_must_lock_nas10")
    if int(retention.get("adapted_n_action_steps", -1)) != 10:
        errors.append("retention_adapted_must_lock_nas10")
    sched = dict(payload.get("batch_schedule") or {})
    if int(sched.get("cycle_length", 0)) != 10:
        errors.append("batch_cycle_must_be_10")
    if int(sched.get("recovery_batches", 0)) != 9 or int(sched.get("clean_batches", 0)) != 1:
        errors.append("batch_schedule_must_be_9_plus_1")
    loss = dict(payload.get("loss") or {})
    if loss.get("type") != "native_flow_matching":
        errors.append("loss_must_be_native_flow_matching")
    if loss.get("auxiliary_sampled_action_mse") is not False:
        errors.append("aux_action_mse_forbidden")
    if loss.get("normalize_weights_to_mean_one") is not True:
        errors.append("weights_must_normalize_mean_one")
    dagger = dict(payload.get("dagger") or {})
    if dagger.get("beta_unit") != "replan_boundary":
        errors.append("beta_unit_must_be_replan_boundary")
    if dagger.get("teacher_query_mode") != "forked_environment":
        errors.append("teacher_query_must_be_forked")
    return errors


def freeze_selected_horizon(
    lock_path: Path | str,
    *,
    selected_horizon: int,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write selected_horizon into protocol and return payload + hash."""

    path = Path(lock_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol lock must be a mapping")
    h = int(selected_horizon)
    evaluation = dict(payload.get("evaluation") or {})
    recovery = dict(evaluation.get("recovery") or {})
    recovery["selected_horizon"] = h
    recovery["base_execution_horizon"] = h
    recovery["adapted_execution_horizon"] = h
    evaluation["recovery"] = recovery
    payload["evaluation"] = evaluation
    sealed = dict(payload.get("sealed") or {})
    sealed["selected_horizon_frozen"] = True
    payload["sealed"] = sealed
    out = Path(output_path) if output_path is not None else path
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    digest = file_sha256(out)
    return {
        "selected_horizon": h,
        "protocol_path": str(out),
        "protocol_sha256": digest,
        "payload": payload,
    }


def select_recovery_horizon(
    sweep_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_horizons: Sequence[int] = (1, 2, 4, 8, 10),
    minimum_adapted_successes: int = 2,
    require_positive_adapter_delta_vs_base: bool = True,
    fallback_horizon: int = 2,
) -> dict[str, Any]:
    """Pick H from same-horizon base vs adapted sweep rows.

    Each row: {horizon, base_success, adapted_success, best_progress?, first_divergence_step?}
    Aggregated per horizon across anchors.
    """

    by_h: dict[int, list[Mapping[str, Any]]] = {}
    for row in sweep_rows:
        by_h.setdefault(int(row["horizon"]), []).append(row)

    candidates: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for h in candidate_horizons:
        rows = by_h.get(int(h), [])
        n = len(rows)
        base = sum(bool(r.get("base_success")) for r in rows)
        adapted = sum(bool(r.get("adapted_success")) for r in rows)
        mean_progress = float(
            np.mean([float(r.get("best_progress", 0.0) or 0.0) for r in rows])
        ) if rows else 0.0
        mean_div = float(
            np.mean(
                [
                    float(r["first_divergence_step"])
                    for r in rows
                    if r.get("first_divergence_step") is not None
                ]
            )
        ) if any(r.get("first_divergence_step") is not None for r in rows) else -1.0
        summary[str(h)] = {
            "n": n,
            "base_successes": base,
            "adapted_successes": adapted,
            "adapter_delta": adapted - base,
            "mean_best_progress": mean_progress,
            "mean_first_divergence_step": mean_div,
        }
        ok = adapted >= int(minimum_adapted_successes)
        if require_positive_adapter_delta_vs_base:
            ok = ok and adapted > base
        if ok:
            candidates.append(
                {
                    "horizon": int(h),
                    "adapted_successes": adapted,
                    "base_successes": base,
                    "mean_best_progress": mean_progress,
                    "mean_first_divergence_step": mean_div,
                }
            )

    if not candidates:
        return {
            "selected_horizon": int(fallback_horizon),
            "selection_mode": "fallback",
            "summary": summary,
            "candidates": [],
        }

    candidates.sort(
        key=lambda c: (
            -int(c["adapted_successes"]),
            int(c["horizon"]),
            -float(c["mean_best_progress"]),
            -float(c["mean_first_divergence_step"]),
        )
    )
    # Prefer smallest horizon among those with max adapted successes and positive delta.
    best_adapted = max(int(c["adapted_successes"]) for c in candidates)
    top = [c for c in candidates if int(c["adapted_successes"]) == best_adapted]
    top.sort(
        key=lambda c: (
            int(c["horizon"]),
            -float(c["mean_best_progress"]),
            -float(c["mean_first_divergence_step"]),
        )
    )
    chosen = top[0]
    return {
        "selected_horizon": int(chosen["horizon"]),
        "selection_mode": "rule",
        "summary": summary,
        "candidates": candidates,
        "chosen": chosen,
    }


def check_receding_invariants(
    *,
    env_steps: int,
    execution_horizon: int,
    model_forward_calls: int,
    cache_resets: int,
    tolerance: int = 1,
) -> dict[str, Any]:
    """Validate fresh-forward ≈ ceil(env_steps / H)."""

    h = max(1, int(execution_horizon))
    expected = int(math.ceil(float(env_steps) / float(h))) if env_steps > 0 else 0
    forward_ok = abs(int(model_forward_calls) - expected) <= int(tolerance)
    reset_ok = int(cache_resets) >= max(0, int(model_forward_calls) - 1)
    mean_steps = (
        float(env_steps) / float(model_forward_calls) if model_forward_calls > 0 else 0.0
    )
    return {
        "env_steps": int(env_steps),
        "execution_horizon": h,
        "model_forward_calls": int(model_forward_calls),
        "cache_resets": int(cache_resets),
        "expected_forward_calls": expected,
        "mean_steps_per_forward": mean_steps,
        "forward_call_ok": forward_ok,
        "cache_reset_ok": reset_ok,
        "passed": bool(forward_ok and reset_ok),
    }


def piecewise_horizon_weights(
    horizon: int,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build mean-normalized piecewise weights [H] for native flow error."""

    import torch

    h = int(horizon)
    weights = torch.ones(h, device=device, dtype=dtype or torch.float32)
    if h > 0:
        weights[: min(4, h)] = 4.0
    if h > 4:
        weights[4 : min(8, h)] = 2.0
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def weighted_flow_loss_from_unreduced(
    losses: Any,
    *,
    enable_weighting: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Apply horizon weights to unreduced flow MSE [B, H, D]."""

    import torch

    if not torch.is_tensor(losses):
        raise TypeError("losses must be a tensor")
    if losses.ndim != 3:
        raise ValueError(f"expected [B,H,D] losses, got {tuple(losses.shape)}")
    metrics: dict[str, float] = {
        "loss_full": float(losses.mean().detach().cpu()),
        "loss_prefix_2": float(losses[:, : min(2, losses.shape[1]), :].mean().detach().cpu()),
        "loss_prefix_4": float(losses[:, : min(4, losses.shape[1]), :].mean().detach().cpu()),
        "loss_tail": float(
            losses[:, min(8, losses.shape[1]) :, :].mean().detach().cpu()
        )
        if losses.shape[1] > 8
        else float(losses.mean().detach().cpu()),
    }
    # Per-dimension groups for reporting (LIBERO 7D: xyz, rot, gripper).
    d = losses.shape[-1]
    prefix = losses[:, : min(4, losses.shape[1]), :]
    if d >= 3:
        metrics["prefix_translation_error"] = float(prefix[..., :3].mean().detach().cpu())
    if d >= 6:
        metrics["prefix_rotation_error"] = float(prefix[..., 3:6].mean().detach().cpu())
    if d >= 7:
        metrics["prefix_gripper_error"] = float(prefix[..., 6:7].mean().detach().cpu())

    if enable_weighting:
        w = piecewise_horizon_weights(
            losses.shape[1], device=losses.device, dtype=losses.dtype
        ).view(1, -1, 1)
        loss = (losses * w).mean()
    else:
        loss = losses.mean()
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


def native_flow_forward_weighted(
    policy: Any,
    batch: MutableMapping[str, Any],
    *,
    enable_weighting: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Compute native SmolVLA flow loss with optional horizon weighting.

    Uses the same preprocessing path as ``policy.forward`` but keeps the
    unreduced ``[B,H,D]`` flow MSE before reduction. Never adds sampled-action MSE.
    """

    import torch
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    # Work on a shallow copy so caller batch is not mutated unexpectedly.
    batch = dict(batch)
    if policy.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    actions_is_pad = batch.get("action_is_pad")
    losses = policy.model.forward(
        images, img_masks, lang_tokens, lang_masks, state, actions, None, None
    )
    original_action_dim = policy.config.action_feature.shape[0]
    losses = losses[:, :, :original_action_dim]
    if actions_is_pad is not None:
        losses = losses * (~actions_is_pad).unsqueeze(-1)
    losses = losses[:, :, : policy.config.max_action_dim]
    return weighted_flow_loss_from_unreduced(losses, enable_weighting=enable_weighting)


def choose_batch_kind(step: int, *, cycle_length: int = 10, clean_batches: int = 1) -> str:
    """Fixed 9 recovery + 1 clean schedule (clean on last slot of cycle)."""

    cycle = int(cycle_length)
    clean_n = int(clean_batches)
    if cycle <= 0 or clean_n < 0 or clean_n > cycle:
        raise ValueError("invalid batch schedule")
    index = int(step) % cycle
    # Last `clean_n` slots in the cycle are clean.
    if index >= cycle - clean_n:
        return "clean"
    return "recovery"


def sample_recovery_row(
    *,
    student_rows: Sequence[Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
    student_weight: float = 0.7222,
    original_weight: float = 0.2778,
    offset_weights: Mapping[int, float] | None = None,
    rng: random.Random | None = None,
) -> Mapping[str, Any]:
    """Sample one recovery row with source weights and query-offset bias."""

    rng = rng or random.Random()
    sw = float(student_weight)
    ow = float(original_weight)
    total = sw + ow
    if total <= 0:
        raise ValueError("recovery source weights must be positive")
    pick_student = rng.random() < (sw / total)
    pool = list(student_rows) if pick_student and student_rows else list(original_rows)
    if not pool and student_rows:
        pool = list(student_rows)
    if not pool and original_rows:
        pool = list(original_rows)
    if not pool:
        raise ValueError("no recovery rows available")

    # Anchor-balanced then offset-weighted within student pool.
    by_anchor: dict[str, list[Mapping[str, Any]]] = {}
    for row in pool:
        key = str(row.get("anchor_id") or row.get("state_key") or row.get("episode_id"))
        by_anchor.setdefault(key, []).append(row)
    anchor = rng.choice(sorted(by_anchor))
    candidates = by_anchor[anchor]
    weights_map = dict(offset_weights or {0: 1.0, 1: 0.75, 2: 0.50})
    weights = []
    for row in candidates:
        offset = int(row.get("offset_from_student_state", 0) or 0)
        source = str(row.get("source") or "")
        if source == "original_recovery":
            weights.append(1.0)
        else:
            weights.append(float(weights_map.get(offset, 0.25)))
    total_w = sum(weights)
    if total_w <= 0:
        return rng.choice(candidates)
    draw = rng.random() * total_w
    cumulative = 0.0
    for row, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if draw <= cumulative:
            return row
    return candidates[-1]


def sample_clean_row(
    clean_rows: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random | None = None,
) -> Mapping[str, Any]:
    rng = rng or random.Random()
    if not clean_rows:
        raise ValueError("no clean rows available")
    return rng.choice(list(clean_rows))


def dagger_qc_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """QC that distinguishes query-state vs teacher-suffix coverage."""

    anchors = sorted({str(r.get("anchor_id") or r.get("failure_key")) for r in rows})
    query = [r for r in rows if str(r.get("source")) == "student_query_state"]
    suffix = [r for r in rows if str(r.get("source")) == "teacher_suffix_after_student_query"]
    unique_queries = {
        (str(r.get("anchor_id") or r.get("failure_key")), str(r.get("query_state_id") or r.get("sample_id")))
        for r in query
    }
    successes = sum(1 for r in rows if bool(r.get("teacher_rollout_success", True)))
    lengths = [int(r["teacher_recovery_length"]) for r in rows if r.get("teacher_recovery_length") is not None]
    return {
        "schema_version": "rase-pre-c1-2-dagger-qc/v1",
        "n_rows": len(rows),
        "anchors_covered": len(anchors),
        "anchor_ids": anchors,
        "unique_student_query_states": len(unique_queries),
        "query_state_chunks": len(query),
        "teacher_suffix_chunks": len(suffix),
        "successful_teacher_rows": successes,
        "median_teacher_recovery_length": float(np.median(lengths)) if lengths else 0.0,
        "n_oft_queries": len({str(r.get("query_id")) for r in rows if r.get("query_id")}),
    }


def interface_mismatch_decision(
    *,
    env_action_mae: float,
    cross_successor_error: float,
    sim_floor_error: float,
    student_repeat_error: float = 0.0,
    cross_over_sim_floor_ratio: float = 5.0,
    env_action_mae_low: float = 0.02,
    abs_cross_floor: float = 1e-3,
    abs_cross_mismatch: float = 0.20,
) -> dict[str, Any]:
    """Go/No-Go for Phase 0 interface integrity.

    Near-perfect restores make raw sim-floor ≈ 0, which inflates ratios into the
    millions even for ordinary cross-policy successor gaps. Use a robust floor
    and require BOTH near-identical env actions and a large absolute successor
    gap before blocking training.
    """

    floor = max(float(sim_floor_error), float(student_repeat_error), float(abs_cross_floor))
    ratio = float(cross_successor_error) / floor
    low_mae = float(env_action_mae) <= float(env_action_mae_low)
    large_cross = float(cross_successor_error) >= float(abs_cross_mismatch)
    mismatch = low_mae and large_cross and ratio >= float(cross_over_sim_floor_ratio)
    return {
        "env_action_mae": float(env_action_mae),
        "cross_successor_error": float(cross_successor_error),
        "sim_floor_error": float(sim_floor_error),
        "student_repeat_error": float(student_repeat_error),
        "robust_floor": floor,
        "cross_over_sim_floor_ratio": ratio,
        "interface_mismatch": mismatch,
        "block_training": mismatch,
        "decision": "fix_interface" if mismatch else "proceed",
    }


def aggregate_dagger_global_qc(
    run_payloads: Sequence[Mapping[str, Any]],
    *,
    locked_state_keys: Sequence[str],
    seeds_per_anchor: int = 5,
    mins: Mapping[str, Any] | None = None,
    failed_teacher_count: int = 0,
) -> dict[str, Any]:
    """Global DAgger QC over root-level run summaries (not suite-local dagger_qc.json)."""

    mins = dict(mins or {})
    locked = [str(k) for k in locked_state_keys]
    runs = [
        p
        for p in run_payloads
        if str(p.get("schema_version")) == "rase-pre-c1-2-dagger-run/v1"
    ]
    by_anchor: dict[str, list[Mapping[str, Any]]] = {}
    all_rows: list[Mapping[str, Any]] = []
    n_queries = 0
    n_successful_teacher = 0
    for payload in runs:
        anchor = str(payload.get("anchor_id") or "")
        by_anchor.setdefault(anchor, []).append(payload)
        all_rows.extend(list(payload.get("accepted_rows") or []))
        n_queries += int(payload.get("n_queries") or 0)
        n_successful_teacher += int(payload.get("n_successful_teacher") or 0)

    # Distinct successful teacher queries = unique query_id among accepted rows.
    success_query_ids = sorted(
        {
            str(r.get("query_id"))
            for r in all_rows
            if r.get("query_id") and bool(r.get("teacher_rollout_success", True))
        }
    )
    trigger_query_success: dict[str, set[str]] = {}
    trigger_query_rows: dict[str, int] = {}
    student_steps_success: list[int] = []
    recovery_lengths: list[int] = []
    offset_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in all_rows:
        source = str(row.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        offset = str(int(row.get("offset_from_student_state", 0) or 0))
        offset_counts[offset] = offset_counts.get(offset, 0) + 1
        trigger = str(row.get("query_trigger") or "unknown")
        trigger_query_rows[trigger] = trigger_query_rows.get(trigger, 0) + 1
        qid = str(row.get("query_id") or "")
        if qid and bool(row.get("teacher_rollout_success", True)):
            trigger_query_success.setdefault(trigger, set()).add(qid)
        if source == "student_query_state" and row.get("teacher_recovery_length") is not None:
            recovery_lengths.append(int(row["teacher_recovery_length"]))
        # student step proxy: parse ...__qN
        if source == "student_query_state" and qid:
            try:
                q_index = int(qid.rsplit("__q", 1)[-1])
            except ValueError:
                q_index = -1
            # Collector queries every period and at start; approximate step via index*period later.
            student_steps_success.append(q_index)

    per_anchor: dict[str, Any] = {}
    per_anchor_ok = True
    for key in locked:
        payloads = by_anchor.get(key, [])
        rows = []
        for payload in payloads:
            rows.extend(list(payload.get("accepted_rows") or []))
        query_rows = [r for r in rows if str(r.get("source")) == "student_query_state"]
        unique_states = {
            str(r.get("query_state_id") or r.get("sample_id")) for r in query_rows
        }
        success_relabels = len({str(r.get("query_id")) for r in query_rows if r.get("query_id")})
        seeds = sorted({int(p.get("seed")) for p in payloads if p.get("seed") is not None})
        # Also count seed index from run_id ...__sN
        seed_indices = sorted(
            {
                int(str(p.get("run_id")).rsplit("__s", 1)[-1])
                for p in payloads
                if "__s" in str(p.get("run_id") or "")
            }
        )
        n_seeds = max(len(seeds), len(seed_indices), len(payloads))
        accepted_near = len(rows)
        min_unique = int(mins.get("unique_student_query_states_per_anchor", 10))
        min_success = int(mins.get("successful_teacher_relabels_per_anchor", 5))
        min_near = int(mins.get("accepted_query_near_chunks_per_anchor", 20))
        min_seeds = int(mins.get("seeds_per_anchor", seeds_per_anchor))
        ok = (
            n_seeds >= min_seeds
            and len(unique_states) >= min_unique
            and success_relabels >= min_success
            and accepted_near >= min_near
        )
        if not ok:
            per_anchor_ok = False
        per_anchor[key] = {
            "n_run_summaries": len(payloads),
            "n_seeds": n_seeds,
            "seed_indices": seed_indices,
            "unique_student_query_states": len(unique_states),
            "successful_teacher_relabels": success_relabels,
            "accepted_query_near_chunks": accepted_near,
            "query_state_chunks": len(query_rows),
            "meets_round1_minimum": ok,
            "triggers": dict(
                sorted(
                    Counter(str(r.get("query_trigger") or "unknown") for r in query_rows).items()
                )
            ),
        }

    overall_success_rate = (
        float(n_successful_teacher) / float(n_queries) if n_queries else 0.0
    )
    by_trigger = {}
    for trigger, qids in sorted(trigger_query_success.items()):
        # Approximate denominator unavailable without failed-trigger labels; report success counts + row shares.
        by_trigger[trigger] = {
            "successful_queries": len(qids),
            "accepted_rows": int(trigger_query_rows.get(trigger, 0)),
        }

    return {
        "schema_version": "rase-pre-c1-2-dagger-global-qc/v1",
        "n_run_summaries": len(runs),
        "anchors_locked": len(locked),
        "anchors_covered": len([k for k in locked if k in by_anchor]),
        "missing_anchors": [k for k in locked if k not in by_anchor],
        "n_oft_queries": int(n_queries),
        "n_successful_teacher_queries": int(n_successful_teacher),
        "p_oft_success_given_student_query": overall_success_rate,
        "n_distinct_successful_query_ids": len(success_query_ids),
        "n_accepted_rows": len(all_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "offset_counts": dict(sorted(offset_counts.items())),
        "by_trigger_successful_queries": by_trigger,
        "median_teacher_recovery_length": float(np.median(recovery_lengths))
        if recovery_lengths
        else 0.0,
        "success_query_index_quartiles": {
            "p25": float(np.quantile(student_steps_success, 0.25)) if student_steps_success else None,
            "p50": float(np.quantile(student_steps_success, 0.50)) if student_steps_success else None,
            "p75": float(np.quantile(student_steps_success, 0.75)) if student_steps_success else None,
        },
        "failed_teacher_json_count": int(failed_teacher_count),
        "failed_teacher_enter_bc": False,
        "per_anchor": per_anchor,
        "meets_round1_minimum_all_anchors": bool(per_anchor_ok) and len(by_anchor) >= len(locked),
        "note": "Do not use suite-local dagger_qc.json as global QC; aggregate root run summaries.",
    }


def r0_decision_from_diagnostics(
    *,
    teacher_forced: Mapping[str, Any],
    recoverability: Mapping[str, Any],
    dagger_qc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map R0 diagnostics to the next branch (no automatic legacy E3/E4)."""

    tf_orig = dict(teacher_forced.get("original_c1_1") or teacher_forced.get("original") or {})
    tf_query = dict(teacher_forced.get("r1_student_query") or teacher_forced.get("query") or {})
    adapted_orig = float(tf_orig.get("adapted_loss_full", tf_orig.get("adapted_mean_loss", 1e9)))
    base_orig = float(tf_orig.get("base_loss_full", tf_orig.get("base_mean_loss", 1e9)))
    adapted_query = float(
        tf_query.get("adapted_loss_full", tf_query.get("adapted_mean_loss", 1e9))
    )
    base_query = float(tf_query.get("base_loss_full", tf_query.get("base_mean_loss", 1e9)))

    # Heuristic thresholds for native flow loss on successful OFT chunks.
    tf_orig_good = adapted_orig < 0.75 * base_orig and adapted_orig < 0.35
    tf_query_good = adapted_query < 0.90 * base_query and adapted_query < 0.45

    curves = dict(recoverability.get("curves") or {})
    r_base = {int(k): float(v) for k, v in dict(curves.get("base") or {}).items()}
    r_adapted = {int(k): float(v) for k, v in dict(curves.get("adapted") or {}).items()}
    r0 = float(recoverability.get("R_oft_k0", r_base.get(0, curves.get("oft_k0", 0.0)) or 0.0))
    if "oft_k0" in curves and not isinstance(curves.get("oft_k0"), dict):
        r0 = float(curves["oft_k0"])
    rb1 = float(r_base.get(1, 0.0))
    ra1 = float(r_adapted.get(1, 0.0))
    rb4 = float(r_base.get(4, 0.0))
    ra4 = float(r_adapted.get(4, 0.0))
    decay_fast = (r0 - max(rb1, ra1)) >= 0.25 or (max(rb1, ra1) - max(rb4, ra4)) >= 0.25
    adapted_one_step_gain = (ra1 - rb1) >= 0.05
    adapted_short_gain = (ra4 - rb4) >= 0.05
    oft_success = float(
        (dagger_qc or {}).get("p_oft_success_given_student_query", 0.0) or 0.0
    )

    if not tf_orig_good:
        branch = "stop_dagger_inspect_optimization_capacity_target"
        rationale = "Teacher-forced fit on original C1.1 OFT states is weak; do not expand DAgger or run full BC."
        next_actions = [
            "exact_state_overfit_smoke_8_to_32",
            "check_native_flow_loss_wiring",
            "optional_full_ft_diagnostic_before_capacity_ladder",
        ]
    elif tf_orig_good and (not tf_query_good) and oft_success < 0.60:
        branch = "recoverability_aware_dagger_early_query"
        rationale = "Original TF OK but student-query TF/occupancy weak; resample early recoverable states."
        next_actions = [
            "prefer_anchor_start_and_first_deviation_queries",
            "disable_or_weaken_periodic_late_queries",
            "train_only_d_student_intersect_R_oft",
            "drop_long_teacher_suffix_from_bc",
        ]
    elif tf_orig_good and (max(rb1, ra1) < 0.70 * max(r0, 1e-6) or max(rb1, ra1) + 1e-9 < r0 - 0.20):
        branch = "first_action_correction_residual"
        rationale = "TF OK but one-step recoverability already drops; fix first corrective action / residual target."
        next_actions = [
            "inspect_fixed_seed_sampled_action_and_successor_effect",
            "train_short_horizon_residual_correction",
            "do_not_run_long_trajectory_oft_bc",
        ]
    elif decay_fast and not adapted_short_gain:
        branch = "short_horizon_corrective_plus_aware_dagger"
        rationale = "R(1) ok-ish but R(2-4) decays; prioritize short-horizon corrective learning and early queries."
        next_actions = [
            "train_for_R_adapted_1_to_4_gt_R_base",
            "reobserve_every_1_to_4_steps",
            "recoverability_aware_dagger",
        ]
    elif adapted_one_step_gain and adapted_short_gain:
        branch = "repeated_correction_handback"
        rationale = "Adapted improves short recoverability; add repeated correction / handback before capacity."
        next_actions = [
            "recovery_conditioned_lora_on_off",
            "return_to_base_competence_handback",
            "terminal_success_as_final_gate_only",
        ]
    else:
        branch = "revised_short_horizon_training"
        rationale = "Default revised path: early recoverable student states + short-horizon objective; legacy E3/E4 remains paused."
        next_actions = [
            "build_early_query_dataset",
            "residual_or_prefix_weighted_short_horizon_train",
            "gate_on_R_k_before_terminal_8pp",
        ]

    return {
        "schema_version": "rase-pre-c1-2-r0-decision/v1",
        "branch": branch,
        "rationale": rationale,
        "next_actions": next_actions,
        "metrics": {
            "tf_original_adapted_loss": adapted_orig,
            "tf_original_base_loss": base_orig,
            "tf_original_good": tf_orig_good,
            "tf_query_adapted_loss": adapted_query,
            "tf_query_base_loss": base_query,
            "tf_query_good": tf_query_good,
            "R_oft_0": r0,
            "R_base_1": rb1,
            "R_adapted_1": ra1,
            "R_base_4": rb4,
            "R_adapted_4": ra4,
            "adapted_one_step_gain": adapted_one_step_gain,
            "adapted_short_gain": adapted_short_gain,
            "decay_fast": decay_fast,
            "p_oft_success_student_query": oft_success,
        },
        "legacy_e3_e4_allowed": False,
        "capacity_ladder_allowed": branch == "stop_dagger_inspect_optimization_capacity_target"
        and bool(teacher_forced.get("full_ft_beats_lora")),
        "terminal_8pp_is_final_gate_only": True,
    }


def capacity_ladder_step(name: str, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return frozen single-variable capacity experiment config."""

    capacity = dict(lock.get("capacity") or {})
    order = list(capacity.get("order") or [])
    if name not in order:
        raise ValueError(f"unknown capacity step {name}; expected one of {order}")
    base_lora = dict(lock.get("lora") or {})
    train = dict(lock.get("train") or {})
    if name == "expand_lora_targets":
        return {
            "step": name,
            "lora_rank": 16,
            "target_modules": list(capacity.get("expand_lora_targets") or base_lora.get("target_modules")),
            "train_mode": "lora",
            "optimizer": {"lr": train.get("lr"), "weight_decay": train.get("weight_decay")},
        }
    if name == "rank_32":
        return {
            "step": name,
            "lora_rank": 32,
            "target_modules": list(capacity.get("expand_lora_targets") or base_lora.get("target_modules")),
            "train_mode": "lora",
            "optimizer": {"lr": train.get("lr"), "weight_decay": train.get("weight_decay")},
        }
    if name == "rank_64":
        return {
            "step": name,
            "lora_rank": 64,
            "target_modules": list(capacity.get("expand_lora_targets") or base_lora.get("target_modules")),
            "train_mode": "lora",
            "optimizer": {"lr": train.get("lr"), "weight_decay": train.get("weight_decay")},
        }
    if name == "full_action_expert":
        opt = dict(capacity.get("full_action_expert_optimizer") or {})
        return {
            "step": name,
            "lora_rank": None,
            "target_modules": [],
            "train_mode": "full_action_expert",
            "optimizer": opt,
        }
    if name == "top_cross_modal_layers":
        return {
            "step": name,
            "lora_rank": None,
            "target_modules": [],
            "train_mode": "top_cross_modal",
            "optimizer": dict(capacity.get("full_action_expert_optimizer") or {}),
            "requires_visual_probe_evidence": True,
        }
    raise ValueError(f"unhandled capacity step: {name}")

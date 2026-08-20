"""PRE-C1 protocol helpers and recovery gate analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROTOCOL_LOCK_VERSION = "rase-pre-c1-protocol-lock/v1"
PROTOCOL_LOCK_VERSION_C1_1 = "rase-pre-c1-1-protocol-lock/v1"
ALLOWED_PROTOCOL_LOCK_VERSIONS = {PROTOCOL_LOCK_VERSION, PROTOCOL_LOCK_VERSION_C1_1}
ALLOWED_PHASES = {"PRE-C1", "PRE-C1.1"}
DATASET_VERSION = "rase-pre-c1-distill-dataset/v1"
DATASET_VERSION_C1_1 = "rase-pre-c1-1-distill-dataset/v1"
GATE_AUDIT_VERSION = "rase-pre-c1-recovery-gate/v1"
REQUIRED_LOCK_KEYS = (
    "schema_version",
    "phase",
    "method",
    "lora",
    "train",
    "gate",
    "dataset",
    "sealed",
)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_protocol_lock(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PRE-C1 protocol lock must be a mapping")
    missing = [key for key in REQUIRED_LOCK_KEYS if key not in payload]
    if missing:
        raise ValueError(f"PRE-C1 protocol lock missing keys: {missing}")
    if payload.get("schema_version") not in ALLOWED_PROTOCOL_LOCK_VERSIONS:
        raise ValueError(
            f"unexpected protocol schema_version: {payload.get('schema_version')}"
        )
    if payload.get("phase") not in ALLOWED_PHASES:
        raise ValueError(f"unexpected phase: {payload.get('phase')}")
    return payload


def validate_protocol_lock(payload: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation errors (empty => ok)."""

    errors: list[str] = []
    for key in REQUIRED_LOCK_KEYS:
        if key not in payload:
            errors.append(f"missing:{key}")
    if payload.get("schema_version") not in ALLOWED_PROTOCOL_LOCK_VERSIONS:
        errors.append("bad_schema_version")
    if payload.get("phase") not in ALLOWED_PHASES:
        errors.append("bad_phase")
    gate = dict(payload.get("gate") or {})
    for key in ("recovery_gain_pp", "clean_retention_drop_pp", "bootstrap_seed"):
        if key not in gate:
            errors.append(f"missing_gate:{key}")
    lora = dict(payload.get("lora") or {})
    if int(lora.get("rank") or 0) < 1:
        errors.append("lora_rank")
    if not list(lora.get("target_modules") or []):
        errors.append("lora_target_modules")
    sealed = dict(payload.get("sealed") or {})
    if sealed.get("world_model_gate") != "closed":
        errors.append("world_model_must_stay_closed")
    if sealed.get("hidden_test24") != "sealed":
        errors.append("hidden_test_must_stay_sealed")
    return errors


def episode_grouped_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    episodes = sorted({str(row["episode_id"]) for row in rows})
    rng = np.random.default_rng(int(seed))
    order = list(episodes)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * float(val_fraction)))) if order else 0
    val_eps = set(order[:n_val])
    train_eps = [ep for ep in order if ep not in val_eps]
    if order and not train_eps:
        # Keep at least one train episode when possible.
        moved = order[0]
        val_eps.discard(moved)
        train_eps = [moved]
    train_rows = [dict(row) for row in rows if str(row["episode_id"]) not in val_eps]
    val_rows = [dict(row) for row in rows if str(row["episode_id"]) in val_eps]
    return {
        "schema_version": "rase-pre-c1-benchmark-splits/v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_episodes": sorted(train_eps),
        "val_episodes": sorted(val_eps),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "leakage_episode_overlap": sorted(set(train_eps) & set(val_eps)),
    }


def analyze_pre_c1_recovery_gate(
    *,
    recovery_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    recovery_gain_pp: float = 8.0,
    clean_retention_drop_pp: float = 2.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 2_026_080_405,
) -> dict[str, Any]:
    """Dual gate: recovery gain vs clean retention drop."""

    rec = [dict(row) for row in recovery_rows]
    ret = [dict(row) for row in retention_rows]
    if not rec:
        raise ValueError("recovery eval rows required")
    if not ret:
        raise ValueError("retention eval rows required")

    n_rec = len(rec)
    base_rec = sum(bool(row.get("base_success")) for row in rec)
    adapted_rec = sum(bool(row.get("adapted_success")) for row in rec)
    recovery_gain = 100.0 * (adapted_rec - base_rec) / n_rec

    n_ret = len(ret)
    base_ret = sum(bool(row.get("base_success")) for row in ret)
    adapted_ret = sum(bool(row.get("adapted_success")) for row in ret)
    # Retention uses adapter_off for adapted_success in the retention arm.
    retention_drop = 100.0 * (base_ret - adapted_ret) / n_ret

    def _cluster_ci(rows: list[dict[str, Any]], gain_fn) -> dict[str, Any]:
        by_ep: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_ep.setdefault(str(row.get("episode_id") or row.get("state_key")), []).append(row)
        eps = sorted(by_ep)
        point = gain_fn(rows)
        rng = np.random.default_rng(bootstrap_seed)
        draws = np.empty(int(bootstrap_replicates), dtype=np.float64)
        for i in range(int(bootstrap_replicates)):
            sample_ids = rng.choice(eps, size=len(eps), replace=True)
            sample = [row for ep in sample_ids for row in by_ep[ep]]
            draws[i] = gain_fn(sample)
        lower, upper = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
        return {
            "point_estimate_pp": point,
            "ci95_pp": [lower, upper],
            "ci95_lower_positive": lower > 0.0,
            "n_episodes": len(eps),
        }

    def recovery_gain_fn(rows: list[dict[str, Any]]) -> float:
        n = len(rows)
        if n == 0:
            return 0.0
        return 100.0 * (
            sum(bool(r.get("adapted_success")) for r in rows)
            - sum(bool(r.get("base_success")) for r in rows)
        ) / n

    recovery_boot = _cluster_ci(rec, recovery_gain_fn)

    pass_conditions = {
        "recovery_gain_ge_threshold": recovery_gain >= float(recovery_gain_pp),
        "recovery_ci_lower_positive": bool(recovery_boot["ci95_lower_positive"]),
        "clean_retention_drop_le_threshold": retention_drop
        <= float(clean_retention_drop_pp),
    }
    passed = all(pass_conditions.values())
    if passed:
        decision = "same_backbone_recovery_method_eligible"
    elif retention_drop > float(clean_retention_drop_pp):
        decision = "abstention_track_required"
    else:
        decision = "capacity_or_data_review"

    return {
        "schema_version": GATE_AUDIT_VERSION,
        "n_recovery_states": n_rec,
        "n_retention_states": n_ret,
        "recovery_successes": {"base": base_rec, "adapted": adapted_rec},
        "retention_successes": {"base": base_ret, "adapted_off": adapted_ret},
        "recovery_gain_pp": recovery_gain,
        "clean_retention_drop_pp": retention_drop,
        "thresholds": {
            "recovery_gain_pp": float(recovery_gain_pp),
            "clean_retention_drop_pp": float(clean_retention_drop_pp),
        },
        "recovery_bootstrap": recovery_boot,
        "pass_conditions": pass_conditions,
        "gate_pass": passed,
        "decision": decision,
        "same_backbone_recovery_method": "eligible" if passed else "closed",
        "abstention_track": (
            "required" if decision == "abstention_track_required" else "not_required"
        ),
        "world_model_gate": "closed",
        "naming": "recovery LoRA / OFT action distillation",
        "not_runtime_oft": True,
        "not_smolvla_flow_api_guidance": True,
    }

#!/usr/bin/env python3
"""Freeze the PRE-C0 48-state {T1,T3} audit manifest and protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SCHEMA_VERSION = "rase-pre-c0-48-state-manifest/v1"
PROTOCOL_LOCK_VERSION = "rase-pre-c0-protocol-lock/v1"
AUDIT_STAGES = ("T1", "T3")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        # Fallback: YAML-compatible JSON subset.
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def build_manifest(
    stage_keys: dict[str, Any],
    *,
    stages: tuple[str, ...] = AUDIT_STAGES,
) -> dict[str, Any]:
    episode_meta = {
        str(episode["episode_id"]): episode
        for episode in stage_keys.get("episodes") or []
        if isinstance(episode, dict) and episode.get("episode_id")
    }
    records = [
        dict(row)
        for row in stage_keys.get("selected_states") or stage_keys.get("records") or []
        if str(row.get("stage")) in set(stages)
    ]
    # Prefer episode.stages when selected_states absent.
    if not records and stage_keys.get("episodes"):
        for episode in stage_keys["episodes"]:
            for stage in stages:
                stage_row = (episode.get("stages") or {}).get(stage)
                if not stage_row:
                    continue
                records.append(
                    {
                        "episode_id": episode["episode_id"],
                        "task_id": episode.get("logical_task_id") or episode.get("task_id"),
                        "logical_task_id": episode.get("logical_task_id"),
                        "concrete_task_id": episode.get("concrete_task_id"),
                        "suite": episode.get("suite"),
                        "cell": episode.get("cell"),
                        "stage": stage,
                        "stage_name": stage_row.get("name"),
                        "state_key": stage_row["state_key"],
                        "step": stage_row.get("step"),
                        "temporal_fallback": episode.get("temporal_fallback"),
                        "reliable": (episode.get("reliability") or {}).get("reliable"),
                    }
                )
    if not records:
        raise ValueError("no T1/T3 stage records available to freeze")

    enriched = []
    for row in records:
        item = dict(row)
        episode = episode_meta.get(str(item.get("episode_id")), {})
        item.setdefault("suite", episode.get("suite"))
        item.setdefault("cell", episode.get("cell"))
        item.setdefault(
            "task_id",
            item.get("logical_task_id")
            or episode.get("logical_task_id")
            or episode.get("task_id"),
        )
        item.setdefault("concrete_task_id", episode.get("concrete_task_id"))
        item.setdefault("temporal_fallback", episode.get("temporal_fallback"))
        if "reliable" not in item:
            item["reliable"] = (episode.get("reliability") or {}).get("reliable")
        enriched.append(item)
    records = enriched

    episode_ids = sorted({str(row["episode_id"]) for row in records})
    state_keys = [str(row["state_key"]) for row in records]
    if len(state_keys) != len(set(state_keys)):
        raise ValueError("duplicate state_key in 48-state cohort")
    if any(row.get("cell") is None for row in records):
        raise ValueError("every frozen state requires a cell label")

    by_episode_stage = {(str(row["episode_id"]), str(row["stage"])) for row in records}
    missing_pairs = []
    for episode_id in episode_ids:
        for stage in stages:
            if (episode_id, stage) not in by_episode_stage:
                missing_pairs.append({"episode_id": episode_id, "stage": stage})

    return {
        "schema_version": SCHEMA_VERSION,
        "n_states": len(records),
        "n_episodes": len(episode_ids),
        "stages": list(stages),
        "expected_n_states": len(episode_ids) * len(stages),
        "missing_episode_stage_pairs": missing_pairs,
        "records": sorted(
            records,
            key=lambda row: (str(row["episode_id"]), str(row["stage"]), str(row["state_key"])),
        ),
        "inclusion_rules": {
            "source_split": "train",
            "stages": list(stages),
            "one_state_per_episode_stage": True,
            "uses_corrective_outcomes": False,
        },
        "exclusion_rules": {
            "pre_a3_val": True,
            "pre_a3_test": True,
            "hidden_test24": "sealed",
        },
        "reliability_summary": (stage_keys.get("reliability_summary") or {}),
        "source_stage_keys_schema": stage_keys.get("schema_version"),
    }


def build_protocol_lock(
    *,
    manifest: dict[str, Any],
    design_path: Path,
    stage_keys_path: Path,
    config_path: Path | None,
    pool_path: Path | None,
    policy_checkpoint: Path | None,
) -> dict[str, Any]:
    design = _load(design_path)
    config = _load(config_path) if config_path and config_path.is_file() else {}
    adapter = dict(config.get("adapter_config") or {})
    return {
        "schema_version": PROTOCOL_LOCK_VERSION,
        "frozen_before_gate_a_outcomes": True,
        "design_path": str(design_path),
        "design_sha256": design.get("design_sha256") or _sha_file(design_path),
        "stage_keys_path": str(stage_keys_path),
        "stage_keys_sha256": _sha_file(stage_keys_path),
        "pool_path": None if pool_path is None else str(pool_path),
        "pool_manifest_sha256": (
            None if pool_path is None or not (pool_path / "manifest.json").is_file()
            else _sha_file(pool_path / "manifest.json")
        ),
        "config_path": None if config_path is None else str(config_path),
        "policy_checkpoint": (
            str(policy_checkpoint)
            if policy_checkpoint is not None
            else adapter.get("policy_path", "ckpts/smolvla_libero")
        ),
        "policy_checkpoint_sha256": (
            _sha_file(policy_checkpoint)
            if policy_checkpoint is not None and policy_checkpoint.is_file()
            else None
        ),
        "candidate_arms": {
            "current_suffix": 1,
            "strict_resample_k": 8,
            "fresh_replan_k": 4,
            "execution_horizons": [1, 2, 4],
        },
        "gate_a": {
            "natural_headroom_pp": 5.0,
            "min_rescue_suites": 2,
            "min_rescue_tasks": 3,
            "max_control_harm_rate": 0.05,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 2_026_080_405,
            "cluster_unit": "episode_id",
        },
        "sealed": {
            "pre_a3_method_gate": "closed",
            "hidden_test24": "sealed",
            "world_model_gate": "closed",
            "pre_b_allowed": False,
        },
        "n_states": manifest["n_states"],
        "n_episodes": manifest["n_episodes"],
        "stages": manifest["stages"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-keys", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/collect_pre_c0_deviation_pilot24.json"))
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/pre_c0/pre_c0_48_state_manifest.json"),
    )
    parser.add_argument(
        "--protocol-lock-output",
        type=Path,
        default=Path("artifacts/pre_c0/pre_c0_protocol_lock.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-48", action="store_true", default=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    for path in (args.manifest_output, args.protocol_lock_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}; pass --overwrite")

    stage_keys = _load(args.stage_keys.resolve())
    manifest = build_manifest(stage_keys)
    if not args.allow_incomplete:
        if manifest["missing_episode_stage_pairs"]:
            raise SystemExit(
                f"incomplete episode/stage coverage: {manifest['missing_episode_stage_pairs'][:5]}"
            )
        if args.require_48 and manifest["n_states"] != 48:
            raise SystemExit(
                f"expected 48 states, got {manifest['n_states']} "
                f"across {manifest['n_episodes']} episodes"
            )
        reliability = float((manifest.get("reliability_summary") or {}).get("reliable_rate") or 0.0)
        if reliability < 0.80:
            raise SystemExit(
                f"stage-key reliability {reliability:.3f} < 0.80; refuse to freeze"
            )

    lock = build_protocol_lock(
        manifest=manifest,
        design_path=args.design.resolve(),
        stage_keys_path=args.stage_keys.resolve(),
        config_path=None if args.config is None else args.config.resolve(),
        pool_path=None if args.pool is None else args.pool.resolve(),
        policy_checkpoint=(
            None if args.policy_checkpoint is None else args.policy_checkpoint.resolve()
        ),
    )
    _write_json(args.manifest_output, manifest)
    _write_yaml(args.protocol_lock_output, lock)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_output),
                "protocol_lock": str(args.protocol_lock_output),
                "n_states": manifest["n_states"],
                "n_episodes": manifest["n_episodes"],
                "reliable_rate": (manifest.get("reliability_summary") or {}).get(
                    "reliable_rate"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

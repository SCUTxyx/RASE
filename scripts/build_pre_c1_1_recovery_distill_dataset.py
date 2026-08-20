#!/usr/bin/env python3
"""Build PRE-C1.1 multi-chunk distill dataset from successful OFT trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rase.adapt.pre_c1 import (
    DATASET_VERSION_C1_1,
    episode_grouped_split,
    load_protocol_lock,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_rollouts(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c0-corrective-rollout/v1":
            continue
        rows.append(payload)
    return rows


def _current_arm(row: dict[str, Any]) -> dict[str, Any] | None:
    for arm in row.get("arms") or []:
        if str(arm.get("family")) == "current_suffix":
            return dict(arm)
    return None


def _load_teacher_index(teacher_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(teacher_dir.glob("*.json")):
        if path.name.endswith("_qc.json") or path.name == "run_manifest.json":
            continue
        if path.name.startswith("suite_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rase-pre-c1-1-oft-success-traj/v1":
            continue
        index[str(payload["state_key"])] = payload
    # Also scan suite_* subdirs.
    for sub in sorted(teacher_dir.glob("suite_*")):
        if not sub.is_dir():
            continue
        for path in sorted(sub.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "rase-pre-c1-1-oft-success-traj/v1":
                continue
            index[str(payload["state_key"])] = payload
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--oft-teacher-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--flat-chunks-jsonl", type=Path, required=True)
    parser.add_argument("--splits-output", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--qc-md", type=Path, required=True)
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock.resolve())
    ds_cfg = dict(lock["dataset"])
    rollouts = _load_rollouts(args.rollout_dir.resolve())
    teachers = _load_teacher_index(args.oft_teacher_dir.resolve())
    rollout_by_key = {str(r["state_key"]): r for r in rollouts}

    samples: list[dict[str, Any]] = []
    flat_chunks: list[dict[str, Any]] = []
    attempted = 0
    success_states = 0
    failed_states = 0
    missing_teacher = 0

    # All successful OFT teachers (pilot48 failures + expanded T0/T2/T4).
    for state_key, teacher in sorted(teachers.items()):
        attempted += 1
        if not bool(teacher.get("rollout_success")):
            failed_states += 1
            continue
        chunks = list(teacher.get("chunks") or [])
        if not chunks:
            failed_states += 1
            continue
        success_states += 1
        row = rollout_by_key.get(state_key) or {}
        for chunk_meta in chunks:
            chunk_path = str(chunk_meta["chunk_path"])
            sample = {
                "schema_version": DATASET_VERSION_C1_1,
                "state_key": state_key,
                "episode_id": teacher.get("episode_id") or row.get("episode_id"),
                "task_id": teacher.get("task_id") or row.get("task_id"),
                "concrete_task_id": row.get("concrete_task_id"),
                "suite": teacher.get("suite") or row.get("suite"),
                "cell": teacher.get("cell") or row.get("cell"),
                "stage": teacher.get("stage") or row.get("stage"),
                "clean_flag": False,
                "teacher_source": "oft",
                "chunk_index": int(chunk_meta.get("chunk_index", 0)),
                "timestep": int(chunk_meta.get("timestep", 0)),
                "chunk_path": chunk_path,
                "teacher_actions": None,
                "obs_cached": True,
                "episode_success": True,
                "dual_track_label": "oft_success_multi_chunk",
                "source_episode_outcome": row.get("source_episode_outcome"),
                "sample_id": f"{state_key}::chunk_{int(chunk_meta.get('chunk_index', 0)):04d}",
            }
            samples.append(sample)
            flat_chunks.append(
                {
                    "state_key": state_key,
                    "episode_id": sample["episode_id"],
                    "suite": sample["suite"],
                    "chunk_index": sample["chunk_index"],
                    "timestep": sample["timestep"],
                    "chunk_path": chunk_path,
                    "rollout_success": True,
                }
            )

    # Clean retention from PRE-C0 rollouts.
    for row in rollouts:
        is_clean = (
            str(row.get("cell")) == str(ds_cfg.get("clean_retention_cell", "clean:L0"))
            and str(row.get("source_episode_outcome"))
            == str(ds_cfg.get("clean_retention_source_outcome", "success"))
        )
        if not is_clean:
            continue
        current = _current_arm(row)
        if current is None or not current.get("action_tensor"):
            continue
        samples.append(
            {
                "schema_version": DATASET_VERSION_C1_1,
                "state_key": row["state_key"],
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id"),
                "concrete_task_id": row.get("concrete_task_id"),
                "suite": row.get("suite"),
                "cell": row.get("cell"),
                "stage": row.get("stage"),
                "clean_flag": True,
                "teacher_source": "smolvla_base",
                "teacher_actions": current.get("action_tensor") or [],
                "chunk_path": None,
                "obs_cached": False,
                "dual_track_label": "clean_retention",
                "source_episode_outcome": row.get("source_episode_outcome"),
                "sample_id": f"{row['state_key']}::clean",
            }
        )

    recovery_chunks = sum(1 for s in samples if not s["clean_flag"])
    clean_n = sum(1 for s in samples if s["clean_flag"])
    min_success = int(ds_cfg.get("min_successful_recovery_states", 16))
    min_chunks = int(ds_cfg.get("min_train_chunks", 200))
    min_clean = int(ds_cfg.get("min_clean_states", 4))
    hard_stop = (
        success_states < min_success
        or recovery_chunks < min_chunks
        or clean_n < min_clean
    )

    splits = episode_grouped_split(
        samples,
        seed=int(ds_cfg.get("split_seed", 2_026_080_405)),
        val_fraction=float(ds_cfg.get("val_episode_fraction", 0.25)),
    )
    train_eps = set(splits["train_episodes"])
    n_train_chunks = sum(
        1 for s in samples if (not s["clean_flag"]) and str(s["episode_id"]) in train_eps
    )

    _write(
        args.output_jsonl.resolve(),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
    )
    _write(
        args.flat_chunks_jsonl.resolve(),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in flat_chunks),
    )
    _write(
        args.splits_output.resolve(),
        json.dumps(splits, indent=2, sort_keys=True) + "\n",
    )

    qc = {
        "schema_version": "rase-pre-c1-1-dataset-qc/v1",
        "n_samples": len(samples),
        "n_recovery_chunks": recovery_chunks,
        "n_train_recovery_chunks": n_train_chunks,
        "n_clean_retention": clean_n,
        "n_attempted_failure_states": attempted,
        "n_successful_recovery_states": success_states,
        "n_failed_oft_states": failed_states,
        "missing_oft_teacher": missing_teacher,
        "min_successful_recovery_states": min_success,
        "min_train_chunks": min_chunks,
        "min_clean_states": min_clean,
        "hard_stop": hard_stop,
        "suite_counts": dict(Counter(str(s["suite"]) for s in samples)),
        "cell_counts": dict(Counter(str(s["cell"]) for s in samples)),
        "leakage_episode_overlap": splits["leakage_episode_overlap"],
        "splits": {
            "n_train_rows": splits["n_train_rows"],
            "n_val_rows": splits["n_val_rows"],
            "n_train_episodes": len(splits["train_episodes"]),
            "n_val_episodes": len(splits["val_episodes"]),
        },
    }
    _write(args.qc_json.resolve(), json.dumps(qc, indent=2, sort_keys=True) + "\n")
    md = [
        "# PRE-C1.1 distill dataset QC",
        "",
        f"- samples: {qc['n_samples']}",
        f"- successful recovery states: {qc['n_successful_recovery_states']}",
        f"- recovery chunks: {qc['n_recovery_chunks']}",
        f"- train recovery chunks: {qc['n_train_recovery_chunks']}",
        f"- failed OFT states (QC only): {qc['n_failed_oft_states']}",
        f"- clean retention: {qc['n_clean_retention']}",
        f"- hard_stop: `{qc['hard_stop']}`",
        f"- suite_counts: `{qc['suite_counts']}`",
        f"- leakage_episode_overlap: `{qc['leakage_episode_overlap']}`",
        "",
        "Only successful long-horizon OFT trajectories enter training.",
        "",
    ]
    _write(args.qc_md.resolve(), "\n".join(md))
    print(json.dumps(qc, sort_keys=True))
    if hard_stop:
        raise SystemExit(
            f"dataset hard-stop: success_states={success_states}<{min_success} "
            f"or chunks={recovery_chunks}<{min_chunks} or clean={clean_n}<{min_clean}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

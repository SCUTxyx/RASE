#!/usr/bin/env python3
"""Build PRE-C1 OFT→Smol recovery distill dataset + episode-grouped splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rase.adapt.pre_c1 import (
    DATASET_VERSION,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--oft-teacher-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--splits-output", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--qc-md", type=Path, required=True)
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock.resolve())
    ds_cfg = dict(lock["dataset"])
    rollouts = _load_rollouts(args.rollout_dir.resolve())
    teacher_dir = args.oft_teacher_dir.resolve()

    samples: list[dict[str, Any]] = []
    missing_teacher = 0
    for row in rollouts:
        current_fail = not bool(row.get("family_success", {}).get("current_suffix"))
        is_clean = (
            str(row.get("cell")) == str(ds_cfg.get("clean_retention_cell", "clean:L0"))
            and str(row.get("source_episode_outcome"))
            == str(ds_cfg.get("clean_retention_source_outcome", "success"))
        )
        if current_fail:
            teacher_path = teacher_dir / f"{row['state_key']}.json"
            if not teacher_path.is_file():
                missing_teacher += 1
                continue
            teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
            actions = teacher.get("teacher_actions") or []
            if not actions:
                missing_teacher += 1
                continue
            current = _current_arm(row)
            samples.append(
                {
                    "schema_version": DATASET_VERSION,
                    "state_key": row["state_key"],
                    "episode_id": row.get("episode_id"),
                    "task_id": row.get("task_id"),
                    "concrete_task_id": row.get("concrete_task_id"),
                    "suite": row.get("suite"),
                    "cell": row.get("cell"),
                    "stage": row.get("stage"),
                    "clean_flag": False,
                    "teacher_source": "oft",
                    "teacher_actions": actions,
                    "smol_failed_actions": (current or {}).get("action_tensor") or [],
                    "dual_track_label": "oft_teacher_on_current_failure",
                    "source_episode_outcome": row.get("source_episode_outcome"),
                }
            )
        elif is_clean:
            current = _current_arm(row)
            if current is None or not current.get("action_tensor"):
                continue
            # Retain target: base SmolVLA successful suffix (match base behavior).
            samples.append(
                {
                    "schema_version": DATASET_VERSION,
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
                    "smol_failed_actions": [],
                    "dual_track_label": "clean_retention",
                    "source_episode_outcome": row.get("source_episode_outcome"),
                }
            )

    recovery_n = sum(1 for s in samples if not s["clean_flag"])
    clean_n = sum(1 for s in samples if s["clean_flag"])
    min_rec = int(ds_cfg.get("min_recovery_states", 8))
    min_clean = int(ds_cfg.get("min_clean_states", 4))
    hard_stop = recovery_n < min_rec or clean_n < min_clean

    splits = episode_grouped_split(
        samples,
        seed=int(ds_cfg.get("split_seed", 2_026_080_405)),
        val_fraction=float(ds_cfg.get("val_episode_fraction", 0.25)),
    )
    _write(
        args.output_jsonl.resolve(),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
    )
    _write(
        args.splits_output.resolve(),
        json.dumps(splits, indent=2, sort_keys=True) + "\n",
    )

    from collections import Counter

    qc = {
        "schema_version": "rase-pre-c1-dataset-qc/v1",
        "n_samples": len(samples),
        "n_recovery": recovery_n,
        "n_clean_retention": clean_n,
        "missing_oft_teacher": missing_teacher,
        "min_recovery_states": min_rec,
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
        "# PRE-C1 distill dataset QC",
        "",
        f"- samples: {qc['n_samples']}",
        f"- recovery (OFT teacher): {qc['n_recovery']}",
        f"- clean retention: {qc['n_clean_retention']}",
        f"- missing OFT teacher files: {qc['missing_oft_teacher']}",
        f"- hard_stop: `{qc['hard_stop']}`",
        f"- suite_counts: `{qc['suite_counts']}`",
        f"- leakage_episode_overlap: `{qc['leakage_episode_overlap']}`",
        "",
        "Teacher naming: offline OFT recovery chunks; not runtime OFT.",
        "",
    ]
    _write(args.qc_md.resolve(), "\n".join(md))
    print(json.dumps(qc, sort_keys=True))
    if hard_stop:
        raise SystemExit(
            f"dataset hard-stop: recovery={recovery_n}<{min_rec} or clean={clean_n}<{min_clean}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

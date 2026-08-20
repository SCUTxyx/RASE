#!/usr/bin/env python3
"""Build one causal t8 feature row per audited R10-B K=3 group."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        value[int.from_bytes(digest[:4], "little") % dim] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def metadata_path(root: Path, row: dict) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    return (root / f"suite_{row['suite'].lower()}" / row["policy_id"]
            / f"seed_{row['seed_index']}" / "rep0" / f"{stem}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--repro-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.repro_audit.read_text())
    if manifest.get("status") != "frozen" or audit.get("status") != "PASS":
        raise ValueError("R10-B manifest/repro gate is not ready")
    if audit.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("R10-B audit is bound to another manifest")
    audited = {row["group_id"]: row for row in audit["records"]}
    rows = []
    for item in manifest["records"]:
        verdict = audited[item["group_id"]]
        path = metadata_path(args.collect_root, item)
        payload = json.loads(path.read_text())
        by_elapsed = {int(row["elapsed_source_steps"]): (index, row)
                      for index, row in enumerate(payload["rows"])}
        index, boundary = by_elapsed[8]
        with np.load(payload["npz"], allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        image_history = arrays["temporal_image_history"][index].astype(np.uint8)
        proprio_history = arrays["temporal_proprio_history"][index].astype(np.float32)
        action_history = arrays["temporal_action_history"][index].astype(np.float32)
        if image_history.shape[0] != 8 or proprio_history.shape[0] != 8 or action_history.shape[0] != 8:
            raise ValueError(f"unexpected temporal history: {path}")
        rows.append({
            **{key: item[key] for key in ("group_id", "state_key", "task_id", "suite",
                                          "policy_id", "seed_index", "outer_fold")},
            "cohort_role": "case_control_development",
            "instruction": boundary["instruction"],
            "hazard_label": int(verdict["label_k3"]),
            "source_failure": int(not bool(payload["source_success"])),
            "t8_teacher_steps": float(boundary["persistent_teacher_steps_if_enter_now"] or 0.0),
            "image": arrays["image"][index].astype(np.uint8),
            "proprio": arrays["proprio"][index].astype(np.float32),
            "action_summary": arrays["source_action_summary"][index].astype(np.float32),
            "image_history": image_history,
            "proprio_history": proprio_history,
            "proprio_delta_history": np.diff(
                proprio_history, axis=0, prepend=proprio_history[:1]).astype(np.float32),
            "proprio_accel_history": np.diff(
                np.diff(proprio_history, axis=0, prepend=proprio_history[:1]),
                axis=0, prepend=np.zeros_like(proprio_history[:1])).astype(np.float32),
            "action_history": action_history,
            "action_delta_history": np.diff(
                action_history, axis=0, prepend=action_history[:1]).astype(np.float32),
        })
    rows.sort(key=lambda row: (row["outer_fold"], row["task_id"], row["group_id"]))
    policy_order = sorted({row["policy_id"] for row in rows})
    arrays = {
        "image": np.stack([row["image"] for row in rows]),
        "proprio": np.stack([row["proprio"] for row in rows]),
        "action_summary": np.stack([row["action_summary"] for row in rows]),
        "image_history": np.stack([row["image_history"] for row in rows]),
        "proprio_history": np.stack([row["proprio_history"] for row in rows]),
        "proprio_delta_history": np.stack([row["proprio_delta_history"] for row in rows]),
        "proprio_accel_history": np.stack([row["proprio_accel_history"] for row in rows]),
        "action_history": np.stack([row["action_history"] for row in rows]),
        "action_delta_history": np.stack([row["action_delta_history"] for row in rows]),
        "language_hash": np.stack([hashed_instruction(row["instruction"]) for row in rows]),
        "hazard_label": np.asarray([row["hazard_label"] for row in rows], np.float32),
        "source_failure": np.asarray([row["source_failure"] for row in rows], np.float32),
        "t8_teacher_steps": np.asarray([row["t8_teacher_steps"] for row in rows], np.float32),
        "policy_index": np.asarray([policy_order.index(row["policy_id"]) for row in rows], np.int64),
        "outer_fold": np.asarray([row["outer_fold"] for row in rows], np.int64),
        "seed_index": np.asarray([row["seed_index"] for row in rows], np.int64),
        "policy_order": np.asarray(policy_order),
    }
    for field in ("group_id", "state_key", "task_id", "suite", "policy_id",
                  "cohort_role", "instruction"):
        arrays[field] = np.asarray([row[field] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "schema_version": "rase-r10b-case-control-dataset/v1", "status": "complete",
        "scientific_scope": "label-balanced representation development only",
        "dataset_sha256": sha256(args.output), "manifest_sha256": sha256(args.manifest),
        "repro_audit_sha256": sha256(args.repro_audit), "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "labels": {str(label): sum(row["hazard_label"] == label for row in rows)
                   for label in (0, 1)},
        "policies": {policy: sum(row["policy_id"] == policy for row in rows)
                     for policy in policy_order},
        "history": 8, "decision_boundary": 8, "target_boundary": 16,
        "forbidden_features": ["future frames", "OFT action", "K2 selection label",
                               "task ID", "suite ID", "simulator object state"],
    }
    args.output.with_suffix(".npz.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

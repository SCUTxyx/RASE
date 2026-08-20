#!/usr/bin/env python3
"""Build the frozen causal temporal R9-B transition dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        value[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--repro-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.repro_audit.read_text())
    if manifest.get("status") != "frozen" or audit.get("status") != "PASS":
        raise ValueError("R9-B manifest or reproducibility gate is not PASS")
    if audit.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("R9-B audit is bound to a different manifest")
    if args.history < 1 or args.history > 16:
        raise ValueError("history must be in [1,16]")

    rows = []
    for item in manifest["records"]:
        for policy in manifest["source_policies"]:
            seed = 4 if policy == "pi05_libero" else 2
            for replica in range(3):
                stem = f"{item['state_key']}__seed{seed}"
                if replica:
                    stem += f"__rep{replica}"
                metadata_path = (args.collect_root / f"suite_{item['suite'].lower()}" / policy
                                 / f"rep{replica}" / f"{stem}.json")
                payload = json.loads(metadata_path.read_text())
                npz_path = Path(payload["npz"])
                with np.load(npz_path, allow_pickle=False) as loaded:
                    arrays = {key: loaded[key] for key in loaded.files}
                by_elapsed = {int(row["elapsed_source_steps"]): (index, row)
                              for index, row in enumerate(payload["rows"])}
                # Successful episodes may terminate before the next planned
                # boundary.  Keep every boundary that was actually observed;
                # the transition pass below naturally drops starts without a
                # following boundary.
                for elapsed in sorted(by_elapsed):
                    if elapsed not in (0, 4, 8, 12, 16):
                        raise ValueError(f"unexpected boundary {elapsed}: {metadata_path}")
                    index, boundary = by_elapsed[elapsed]
                    row = {
                        "state_key": item["state_key"], "task_id": item["task_id"],
                        "suite": item["suite"], "policy_id": policy,
                        "group_id": f"{item['state_key']}:{policy}:seed{seed}:rep{replica}",
                        "base_group_id": f"{item['state_key']}:{policy}:seed{seed}",
                        "replicate_index": replica, "cohort_role": item["role"],
                        "perturb_dim": item["perturb_dim"],
                        "perturb_level": item["perturb_level"],
                        "elapsed_source_steps": elapsed,
                        "instruction": boundary["instruction"],
                        "source_success": int(bool(payload["source_success"])),
                        "source_final_success": int(bool(boundary["source_final_success"])),
                        "persistent_success": int(bool(boundary["persistent_success_if_enter_now"])),
                        "persistent_teacher_steps": float(boundary["persistent_teacher_steps_if_enter_now"] or 0.0),
                    }
                    for key in ("image", "proprio", "source_action_summary",
                                "temporal_image_history", "temporal_proprio_history",
                                "temporal_action_history"):
                        if key not in arrays:
                            raise ValueError(f"missing R9 temporal array {key}: {npz_path}")
                    row["image"] = arrays["image"][index].astype(np.uint8)
                    row["proprio"] = arrays["proprio"][index].astype(np.float32)
                    row["action_summary"] = arrays["source_action_summary"][index].astype(np.float32)
                    row["image_history"] = arrays["temporal_image_history"][index].astype(np.uint8)
                    row["proprio_history"] = arrays["temporal_proprio_history"][index].astype(np.float32)
                    row["action_history"] = arrays["temporal_action_history"][index].astype(np.float32)
                    row["proprio_delta_history"] = np.diff(
                        row["proprio_history"], axis=0, prepend=row["proprio_history"][:1]
                    ).astype(np.float32)
                    rows.append(row)

    # Transition labels are attached only to starts t={0,4,8,12}; t=16 remains
    # available as a terminal boundary but has no future target.
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    transitions = []
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: row["elapsed_source_steps"])
        by_elapsed = {row["elapsed_source_steps"]: row for row in group_rows}
        for elapsed in (0, 4, 8, 12):
            if elapsed not in by_elapsed or elapsed + 4 not in by_elapsed:
                continue
            current = by_elapsed[elapsed]
            following = by_elapsed[elapsed + 4]
            current["current_recoverable"] = current["persistent_success"]
            current["next_recoverable"] = following["persistent_success"]
            current["loss_hazard"] = int(
                current["persistent_success"] == 1 and following["persistent_success"] == 0
            )
            transitions.append(current)
    if not transitions:
        raise ValueError("no R9-B transitions")
    transitions.sort(key=lambda row: (row["task_id"], row["state_key"], row["policy_id"],
                                      row["replicate_index"], row["elapsed_source_steps"]))
    n = len(transitions)
    string_fields = ("state_key", "task_id", "suite", "policy_id", "group_id",
                     "base_group_id", "cohort_role", "perturb_dim", "instruction")
    policy_order = sorted({row["policy_id"] for row in transitions})
    output_arrays = {
        "image": np.stack([row["image"] for row in transitions]),
        "proprio": np.stack([row["proprio"] for row in transitions]),
        "action_summary": np.stack([row["action_summary"] for row in transitions]),
        "image_history": np.stack([row["image_history"] for row in transitions]),
        "proprio_history": np.stack([row["proprio_history"] for row in transitions]),
        "proprio_delta_history": np.stack([row["proprio_delta_history"] for row in transitions]),
        "action_history": np.stack([row["action_history"] for row in transitions]),
        "language_hash": np.stack([hashed_instruction(row["instruction"]) for row in transitions]),
        "elapsed_source_steps": np.asarray([row["elapsed_source_steps"] for row in transitions], np.int32),
        "policy_index": np.asarray([policy_order.index(row["policy_id"]) for row in transitions], np.int64),
        "replicate_index": np.asarray([row["replicate_index"] for row in transitions], np.int64),
        "perturb_level": np.asarray([row["perturb_level"] for row in transitions], np.int32),
        "persistent_success": np.asarray([row["persistent_success"] for row in transitions], np.float32),
        "persistent_teacher_steps": np.asarray([row["persistent_teacher_steps"] for row in transitions], np.float32),
        "source_success": np.asarray([row["source_success"] for row in transitions], np.float32),
        "current_recoverable": np.asarray([row["current_recoverable"] for row in transitions], np.float32),
        "next_recoverable": np.asarray([row["next_recoverable"] for row in transitions], np.float32),
        "loss_hazard": np.asarray([row["loss_hazard"] for row in transitions], np.float32),
    }
    for field in string_fields:
        output_arrays[field] = np.asarray([row[field] for row in transitions])
    output_arrays["policy_order"] = np.asarray(policy_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_arrays)
    report = {
        "schema_version": "rase-r9b-temporal-dataset/v1",
        "status": "complete", "scientific_scope": "development temporal transitions",
        "dataset_sha256": sha256(args.output), "manifest_sha256": sha256(args.manifest),
        "repro_audit_sha256": sha256(args.repro_audit), "rows": n,
        "groups": len(set(row["group_id"] for row in transitions)),
        "base_groups": len(set(row["base_group_id"] for row in transitions)),
        "tasks": len(set(row["task_id"] for row in transitions)),
        "policies": policy_order, "boundaries": [0, 4, 8, 12],
        "temporal_history": args.history,
        "hazard_positives": int(sum(row["loss_hazard"] for row in transitions)),
        "hazard_prevalence": float(np.mean(output_arrays["loss_hazard"])),
        "forbidden_features": ["OFT actions", "future frames", "teacher labels", "task ordinal"],
        "feature_policy": "causal four-step image/proprio/action history plus current proposal",
    }
    args.output.with_suffix(".npz.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

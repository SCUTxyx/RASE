#!/usr/bin/env python3
"""Build an exploratory K=3 probabilistic hazard dataset after R10-B FAIL.

R10-B's deterministic case/control gate remains failed.  This builder does not
drop unstable groups or relabel the frozen manifest.  It preserves all 66
groups and represents the independent K=3 outcomes as binomial counts:

    hazard = success_if_enter_at_t8 and failure_if_enter_at_t16.

The resulting artifact is diagnostic only.  It cannot unlock a model,
selector, validation, or test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        value[int.from_bytes(digest[:4], "little") % dim] += (
            1.0 if digest[4] & 1 else -1.0
        )
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def metadata_path(root: Path, row: dict) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    return (
        root
        / f"suite_{row['suite'].lower()}"
        / row["policy_id"]
        / f"seed_{row['seed_index']}"
        / "rep0"
        / f"{stem}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--repro-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.repro_audit.read_text())
    if manifest.get("status") != "frozen":
        raise ValueError("R10-B manifest is not frozen")
    if audit.get("status") != "FAIL" or audit.get("decision") != "STOP_R10B_PROTOCOL":
        raise ValueError("this diagnostic is only valid after the frozen R10-B FAIL")
    if audit.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("R10-B audit is bound to another manifest")
    if len(audit.get("records", [])) != manifest.get("expected_groups"):
        raise ValueError("R10-B audit does not contain every frozen group")
    if not all(bool(row.get("t8_feature_parity")) for row in audit["records"]):
        raise ValueError("causal t8 features do not have replica parity")

    audited = {row["group_id"]: row for row in audit["records"]}
    rows = []
    for item in manifest["records"]:
        verdict = audited[item["group_id"]]
        path = metadata_path(args.collect_root, item)
        payload = json.loads(path.read_text())
        by_elapsed = {
            int(row["elapsed_source_steps"]): (index, row)
            for index, row in enumerate(payload["rows"])
        }
        index, boundary = by_elapsed[8]
        with np.load(payload["npz"], allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}

        image_history = arrays["temporal_image_history"][index].astype(np.uint8)
        proprio_history = arrays["temporal_proprio_history"][index].astype(np.float32)
        action_history = arrays["temporal_action_history"][index].astype(np.float32)
        if (
            image_history.shape[0] != 8
            or proprio_history.shape[0] != 8
            or action_history.shape[0] != 8
        ):
            raise ValueError(f"unexpected temporal history: {path}")

        t8 = np.asarray(verdict["t8_labels_k3"], dtype=np.int64)
        t16 = np.asarray(verdict["t16_labels_k3"], dtype=np.int64)
        if t8.shape != (3,) or t16.shape != (3,):
            raise ValueError(f"expected three outcomes for {item['group_id']}")
        hazard = np.logical_and(t8 == 1, t16 == 0).astype(np.int64)
        rows.append(
            {
                **{
                    key: item[key]
                    for key in (
                        "group_id",
                        "state_key",
                        "task_id",
                        "suite",
                        "policy_id",
                        "seed_index",
                        "outer_fold",
                    )
                },
                "cohort_role": "post_r10b_exploratory_probability_diagnostic",
                "instruction": boundary["instruction"],
                "selection_label_k2": int(item["hazard_label_k2"]),
                "hazard_successes_k3": int(hazard.sum()),
                "hazard_trials_k3": 3,
                "t8_successes_k3": int(t8.sum()),
                "t16_successes_k3": int(t16.sum()),
                "source_failure": int(not bool(payload["source_success"])),
                "image": arrays["image"][index].astype(np.uint8),
                "proprio": arrays["proprio"][index].astype(np.float32),
                "action_summary": arrays["source_action_summary"][index].astype(
                    np.float32
                ),
                "image_history": image_history,
                "proprio_history": proprio_history,
                "proprio_delta_history": np.diff(
                    proprio_history, axis=0, prepend=proprio_history[:1]
                ).astype(np.float32),
                "proprio_accel_history": np.diff(
                    np.diff(
                        proprio_history, axis=0, prepend=proprio_history[:1]
                    ),
                    axis=0,
                    prepend=np.zeros_like(proprio_history[:1]),
                ).astype(np.float32),
                "action_history": action_history,
                "action_delta_history": np.diff(
                    action_history, axis=0, prepend=action_history[:1]
                ).astype(np.float32),
            }
        )

    rows.sort(key=lambda row: (row["outer_fold"], row["task_id"], row["group_id"]))
    policy_order = sorted({row["policy_id"] for row in rows})
    arrays = {
        "image": np.stack([row["image"] for row in rows]),
        "proprio": np.stack([row["proprio"] for row in rows]),
        "action_summary": np.stack([row["action_summary"] for row in rows]),
        "image_history": np.stack([row["image_history"] for row in rows]),
        "proprio_history": np.stack([row["proprio_history"] for row in rows]),
        "proprio_delta_history": np.stack(
            [row["proprio_delta_history"] for row in rows]
        ),
        "proprio_accel_history": np.stack(
            [row["proprio_accel_history"] for row in rows]
        ),
        "action_history": np.stack([row["action_history"] for row in rows]),
        "action_delta_history": np.stack(
            [row["action_delta_history"] for row in rows]
        ),
        "language_hash": np.stack(
            [hashed_instruction(row["instruction"]) for row in rows]
        ),
        "selection_label_k2": np.asarray(
            [row["selection_label_k2"] for row in rows], np.int64
        ),
        "hazard_successes_k3": np.asarray(
            [row["hazard_successes_k3"] for row in rows], np.float32
        ),
        "hazard_trials_k3": np.asarray(
            [row["hazard_trials_k3"] for row in rows], np.float32
        ),
        "t8_successes_k3": np.asarray(
            [row["t8_successes_k3"] for row in rows], np.float32
        ),
        "t16_successes_k3": np.asarray(
            [row["t16_successes_k3"] for row in rows], np.float32
        ),
        "source_failure": np.asarray(
            [row["source_failure"] for row in rows], np.float32
        ),
        "policy_index": np.asarray(
            [policy_order.index(row["policy_id"]) for row in rows], np.int64
        ),
        "outer_fold": np.asarray([row["outer_fold"] for row in rows], np.int64),
        "seed_index": np.asarray([row["seed_index"] for row in rows], np.int64),
        "policy_order": np.asarray(policy_order),
    }
    for field in (
        "group_id",
        "state_key",
        "task_id",
        "suite",
        "policy_id",
        "cohort_role",
        "instruction",
    ):
        arrays[field] = np.asarray([row[field] for row in rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    event_groups = sum(row["hazard_successes_k3"] > 0 for row in rows)
    event_tasks = len(
        {row["task_id"] for row in rows if row["hazard_successes_k3"] > 0}
    )
    report = {
        "schema_version": "rase-r10b-probabilistic-diagnostic-dataset/v1",
        "status": "complete_exploratory",
        "scientific_scope": "post-gate-failure probability diagnostic only",
        "not_valid_for": [
            "formal R10 gate",
            "model selection",
            "prevalence",
            "calibration",
            "selector",
            "validation",
            "test",
        ],
        "dataset_sha256": sha256(args.output),
        "manifest_sha256": sha256(args.manifest),
        "repro_audit_sha256": sha256(args.repro_audit),
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "hazard_events": int(sum(row["hazard_successes_k3"] for row in rows)),
        "hazard_trials": int(sum(row["hazard_trials_k3"] for row in rows)),
        "hazard_positive_groups": int(event_groups),
        "hazard_positive_tasks": int(event_tasks),
        "hazard_count_distribution": dict(
            sorted(Counter(row["hazard_successes_k3"] for row in rows).items())
        ),
        "policies": dict(sorted(Counter(row["policy_id"] for row in rows).items())),
        "suites": dict(sorted(Counter(row["suite"] for row in rows).items())),
        "history": 8,
        "decision_boundary": 8,
        "target_boundary": 16,
        "forbidden_model_features": [
            "selection_label_k2",
            "future frames",
            "OFT action",
            "task ID",
            "suite ID",
            "simulator object state",
        ],
    }
    report_path = args.output.with_suffix(".npz.report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

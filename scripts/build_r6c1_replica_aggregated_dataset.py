#!/usr/bin/env python3
"""Build the replica-aggregated R6-C.1 candidate-arm dataset.

One row is emitted per (policy, seed, state, boundary), never per replica.
Deployment features come from canonical rep0.  Counterfactual replicas are
aggregated into empirical success counts/trials and teacher-cost quantiles.
This prevents repeated rollouts from inflating OOF weight while enabling the
beta-binomial success loss and quantile cost heads required by R6-C.1C.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_r6c1b_label_support import load_groups  # noqa: E402
from build_r6c_dynamic_dataset import (  # noqa: E402
    ARMS,
    HORIZON,
    arm_schema,
    hashed_instruction,
    load_parity_reference,
    recent_action_history,
    sha256,
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def build_dataset(*, input_roots: list[Path], protocol: Path, output: Path,
                  exclusions: Path, atlas_root: Path | None = None,
                  history_window: int = 4,
                  decision_boundaries: tuple[int, ...] = (0, 8, 16)) -> dict:
    protocol_data = read_json(protocol.resolve())
    if protocol_data.get("schema_version") != "rase-r6b1-dynamic-boundary-protocol/v1":
        raise ValueError("unexpected protocol schema")
    qualified = {
        row["policy_id"]: set(int(v) for v in row["dynamic_seed_indices"])
        for row in protocol_data["qualified_source_policies"]
    }
    groups = load_groups(input_roots, exclusions)
    references = load_parity_reference(atlas_root) if atlas_root is not None else {}

    rows = []
    images = []
    proprios = []
    action_summaries = []
    histories = []
    parity_checked = 0
    parity_failures = []
    for group in groups:
        policy = group["policy_id"]
        seed = int(group["seed_index"])
        if policy not in qualified or seed not in qualified[policy]:
            raise ValueError(f"unexpected qualified pair {policy}:{seed}")
        metadata_path = Path(group["canonical_path"])
        metadata = read_json(metadata_path)
        npz_path = Path(metadata["npz"])
        if not npz_path.exists():
            npz_path = metadata_path.with_suffix(".npz")
        data = np.load(npz_path)
        trace = data["source_action_trace"].astype(np.float32)
        positions = {
            int(row["elapsed_source_steps"]): position
            for position, row in enumerate(metadata["rows"])
        }
        reference = references.get((policy, seed, group["state_key"]))
        if reference is not None:
            first = metadata["rows"][0]
            parity_checked += 1
            observed = {
                "rollout_seed": int(first["rollout_seed"]),
                "source_final_success": bool(first["source_final_success"]),
                "source_total_steps": int(first["source_total_steps"]),
            }
            expected = {
                "rollout_seed": int(reference["rollout_seed"]),
                "source_final_success": bool(reference["success"]),
                "source_total_steps": int(reference["env_steps"]),
            }
            if observed != expected:
                parity_failures.append({"key": group["key"], "observed": observed,
                                        "expected": expected})
                continue
        for elapsed in decision_boundaries:
            boundary = group["boundaries"].get(elapsed)
            position = positions.get(elapsed)
            if boundary is None or position is None:
                continue
            canonical_row = metadata["rows"][position]
            n_trials = int(boundary["trials"])
            source_successes = int(bool(group["source_success"])) * n_trials
            cost_values = np.asarray(boundary["teacher_steps"], dtype=np.float32)
            cost_quantiles = np.quantile(cost_values, [0.1, 0.5, 0.9]).astype(np.float32)
            rows.append({
                "state_key": group["state_key"],
                "task_id": group["task_id"],
                "suite": group["suite"],
                "policy_id": policy,
                "seed_index": seed,
                "group_id": f"{group['state_key']}:{policy}:seed{seed}",
                "cohort_role": group["cohort_role"],
                "elapsed_source_steps": elapsed,
                "instruction": group["instruction"],
                "source_success": float(group["source_success"]),
                "source_successes": source_successes,
                "source_trials": n_trials,
                "source_within_8": bool(canonical_row["source_success_within_8"]),
                "source_within_16": bool(canonical_row["source_success_within_16"]),
                "source_within_32": bool(canonical_row["source_success_within_32"]),
                "persistent_probability": float(boundary["success_probability"]),
                "persistent_successes": int(boundary["successes"]),
                "persistent_trials": n_trials,
                "persistent_cost_quantiles": cost_quantiles,
            })
            images.append(data["image"][position].astype(np.uint8))
            proprios.append(data["proprio"][position].astype(np.float32))
            action_summaries.append(
                data["source_action_summary"][position].astype(np.float32)
            )
            histories.append(recent_action_history(trace, elapsed, history_window))
    if parity_failures:
        raise ValueError(f"source parity failed for {len(parity_failures)} groups: "
                         f"{json.dumps(parity_failures[:5], indent=2)}")
    if not rows:
        raise ValueError("no replica-aggregated decision rows")

    order = sorted(range(len(rows)), key=lambda i: (
        rows[i]["group_id"], rows[i]["elapsed_source_steps"]
    ))
    rows = [rows[i] for i in order]
    images = [images[i] for i in order]
    proprios = [proprios[i] for i in order]
    action_summaries = [action_summaries[i] for i in order]
    histories = [histories[i] for i in order]
    policy_order = sorted({row["policy_id"] for row in rows})
    elapsed = np.asarray([row["elapsed_source_steps"] for row in rows], dtype=np.int32)
    source_probability = np.asarray([row["source_success"] for row in rows], dtype=np.float32)
    persistent_probability = np.asarray(
        [row["persistent_probability"] for row in rows], dtype=np.float32
    )
    arm_success = np.stack([source_probability, persistent_probability], axis=-1)
    source_successes = np.asarray([row["source_successes"] for row in rows], dtype=np.float32)
    source_trials = np.asarray([row["source_trials"] for row in rows], dtype=np.float32)
    persistent_successes = np.asarray(
        [row["persistent_successes"] for row in rows], dtype=np.float32
    )
    persistent_trials = np.asarray(
        [row["persistent_trials"] for row in rows], dtype=np.float32
    )
    arm_successes = np.stack([source_successes, persistent_successes], axis=-1)
    arm_trials = np.stack([source_trials, persistent_trials], axis=-1)
    persistent_quantiles = np.stack(
        [row["persistent_cost_quantiles"] for row in rows]
    ).astype(np.float32)
    zeros = np.zeros_like(persistent_quantiles)
    arm_cost_quantiles = np.stack([zeros, persistent_quantiles], axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=np.stack(images),
        proprio=np.stack(proprios),
        action_summary=np.stack(action_summaries),
        history=np.stack(histories),
        elapsed_progress=elapsed.astype(np.float32) / float(HORIZON),
        elapsed_source_steps=elapsed,
        instruction=np.asarray([row["instruction"] for row in rows]),
        language_hash=np.stack([hashed_instruction(row["instruction"]) for row in rows]),
        state_key=np.asarray([row["state_key"] for row in rows]),
        task_id=np.asarray([row["task_id"] for row in rows]),
        suite=np.asarray([row["suite"] for row in rows]),
        group_id=np.asarray([row["group_id"] for row in rows]),
        base_group_id=np.asarray([row["group_id"] for row in rows]),
        replicate_index=np.zeros(len(rows), dtype=np.int64),
        replica_count=source_trials.astype(np.int64),
        cohort_role=np.asarray([row["cohort_role"] for row in rows]),
        policy_id=np.asarray([row["policy_id"] for row in rows]),
        policy_index=np.asarray(
            [policy_order.index(row["policy_id"]) for row in rows], dtype=np.int64
        ),
        source_success=source_probability,
        source_successes=source_successes,
        source_trials=source_trials,
        source_within_8=np.asarray([row["source_within_8"] for row in rows], dtype=np.float32),
        source_within_16=np.asarray([row["source_within_16"] for row in rows], dtype=np.float32),
        source_within_32=np.asarray([row["source_within_32"] for row in rows], dtype=np.float32),
        persistent_success=persistent_probability,
        persistent_successes=persistent_successes,
        persistent_trials=persistent_trials,
        persistent_teacher_steps=persistent_quantiles[:, 1],
        arm_success=arm_success,
        arm_successes=arm_successes,
        arm_trials=arm_trials,
        arm_teacher_steps=arm_cost_quantiles[:, :, 1],
        arm_teacher_step_quantiles=arm_cost_quantiles,
        arm_ids=np.asarray([arm["index"] for arm in ARMS], dtype=np.int64),
    )
    report = {
        "schema_version": "rase-r6c1-replica-aggregated-dataset/v1",
        "status": "complete",
        "scientific_scope": (
            "one row per policy/seed/state/boundary; replica probabilities and cost "
            "quantiles; natural OOF and enrichment training-only"
        ),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256(output),
        "input_roots": [str(path.resolve()) for path in input_roots],
        "protocol": str(protocol.resolve()),
        "protocol_sha256": sha256(protocol.resolve()),
        "exclusions": str(exclusions.resolve()),
        "exclusions_sha256": sha256(exclusions.resolve()),
        "decision_boundaries": list(decision_boundaries),
        "history_window": history_window,
        "n_rows": len(rows),
        "n_groups": len({row["group_id"] for row in rows}),
        "n_tasks": len({row["task_id"] for row in rows}),
        "n_states": len({row["state_key"] for row in rows}),
        "policies": policy_order,
        "cohort_counts": {
            role: len({row["group_id"] for row in rows if row["cohort_role"] == role})
            for role in sorted({row["cohort_role"] for row in rows})
        },
        "replica_count_distribution_by_group": {
            str(value): len({
                row["group_id"] for row in rows if row["source_trials"] == value
            })
            for value in sorted({row["source_trials"] for row in rows})
        },
        "replica_count_distribution_by_row": {
            str(value): int(sum(row["source_trials"] == value for row in rows))
            for value in sorted({row["source_trials"] for row in rows})
        },
        "parity_reference_root": str(atlas_root.resolve()) if atlas_root else None,
        "parity_checked_groups": parity_checked,
        "parity_failures": 0,
        "arm_schema": arm_schema(rows),
        "label_policy": (
            "arm_successes/arm_trials train beta-binomial heads; "
            "arm_teacher_step_quantiles train q10/q50/q90 heads"
        ),
        "forbidden_features": [
            "suite", "task ordinal", "future outcome", "counterfactual OFT result"
        ],
    }
    output.with_suffix(".npz.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, default=None)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--decision-boundary", type=int, action="append",
                        default=None)
    args = parser.parse_args()
    boundaries = tuple(args.decision_boundary or (0, 8, 16))
    report = build_dataset(
        input_roots=args.input_root,
        protocol=args.protocol,
        output=args.output,
        exclusions=args.exclusions,
        atlas_root=args.atlas_root,
        history_window=args.history_window,
        decision_boundaries=boundaries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

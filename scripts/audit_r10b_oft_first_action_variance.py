#!/usr/bin/env python3
"""Decompose R10-B label flips using the saved first OFT action per boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path(root: Path, row: dict, replica: int) -> Path:
    stem = f"{row['state_key']}__seed{row['seed_index']}"
    if replica:
        stem += f"__rep{replica}"
    return (
        root
        / f"suite_{row['suite'].lower()}"
        / row["policy_id"]
        / f"seed_{row['seed_index']}"
        / f"rep{replica}"
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
    if audit.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("audit/manifest hash mismatch")
    audited = {row["group_id"]: row for row in audit["records"]}
    records = []
    for item in manifest["records"]:
        actions_by_boundary: dict[int, list[np.ndarray]] = {8: [], 16: []}
        for replica in range(3):
            path = metadata_path(args.collect_root, item, replica)
            payload = json.loads(path.read_text())
            elapsed = [int(row["elapsed_source_steps"]) for row in payload["rows"]]
            with np.load(payload["npz"], allow_pickle=False) as loaded:
                oft_actions = loaded["oft_action"]
                for boundary in actions_by_boundary:
                    actions_by_boundary[boundary].append(
                        np.asarray(oft_actions[elapsed.index(boundary)], dtype=np.float64)
                    )
        verdict = audited[item["group_id"]]
        boundary_summary = {}
        for boundary, actions in actions_by_boundary.items():
            stack = np.stack(actions)
            exact = all(np.array_equal(stack[0], stack[index]) for index in (1, 2))
            close = all(
                np.allclose(stack[0], stack[index], rtol=0.0, atol=1e-6)
                for index in (1, 2)
            )
            boundary_summary[str(boundary)] = {
                "exact": bool(exact),
                "allclose_1e_6": bool(close),
                "max_abs_pairwise": float(
                    max(
                        np.max(np.abs(stack[left] - stack[right]))
                        for left, right in ((0, 1), (0, 2), (1, 2))
                    )
                ),
            }
        outcome_stable = bool(verdict["k3_stable"])
        records.append(
            {
                "group_id": item["group_id"],
                "task_id": item["task_id"],
                "suite": item["suite"],
                "policy_id": item["policy_id"],
                "outcome_stable": outcome_stable,
                "k2_match": bool(verdict["k3_matches_k2"]),
                "t8_outcomes": verdict["t8_labels_k3"],
                "t16_outcomes": verdict["t16_labels_k3"],
                "oft_first_action": boundary_summary,
            }
        )

    t8_exact = [row["oft_first_action"]["8"]["exact"] for row in records]
    t16_exact = [row["oft_first_action"]["16"]["exact"] for row in records]
    outcome_stable = [row["outcome_stable"] for row in records]
    contingency = Counter(
        ("action_same" if action else "action_diff", "outcome_stable" if stable else "outcome_flip")
        for action, stable in zip(t8_exact, outcome_stable, strict=True)
    )
    result = {
        "schema_version": "rase-r10b-oft-first-action-variance/v1",
        "status": "complete_diagnostic",
        "scientific_scope": "post-R10B-failure root-cause diagnostic",
        "manifest_sha256": sha256(args.manifest),
        "repro_audit_sha256": sha256(args.repro_audit),
        "groups": len(records),
        "t8_first_action_exact_groups": int(sum(t8_exact)),
        "t8_first_action_different_groups": int(len(records) - sum(t8_exact)),
        "t16_first_action_exact_groups": int(sum(t16_exact)),
        "t16_first_action_different_groups": int(len(records) - sum(t16_exact)),
        "t8_action_outcome_contingency": {
            f"{action}|{outcome}": count
            for (action, outcome), count in sorted(contingency.items())
        },
        "max_t8_first_action_difference": float(
            max(row["oft_first_action"]["8"]["max_abs_pairwise"] for row in records)
        ),
        "max_t16_first_action_difference": float(
            max(row["oft_first_action"]["16"]["max_abs_pairwise"] for row in records)
        ),
        "interpretation": (
            "First-action differences implicate oracle input/inference state. "
            "Outcome flips with identical first actions require full OFT-trace "
            "or fixed-action replay to separate later policy and environment variance."
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in result if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

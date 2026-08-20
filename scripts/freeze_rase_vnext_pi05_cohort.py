#!/usr/bin/env python3
"""B1: freeze the pi0.5 cross-policy challenge cohort.

Reuses the exact pi0-fast K3 roots/tasks (same physical state pool) for paired
cross-policy comparison; only the policy id and seed salts change.  This is a
cross-policy challenge, NOT a new independent external confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(salt: str, *parts: object) -> int:
    token = (salt + "\x1f" + "\x1f".join(str(part) for part in parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


K3_OPERATORS = (
    "continue.source", "requery.source",
    "resample.source/candidate.0", "resample.source/candidate.1",
    "fallback.persistent", "abort.safe",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k3-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", default="pi05.libero")
    parser.add_argument("--salt", default="rase-vnext-pi05-challenge-v1")
    args = parser.parse_args()

    k3 = json.loads(args.k3_manifest.read_text())
    if k3.get("status") != "frozen_confirmation":
        raise SystemExit("K3 manifest is not frozen")
    if sha256(args.protocol) != k3.get("protocol_sha256"):
        raise SystemExit("protocol hash mismatch")

    roots = sorted(k3["roots"], key=lambda r: (str(r["suite"]), str(r["task_id"]), str(r["root_id"])))
    jobs: list[dict] = []
    for record in roots:
        for operator in K3_OPERATORS:
            for replica in range(3):
                job_id = stable_seed(args.salt, record["root_id"], operator, replica, "job")
                jobs.append({
                    "available_by_contract": True,
                    "candidate_ids": [],
                    "collection_phase": "pi05_challenge",
                    "contract_mask_reason": None,
                    "decision_point": {
                        "decision_point_id": "source.step.8",
                        "rule": "source_elapsed_step",
                        "value": 8,
                    },
                    "job_id": f"{job_id:012x}",
                    "operator_id": operator,
                    "operator_kind": operator.split("/")[0],
                    "outer_fold": int(k3["task_folds"][str(record["task_id"])]),
                    "policy_id": args.policy_id,
                    "restore_state_ref": str(record["restore_state_ref"]),
                    "root_id": str(record["root_id"]),
                    "seed_ledger": {
                        "environment_seed": int(record["environment_seed"]),
                        "exact_repeat_replica": int(replica),
                        "execution_seed": stable_seed(
                            args.salt, record["root_id"], "exec", replica,
                        ),
                        "init_state_id": int(record["init_state_id"]),
                        "operator_seed": stable_seed(
                            args.salt, record["root_id"], operator, replica,
                        ),
                        "source_sampling_seed": stable_seed(
                            args.salt, record["root_id"], "prefix", replica,
                        ),
                    },
                    "state_key": str(record["state_key"]),
                    "suite": str(record["suite"]),
                    "task_id": str(record["task_id"]),
                })

    manifest = {
        "schema_version": "rase-vnext-pi05-challenge-manifest/v1",
        "status": "frozen_confirmation",
        "scientific_scope": (
            "CROSS_POLICY_CHALLENGE_NOT_EXTERNAL_CONFIRMATION: pi0.5 on the exact "
            "pi0-fast K3 roots/tasks for paired comparison; policy id and seed "
            "salts are the only changes"
        ),
        "parent_k3_manifest": str(args.k3_manifest.resolve()),
        "parent_k3_manifest_sha256": sha256(args.k3_manifest),
        "protocol_sha256": sha256(args.protocol),
        "root_catalog_pool": str(k3["root_catalog_pool"]),
        "selection_salt": args.salt,
        "selection_rule": (
            "reuse K3 cohort roots/tasks verbatim (paired cross-policy "
            "comparison); no outcome read"
        ),
        "decision_point": {"decision_point_id": "source.step.8", "rule": "source_elapsed_step", "value": 8},
        "fixed_repeats": 3,
        "roots_per_task": 3,
        "operators": list(K3_OPERATORS),
        "expected_roots": len(roots),
        "expected_jobs": len(jobs),
        "expected_simulator_executions": len(roots) * 5 * 3,
        "roots": roots,
        "tasks": sorted(k3["tasks"]),
        "suites": sorted(k3["suites"]),
        "task_folds": dict(k3["task_folds"]),
        "jobs": jobs,
        "forbidden_adaptations": sorted(set(k3.get("forbidden_adaptations", [])) | {
            "independent_external_confirmation_claim_from_pi05_challenge",
            "pooled_universal_selector_without_separate_protocol",
        }),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "roots": len(roots), "tasks": len(manifest["tasks"]),
        "jobs": len(jobs), "simulator_executions": manifest["expected_simulator_executions"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

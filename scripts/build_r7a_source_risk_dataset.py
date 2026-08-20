#!/usr/bin/env python3
"""Build the R7-A t0 source-risk dataset; no fallback outcome is admitted."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_r6c_dynamic_dataset import hashed_instruction  # noqa: E402
from rase.risk.canonical_action import summary_from_chunk  # noqa: E402
from rase.risk.vla_action_adapters import create_vla_adapter  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_proposal_summary(
    action_trace: np.ndarray, *, policy_id: str, chunk_steps: int = 10
) -> np.ndarray:
    """Reconstruct the deployable t=0 action proposal from the causal trace.

    The LeRobot source policies are configured with ``n_action_steps=10``.  The
    first ten executed actions therefore come from the action queue produced by
    the first policy forward pass.  The collector already freezes that full
    trace; using it here avoids reducing an action-chunk VLA to only its first
    7-DoF command.  No outcome, future observation, or OFT signal is used.
    """
    trace = np.asarray(action_trace, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 7 or trace.shape[0] < int(chunk_steps):
        raise ValueError(
            f"expected at least {chunk_steps} causal source actions [T,7], got {trace.shape}"
        )
    canonical = create_vla_adapter(policy_id).to_canonical(trace[:int(chunk_steps)])
    return summary_from_chunk(canonical).cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--exact-repeat-audit", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path)
    parser.add_argument("--policy-id", default="pi0fast_libero")
    args = parser.parse_args()

    manifest = json.loads(args.initial_keys.read_text())
    audit = json.loads(args.label_audit.read_text())
    if audit.get("status") != "PASS":
        raise ValueError("R7 label-support gate did not pass")
    if audit.get("initial_keys_sha256") != sha256(args.initial_keys):
        raise ValueError("label audit / initial-key hash mismatch")
    if str(audit.get("policy_id") or "pi0fast_libero") != args.policy_id:
        raise ValueError("label audit / requested policy mismatch")
    excluded: set[str] = set()
    exclusion_sha = None
    if args.exclusion_manifest is not None:
        exclusion = json.loads(args.exclusion_manifest.read_text())
        if exclusion.get("status") != "frozen":
            raise ValueError("R7 exclusion manifest is not frozen")
        excluded = {str(key) for key in exclusion.get("excluded_state_keys", [])}
        exclusion_sha = sha256(args.exclusion_manifest)
        if audit.get("exclusion_manifest_sha256") != exclusion_sha:
            raise ValueError("label audit / exclusion-manifest hash mismatch")
    repeat = json.loads(args.exact_repeat_audit.read_text())
    if repeat.get("status") != "PASS" or repeat.get("audited_records") != 16:
        raise ValueError("R7 exact-repeat gate did not pass 16 frozen records")
    if repeat.get("label_audit_sha256") != sha256(args.label_audit):
        raise ValueError("exact-repeat audit / label-audit hash mismatch")
    if repeat.get("exclusion_manifest_sha256") != exclusion_sha:
        raise ValueError("exact-repeat audit / exclusion-manifest hash mismatch")
    by_key = {str(row["state_key"]): row for row in manifest["records"]}
    data_rows = []
    arrays = {
        "image": [], "proprio": [], "action_summary": [],
        "action_summary_single_step": [],
    }
    for path in sorted(args.input_root.glob("suite_*/seed_0/*__seed0.json")):
        meta = json.loads(path.read_text())
        boundaries = meta.get("rows") or []
        if len(boundaries) != 1 or int(boundaries[0]["elapsed_source_steps"]) != 0:
            raise ValueError(f"non-t0 source label in {path}")
        boundary = boundaries[0]
        if str(boundary.get("policy_id")) != args.policy_id:
            raise ValueError(f"unexpected source policy in {path}")
        key = str(boundary["state_key"])
        if key in excluded:
            continue
        frozen = by_key.get(key)
        if frozen is None:
            raise ValueError(f"unfrozen state {key}")
        if boundary.get("persistent_success_if_enter_now") is not None:
            raise ValueError("fallback outcome leakage into R7-A source-risk dataset")
        npz_path = Path(str(meta["npz"]))
        if sha256(npz_path) != str(meta["npz_sha256"]):
            raise ValueError(f"NPZ checksum mismatch: {npz_path}")
        raw = np.load(npz_path)
        if raw["image"].shape[0] != 1 or raw["proprio"].shape[0] != 1:
            raise ValueError(f"expected one t0 feature row in {npz_path}")
        if raw["oft_action"].shape[0] or raw["oft_action_summary"].shape[0]:
            raise ValueError("OFT feature leakage into source-risk-only dataset")
        data_rows.append({
            **frozen,
            "instruction": str(boundary["instruction"]),
            "source_success": bool(meta["source_success"]),
            "source_steps": int(meta["source_steps"]),
            "rollout_seed": int(boundary["rollout_seed"]),
            "metadata": str(path.resolve()),
        })
        arrays["image"].append(raw["image"][0].astype(np.uint8))
        arrays["proprio"].append(raw["proprio"][0].astype(np.float32))
        arrays["action_summary"].append(initial_proposal_summary(
            raw["source_action_trace"], policy_id=args.policy_id, chunk_steps=10,
        ))
        # Preserve the old first-command representation as a preregistered
        # ablation; it is no longer the canonical R7 action-intent feature.
        arrays["action_summary_single_step"].append(
            raw["source_action_summary"][0].astype(np.float32)
        )

    data_rows.sort(key=lambda row: row["state_key"])
    expected_rows = int(audit.get("states", -1))
    if (len(data_rows) != expected_rows
            or len({row["state_key"] for row in data_rows}) != expected_rows):
        raise ValueError(f"R7 source-risk dataset must contain {expected_rows} unique states")
    # Reorder feature arrays to match the deterministic state-key order.
    source_order = []
    for path in sorted(args.input_root.glob("suite_*/seed_0/*__seed0.json")):
        value = json.loads(path.read_text())
        key = str(value["rows"][0]["state_key"])
        if key not in excluded:
            source_order.append(key)
    positions = {key: index for index, key in enumerate(source_order)}
    order = [positions[row["state_key"]] for row in data_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image=np.stack([arrays["image"][index] for index in order]),
        proprio=np.stack([arrays["proprio"][index] for index in order]),
        action_summary=np.stack([arrays["action_summary"][index] for index in order]),
        action_summary_single_step=np.stack([
            arrays["action_summary_single_step"][index] for index in order
        ]),
        language_hash=np.stack([hashed_instruction(row["instruction"]) for row in data_rows]),
        instruction=np.asarray([row["instruction"] for row in data_rows]),
        source_failure=np.asarray([not row["source_success"] for row in data_rows], dtype=np.float32),
        source_success=np.asarray([row["source_success"] for row in data_rows], dtype=np.float32),
        source_steps=np.asarray([row["source_steps"] for row in data_rows], dtype=np.int32),
        state_key=np.asarray([row["state_key"] for row in data_rows]),
        task_id=np.asarray([row["task_id"] for row in data_rows]),
        suite=np.asarray([row["suite"] for row in data_rows]),
        perturb_dim=np.asarray([row["perturb_dim"] for row in data_rows]),
        init_state_id=np.asarray([row["init_state_id"] for row in data_rows], dtype=np.int32),
        policy_id=np.asarray([args.policy_id] * len(data_rows)),
    )
    report = {
        "schema_version": "rase-r7a-source-risk-dataset/v1",
        "status": "frozen",
        "policy_id": args.policy_id,
        "scientific_scope": "development task-clustered t0 source-risk only",
        "dataset": str(args.output.resolve()), "dataset_sha256": sha256(args.output),
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys),
        "label_audit": str(args.label_audit.resolve()),
        "label_audit_sha256": sha256(args.label_audit),
        "exact_repeat_audit": str(args.exact_repeat_audit.resolve()),
        "exact_repeat_audit_sha256": sha256(args.exact_repeat_audit),
        "exclusion_manifest": (str(args.exclusion_manifest.resolve())
                               if args.exclusion_manifest is not None else None),
        "exclusion_manifest_sha256": exclusion_sha,
        "excluded_state_keys": sorted(excluded),
        "rows": len(data_rows), "tasks": len({row["task_id"] for row in data_rows}),
        "failures": sum(not row["source_success"] for row in data_rows),
        "successes": sum(row["source_success"] for row in data_rows),
        "features": [
            "two RGB views at t0", "proprioception at t0",
            f"{args.policy_id} initial 10-step canonical action-proposal summary at t0",
            "single-step action summary (ablation only)", "instruction hash",
        ],
        "target": "source final failure",
        "forbidden": [
            "pool episode_outcome placeholder", "OFT outcome/action/cost",
            "task ordinal", "future frames", "validation/test state",
        ],
        "split_rule": "all four init states of a task stay in one outer fold",
        "action_proposal_contract": {
            "canonical_feature": "summary of source_action_trace[0:10]",
            "reason": "Pi0Fast is configured with n_action_steps=10",
            "causal_at_t0": True,
            "uses_outcome_or_oft": False,
        },
    }
    report_path = args.output.with_suffix(".npz.report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

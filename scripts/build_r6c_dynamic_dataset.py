#!/usr/bin/env python3
"""Build the R6-C dynamic-boundary multi-VLA risk dataset from B1.2 output.

Each row is one recorded boundary of a source trajectory group
(state_key:policy_id:seed_index).  Deployment features are exactly the
protocol's allowed set; outcome labels (source final success, persistent
success/cost if entered now, short-horizon source success) are supervision only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HORIZON = 600  # LIBERO max episode steps used by the collector's persistent branch


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    """Deployment-safe, model-agnostic lexical instruction features."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    words = normalized.split()
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    features += [normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))]
    value = np.zeros(dim, dtype=np.float32)
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        value[index] += sign
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def recent_action_history(trace: np.ndarray, elapsed: int, window: int = 4) -> np.ndarray:
    """Causal source-action history: last up-to-``window`` source actions before
    the boundary, zero-padded and normalized by the collector's action range."""
    start = max(0, elapsed - window)
    chunk = trace[start:elapsed] if elapsed > 0 else np.zeros((0, 7), dtype=np.float32)
    pad = np.zeros((window - len(chunk), 7), dtype=np.float32)
    values = np.concatenate([pad, chunk]) if len(chunk) else np.zeros((window, 7), np.float32)
    return values.reshape(-1).astype(np.float32)


ARMS = [
    {
        "arm_id": "CONTINUE_SOURCE",
        "index": 0,
        "source_id": None,           # filled per group with the group's source policy
        "fallback_id": None,
        "action_representation": "source canonical action summary (20-D)",
        "compute_budget": "no teacher steps; source continues",
        "safe_terminate": False,
    },
    {
        "arm_id": "ENTER_PERSISTENT_OFT",
        "index": 1,
        "source_id": None,           # filled per group with the group's source policy
        "fallback_id": "openvla_oft",
        "action_representation": "persistent OFT continuation actions",
        "compute_budget": "persistent_teacher_steps_if_enter_now",
        "safe_terminate": False,
    },
]


def arm_schema(rows: list[dict]) -> dict:
    """Candidate-arm schema report for the fixed current arms."""
    arms = []
    for arm in ARMS:
        arms.append({
            "arm_id": arm["arm_id"],
            "index": arm["index"],
            "source_id": arm["source_id"],
            "fallback_id": arm["fallback_id"],
            "action_representation": arm["action_representation"],
            "compute_budget": arm["compute_budget"],
            "safe_terminate": arm["safe_terminate"],
        })
    return {
        "schema_version": "rase-r6c-candidate-arm-schema/v1",
        "arms": arms,
        "future_arms": [
            "ENTER_RECOVERY_STUDENT",
            "SAFE_ABORT",
            "other VLA / planner / recovery expert",
        ],
        "policy": "every boundary row records outcomes for all current arms; "
                  "future arms append new counterfactual labels without changing schema",
    }


def load_parity_reference(atlas_root: Path) -> dict[tuple[str, int, str], dict]:
    """Load frozen R6-A per-state source reference for parity rechecks.

    Key: (policy_id, seed_index, state_key).  Value: rollout_seed / success /
    env_steps / suite / task_id, exactly as consumed by audit_r6b1_source_parity.py.
    Only the canonical ``<policy_id>/seed_<k>/summary.json`` layout is accepted;
    stray diagnostic trees (for example ``smoke/``) are ignored.
    """
    references: dict[tuple[str, int, str], dict] = {}
    for summary in atlas_root.glob("*/*/summary.json"):
        parts = summary.parts
        seed_dir = parts[-2]
        if not seed_dir.startswith("seed_"):
            continue
        try:
            seed_index = int(seed_dir.removeprefix("seed_"))
        except ValueError:
            continue
        policy_id = parts[-3]
        data = json.loads(summary.read_text())
        for rec in data.get("per_state", []):
            references[(policy_id, seed_index, str(rec["state_key"]))] = {
                "rollout_seed": int(rec["rollout_seed"]),
                "success": bool(rec.get("source_success", rec["result"]["success"])),
                "env_steps": int(rec["result"]["env_steps"]),
                "suite": rec["suite"],
                "task_id": rec["task_id"],
            }
    return references


def load_exclusions(path: Path) -> set[tuple[str, int, str]]:
    """Load a frozen exclusion manifest into {(policy_id, seed_index, state_key)}.

    Excluded trajectory groups are dropped from the dataset (they are
    pre-recorded known nondeterministic states) and reported as ``excluded`` in
    the dataset report.
    """
    if path is None:
        return set()
    data = json.loads(path.read_text())
    excluded = set()
    for entry in data["excluded"]:
        policy, seed, state = entry
        excluded.add((str(policy), int(seed), str(state)))
    return excluded


def build_dataset(*, input_root: Path | list[Path], protocol: Path, output: Path,
                  history_window: int = 4, max_groups: int = 0,
                  atlas_root: Path | None = None,
                  exclusions: Path | None = None) -> tuple[list[dict], dict]:
    """Extract candidate-arm boundary rows from a B1.2 collection.

    Returns (rows, report).  Rows are sorted by (group_id, elapsed_source_steps)
    so the two-boundary dwell can be evaluated in order.  When ``atlas_root`` is
    provided, every row whose (policy, seed, state) has a frozen R6-A reference
    is rechecked for parity; rows without a reference (R6-C.1B new states/seeds)
    are recorded as ``no_reference`` and validated by the reproducibility audit
    instead of strict parity.  ``exclusions`` optionally points at a frozen
    manifest of known nondeterministic (policy_id, seed_index, state_key) groups
    to drop.  ``input_root`` may be a list to merge the B1.2 collection with the
    R6-C.1B re-collection; duplicate groups are skipped.
    """
    protocol_data = read_json(protocol.resolve())
    if protocol_data.get("schema_version") != "rase-r6b1-dynamic-boundary-protocol/v1":
        raise ValueError("unexpected protocol schema")
    qualified = {row["policy_id"]: row["dynamic_seed_indices"]
                 for row in protocol_data["qualified_source_policies"]}

    references = load_parity_reference(atlas_root) if atlas_root is not None else {}
    excluded = load_exclusions(exclusions)
    roots = [input_root] if isinstance(input_root, Path) else list(input_root)

    metadata_paths = []
    for root in roots:
        for path in sorted(glob.glob(str(root / "suite_*" / "*" / "**" / "*__seed*.json"), recursive=True)):
            name = Path(path).name
            if name == "report.json":
                continue
            metadata_paths.append(path)
    if not metadata_paths:
        raise ValueError(f"no trajectory metadata under {roots}")

    rows: list[dict] = []
    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    action_summaries: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    seen_groups: set[str] = set()
    excluded_groups: list[dict] = []
    parity_failures: list[dict] = []
    no_reference: dict[str, dict] = {}
    parity_checked_rows = 0
    for path_string in metadata_paths:
        path = Path(path_string)
        data = read_json(path.resolve())
        path_parts = set(path.parts)
        policy_id = data["rows"][0]["policy_id"]
        seed_index = int(data["rows"][0]["seed_index"])
        replicate_index = int(data.get("rollout_index", 0))
        if "train_enrichment" in path_parts:
            cohort_role = "enrichment"
        elif "natural_development_eval" in path_parts:
            # Only the canonical rep0 represents the natural OOF distribution.
            # Exact-repeat rep1/2 may augment training, but never calibration or
            # validation, avoiding duplicated evaluation weight.
            cohort_role = "natural" if replicate_index == 0 else "replicate_training"
        else:
            # Frozen B1.2 is the original natural development cohort.
            cohort_role = "natural"
        if policy_id not in qualified or seed_index not in qualified[policy_id]:
            raise ValueError(f"unexpected qualified pair {policy_id}:{seed_index} in {path.name}")
        group = str(data["rows"][0]["group_id"])
        base_group = re.sub(r":rep\d+$", "", group)
        if group in seen_groups:
            continue
        if max_groups and len(seen_groups) >= max_groups:
            break
        state_key = str(data["rows"][0]["state_key"])
        if (policy_id, seed_index, state_key) in excluded:
            excluded_groups.append({"key": [policy_id, seed_index, state_key], "group_id": group})
            continue
        seen_groups.add(group)
        npz = np.load(data["npz"])
        trace = npz["source_action_trace"].astype(np.float32)
        for position, boundary in enumerate(data["rows"]):
            elapsed = int(boundary["elapsed_source_steps"])
            if references:
                reference = references.get((policy_id, seed_index, str(boundary["state_key"])))
                if reference is None:
                    # R6-C.1B: new states/seeds have no R6-A reference.  They are
                    # validated by the reproducibility audit (audit_r6c1b_repro.py),
                    # NOT by strict parity.  Record and continue.
                    no_reference.setdefault(group, {
                        "key": [policy_id, seed_index, str(boundary["state_key"])],
                        "group_id": group,
                    })
                else:
                    parity_checked_rows += 1
                    observed = {
                        "rollout_seed": int(boundary["rollout_seed"]),
                        "source_final_success": bool(boundary["source_final_success"]),
                        "source_total_steps": int(boundary["source_total_steps"]),
                    }
                    expected = {
                        "rollout_seed": reference["rollout_seed"],
                        "source_final_success": bool(reference["success"]),
                        "source_total_steps": reference["env_steps"],
                    }
                    if observed != expected:
                        parity_failures.append({
                            "key": [policy_id, seed_index, str(boundary["state_key"])],
                            "observed": observed, "expected": expected,
                        })
            rows.append({
                "state_key": str(boundary["state_key"]),
                "task_id": str(boundary["task_id"]),
                "suite": str(boundary["suite"]),
                "policy_id": policy_id,
                "seed_index": seed_index,
                "group_id": group,
                "base_group_id": base_group,
                "replicate_index": replicate_index,
                "cohort_role": cohort_role,
                "elapsed_source_steps": elapsed,
                "instruction": str(boundary["instruction"]),
                "source_final_success": bool(boundary["source_final_success"]),
                "source_success_within_8": bool(boundary["source_success_within_8"]),
                "source_success_within_16": bool(boundary["source_success_within_16"]),
                "source_success_within_32": bool(boundary["source_success_within_32"]),
                "persistent_success": bool(boundary["persistent_success_if_enter_now"]),
                "persistent_teacher_steps": float(boundary["persistent_teacher_steps_if_enter_now"] or 0.0),
                "counterfactual_timing": str(boundary.get("counterfactual_timing", "skipped")),
            })
            images.append(npz["image"][position].astype(np.uint8))
            proprios.append(npz["proprio"][position].astype(np.float32))
            action_summaries.append(npz["source_action_summary"][position].astype(np.float32))
            histories.append(recent_action_history(trace, elapsed, history_window))
    if not rows:
        raise ValueError("no boundary rows produced")
    if parity_failures:
        raise ValueError(f"parity gate failed on {len(parity_failures)} rows: "
                         f"{json.dumps(parity_failures[:5], indent=1, sort_keys=True)}")

    # Group rows stay contiguous per trajectory so the trainer can apply the
    # two-boundary dwell across each group's boundaries in order.
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["group_id"], rows[i]["elapsed_source_steps"]))
    rows = [rows[i] for i in order]
    images = [images[i] for i in order]
    proprios = [proprios[i] for i in order]
    action_summaries = [action_summaries[i] for i in order]
    histories = [histories[i] for i in order]

    policy_order = sorted({row["policy_id"] for row in rows})
    elapsed = np.asarray([row["elapsed_source_steps"] for row in rows], dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=np.stack(images), proprio=np.stack(proprios),
        action_summary=np.stack(action_summaries), history=np.stack(histories),
        elapsed_progress=elapsed / float(HORIZON),
        elapsed_source_steps=elapsed.astype(np.int32),
        instruction=np.asarray([row["instruction"] for row in rows]),
        language_hash=np.stack([hashed_instruction(row["instruction"]) for row in rows]),
        state_key=np.asarray([row["state_key"] for row in rows]),
        task_id=np.asarray([row["task_id"] for row in rows]),
        suite=np.asarray([row["suite"] for row in rows]),
        group_id=np.asarray([row["group_id"] for row in rows]),
        base_group_id=np.asarray([row["base_group_id"] for row in rows]),
        replicate_index=np.asarray([row["replicate_index"] for row in rows], dtype=np.int64),
        cohort_role=np.asarray([row["cohort_role"] for row in rows]),
        policy_id=np.asarray([row["policy_id"] for row in rows]),
        policy_index=np.asarray([policy_order.index(row["policy_id"]) for row in rows], dtype=np.int64),
        source_success=np.asarray([row["source_final_success"] for row in rows], dtype=np.float32),
        source_trials=np.ones(len(rows), dtype=np.float32),
        source_within_8=np.asarray([row["source_success_within_8"] for row in rows], dtype=np.float32),
        source_within_16=np.asarray([row["source_success_within_16"] for row in rows], dtype=np.float32),
        source_within_32=np.asarray([row["source_success_within_32"] for row in rows], dtype=np.float32),
        persistent_success=np.asarray([row["persistent_success"] for row in rows], dtype=np.float32),
        persistent_teacher_steps=np.asarray([row["persistent_teacher_steps"] for row in rows], dtype=np.float32),
        # Candidate-arm outcome labels: arm 0 = CONTINUE_SOURCE, arm 1 = ENTER_PERSISTENT_OFT.
        arm_success=np.stack([
            np.asarray([row["source_final_success"] for row in rows], dtype=np.float32),
            np.asarray([row["persistent_success"] for row in rows], dtype=np.float32),
        ], axis=-1),
        arm_teacher_steps=np.stack([
            np.zeros(len(rows), dtype=np.float32),
            np.asarray([row["persistent_teacher_steps"] for row in rows], dtype=np.float32),
        ], axis=-1),
        arm_ids=np.asarray([arm["index"] for arm in ARMS], dtype=np.int64),
    )
    report = {
        "schema_version": "rase-r6c-candidate-arm-dataset/v1",
        "status": "complete" if not max_groups else "diagnostic_smoke",
        "scientific_scope": protocol_data["scientific_scope"],
        "dataset": str(output.resolve()), "dataset_sha256": sha256(output),
        "input_root": [str(r.resolve()) for r in roots] if len(roots) > 1 else str(roots[0].resolve()),
        "protocol": str(protocol.resolve()), "protocol_sha256": sha256(protocol.resolve()),
        "qualified_source_policies": qualified,
        "history_window": history_window,
        "horizon": HORIZON,
        "n_rows": len(rows),
        "n_groups": len({row["group_id"] for row in rows}),
        "n_base_groups": len({row["base_group_id"] for row in rows}),
        "n_states": len({row["state_key"] for row in rows}),
        "n_tasks": len({row["task_id"] for row in rows}),
        "policies": policy_order,
        "cohort_counts": {
            role: len({row["group_id"] for row in rows if row["cohort_role"] == role})
            for role in sorted({row["cohort_role"] for row in rows})
        },
        "replicate_policy": ("each reproducible replica is an independent trajectory trial; "
                             "all replicas of a base triple remain in the same task fold; "
                             "success-flip triples are excluded by the repro manifest"),
        "arm_schema": arm_schema(rows),
        "parity_reference_root": str(atlas_root.resolve()) if atlas_root is not None else None,
        "parity_checked_rows": parity_checked_rows,
        "parity_failures": len(parity_failures),
        "n_no_reference_groups": len(no_reference),
        "no_reference_groups": list(no_reference.values())[:200],
        "exclusions": str(exclusions.resolve()) if exclusions is not None else None,
        "n_excluded_groups": len(excluded_groups),
        "excluded_groups": excluded_groups,
        "feature_policy": "two RGB views + 8D proprio + 20D source action summary + causal action history + elapsed progress + instruction hash + policy identity + candidate-arm outcome labels",
        "forbidden_features": ["suite", "task ordinal", "future outcomes", "counterfactual OFT result"],
    }
    output.with_suffix(".npz.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="R6-B1.2 collection output root (contains suite_*)")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="frozen r6b1_dynamic_boundary_protocol_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--atlas-root", type=Path, default=None,
                        help="R6-A reference root (policy_pair_atlas_v1); enables parity hard gate")
    parser.add_argument("--exclusions", type=Path, default=None,
                        help="frozen exclusion manifest of known nondeterministic groups")
    args = parser.parse_args()

    _, report = build_dataset(
        input_root=args.input_root, protocol=args.protocol, output=args.output,
        history_window=args.history_window, max_groups=args.max_groups,
        atlas_root=args.atlas_root, exclusions=args.exclusions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

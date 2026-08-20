#!/usr/bin/env python3
"""Adjudicate one formal R7 exact-repeat failure without rewriting that audit.

The third rollout is diagnostic only.  A stable t=0 contract plus stable final
outcome permits a frozen reproducibility exclusion; it never turns the original
full-trace FAIL into PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from freeze_r7a_exact_repeat_manifest import rank


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_divergence(left: np.ndarray, right: np.ndarray, atol: float) -> int | None:
    if left.shape != right.shape:
        return 0
    if not left.size:
        return None
    if np.issubdtype(left.dtype, np.floating):
        changed = np.any(np.abs(left.astype(np.float64) - right.astype(np.float64)) > atol,
                         axis=tuple(range(1, left.ndim)))
    else:
        changed = np.any(left != right, axis=tuple(range(1, left.ndim)))
    indices = np.flatnonzero(changed)
    return int(indices[0]) if len(indices) else None


def load_rollout(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    payload = json.loads(path.read_text())
    npz = Path(str(payload["npz"]))
    if not npz.is_file() or sha256(npz) != payload.get("npz_sha256"):
        raise ValueError(f"missing or changed NPZ for {path}")
    with np.load(npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    return payload, arrays


def pairwise(name_a: str, run_a: tuple[dict, dict[str, np.ndarray]],
             name_b: str, run_b: tuple[dict, dict[str, np.ndarray]],
             atol: float) -> dict:
    meta_a, arr_a = run_a
    meta_b, arr_b = run_b
    trace_a, trace_b = arr_a["source_action_trace"], arr_b["source_action_trace"]
    same_t0 = all(
        left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=atol)
        for left, right in ((arr_a["image"], arr_b["image"]),
                            (arr_a["proprio"], arr_b["proprio"]),
                            (arr_a["source_action"], arr_b["source_action"]),
                            (arr_a["source_action_summary"], arr_b["source_action_summary"]))
    )
    max_diff = (float("inf") if trace_a.shape != trace_b.shape else
                float(np.max(np.abs(trace_a.astype(np.float64)
                                    - trace_b.astype(np.float64)))) if trace_a.size else 0.0)
    return {
        "pair": [name_a, name_b],
        "same_rollout_seed": int(meta_a["rows"][0]["rollout_seed"])
        == int(meta_b["rows"][0]["rollout_seed"]),
        "same_t0_features": same_t0,
        "same_source_success": bool(meta_a["source_success"]) == bool(meta_b["source_success"]),
        "same_source_steps": int(meta_a["source_steps"]) == int(meta_b["source_steps"]),
        "same_stop_reason": meta_a.get("stop_reason") == meta_b.get("stop_reason"),
        "trace_shapes": [list(trace_a.shape), list(trace_b.shape)],
        "first_action_divergence_step": first_divergence(trace_a, trace_b, atol),
        "max_action_abs_difference": max_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-output", type=Path, required=True)
    parser.add_argument("--float-atol", type=float, default=1e-6)
    args = parser.parse_args()

    original = json.loads(args.original_audit.read_text())
    manifest = json.loads(args.manifest.read_text())
    label_audit = json.loads(args.label_audit.read_text())
    mismatches = [row for row in original.get("errors", [])
                  if row.get("reason") == "repeat_mismatch"]
    if original.get("status") != "FAIL" or len(mismatches) != 1:
        raise ValueError("adjudication requires an immutable FAIL audit with one mismatch")
    key = str(mismatches[0]["state_key"])
    selected = next((row for row in manifest.get("records", [])
                     if str(row["state_key"]) == key), None)
    if selected is None or sha256(args.manifest) != original.get("manifest_sha256"):
        raise ValueError("original audit / manifest binding mismatch")
    suite = str(selected["suite"])
    canonical_path = Path(selected["canonical_metadata"])
    rep1_path = (args.repeat_root / f"suite_{suite.lower()}" / "seed_0"
                 / f"{key}__seed0__rep1.json")
    rep2_path = (args.repeat_root / f"suite_{suite.lower()}" / "seed_0"
                 / f"{key}__seed0__rep2.json")
    runs = {
        "canonical": load_rollout(canonical_path),
        "repeat_1": load_rollout(rep1_path),
        "repeat_2": load_rollout(rep2_path),
    }
    pairs = [pairwise(a, runs[a], b, runs[b], args.float_atol)
             for a, b in (("canonical", "repeat_1"),
                          ("canonical", "repeat_2"),
                          ("repeat_1", "repeat_2"))]
    t0_stable = all(row["same_rollout_seed"] and row["same_t0_features"] for row in pairs)
    outcome_stable = all(row["same_source_success"] for row in pairs)
    original_keys = {str(row["state_key"]) for row in manifest.get("records", [])}
    candidates = []
    for path in sorted(args.input_root.glob(f"suite_{suite.lower()}/seed_0/*__seed0.json")):
        payload = json.loads(path.read_text())
        row = payload["rows"][0]
        candidate_key = str(row["state_key"])
        if (candidate_key not in original_keys
                and bool(payload["source_success"]) == bool(selected["source_success"])):
            candidates.append({
                "state_key": candidate_key,
                "suite": suite,
                "task_id": str(row["task_id"]),
                "source_success": bool(payload["source_success"]),
                "selection_rank": rank(candidate_key),
                "canonical_metadata": str(path.resolve()),
                "canonical_metadata_sha256": sha256(path),
            })
    if not candidates:
        raise ValueError("no deterministic same-suite/same-outcome replacement candidate")
    replacement = min(candidates, key=lambda row: row["selection_rank"])
    decision = "EXCLUDE_AND_REPLACE" if t0_stable and outcome_stable else "STOP_PROTOCOL"
    adjudication = {
        "schema_version": "rase-r7a-exact-repeat-adjudication/v1",
        "status": decision,
        "scientific_scope": "diagnostic third repeat; original v1 FAIL remains immutable",
        "state_key": key, "suite": suite,
        "original_audit": str(args.original_audit.resolve()),
        "original_audit_sha256": sha256(args.original_audit),
        "manifest_sha256": sha256(args.manifest),
        "label_audit_sha256": sha256(args.label_audit),
        "third_repeat_metadata": str(rep2_path.resolve()),
        "third_repeat_metadata_sha256": sha256(rep2_path),
        "pairwise": pairs, "t0_stable": t0_stable,
        "outcome_stable": outcome_stable,
        "replacement_selection_rule": "same suite/outcome; lowest frozen v1 hash rank; exclude original 16",
        "proposed_replacement": replacement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    if decision != "EXCLUDE_AND_REPLACE":
        print(json.dumps(adjudication, indent=2, sort_keys=True))
        return 2
    exclusion = {
        "schema_version": "rase-r7a-reproducibility-exclusion/v1",
        "status": "frozen",
        "scientific_scope": "canonical 191-state source-risk cohort only",
        "initial_keys_sha256": label_audit["initial_keys_sha256"],
        "source_label_audit_sha256": sha256(args.label_audit),
        "original_exact_repeat_audit_sha256": sha256(args.original_audit),
        "adjudication": str(args.output.resolve()),
        "adjudication_sha256": sha256(args.output),
        "excluded_state_keys": [key],
        "records": [{
            "state_key": key, "suite": suite, "task_id": selected["task_id"],
            "source_success": bool(selected["source_success"]),
            "reason": "late_closed_loop_action_trace_nondeterminism",
        }],
        "proposed_exact_repeat_replacement": replacement,
        "prohibitions": [
            "do not rewrite original exact_repeat_audit.json",
            "do not relabel or resample the excluded state",
            "do not use excluded state in canonical dataset or OOF",
        ],
    }
    args.exclusion_output.write_text(json.dumps(exclusion, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": decision, "state_key": key,
                      "replacement": replacement["state_key"],
                      "pairwise": pairs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""R6-C.1B-3: reproducibility audit for newly collected (state, policy, seed).

Red line #2: strict parity applies ONLY to (state, policy, seed) triples with a
frozen R6-A reference.  New states/seeds have no historical reference, so this
audit applies the reproducibility protocol:

- two repeated rollouts of the same (state, policy, seed_index) are compared;
- if source success, source terminal steps and boundary success labels agree
  -> the hard labels are reproducible;
- if any of those labels disagree, a third rollout is required;
- a success flip  -> the group is excluded or kept as a probabilistic-label
  group (never a hard label);
- teacher-cost-only differences never trigger a third rollout: all replicas
  are kept, cost supervision uses the median/quantile and spread is recorded;
- same success but different source terminal steps does trigger adjudication,
  because it changes the boundary trajectory even when the final label agrees.

The audit reads the collection tree layout
``<root>/suite_*/<policy>/seed_<k>/*__seed<k>*.json`` plus optional
``__rep<n>`` suffix replicas.  It emits the frozen R6-C.1B exclusion manifest
extension used by the dataset builder.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="R6-C.1B re-collection output root (contains suite_*)")
    parser.add_argument("--atlas-root", type=Path, default=None,
                        help="R6-A reference root; triples present here are strict-parity (skipped by this repro audit)")
    parser.add_argument("--exclusions-output", type=Path, required=True,
                        help="extended exclusion manifest path (repro-flagged groups appended)")
    parser.add_argument("--base-exclusions", type=Path, default=None,
                        help="existing frozen exclusion manifest to extend")
    parser.add_argument("--max-replicas", type=int, default=3,
                        help="how many replicas were collected per new triple")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(args.input_root / "suite_*" / "*" / "**" / "*__seed*.json"),
                             recursive=True))
    paths = [p for p in paths if Path(p).name != "report.json"]
    if not paths:
        raise SystemExit(f"no trajectory metadata under {args.input_root}")

    # Index replicas by (policy, seed_index, state_key).
    by_triple: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for path_string in paths:
        data = read_json(Path(path_string))
        if not data["rows"]:
            continue
        triple = (str(data["rows"][0]["policy_id"]),
                  int(data["rows"][0]["seed_index"]),
                  str(data["rows"][0]["state_key"]))
        by_triple[triple].append({
            "path": path_string,
            "rollout_index": int(data.get("rollout_index", 0)),
            "rollout_seed": int(data["rows"][0]["rollout_seed"]),
            "source_success": bool(data["source_success"]),
            "source_steps": int(data["source_steps"]),
            "stop_reason": str(data.get("stop_reason", "")),
            "n_boundaries": len(data["rows"]),
            "boundary_signature": [
                {
                    "elapsed": int(row["elapsed_source_steps"]),
                    "source_final_success": bool(row["source_final_success"]),
                    "persistent_success": bool(row["persistent_success_if_enter_now"]),
                    "persistent_teacher_steps": float(
                        row["persistent_teacher_steps_if_enter_now"] or 0.0),
                }
                for row in data["rows"]
            ],
        })

    # Atlas-referenced triples additionally face strict source parity in the
    # dataset builder.  They still undergo this replica audit because strict
    # source parity says nothing about OFT boundary-label reproducibility.
    strict_triples: set[tuple[str, int, str]] = set()
    if args.atlas_root is not None:
        for summary in args.atlas_root.glob("*/*/summary.json"):
            parts = summary.parts
            seed_dir = parts[-2]
            if not seed_dir.startswith("seed_"):
                continue
            try:
                seed_index = int(seed_dir.removeprefix("seed_"))
            except ValueError:
                continue
            policy_id = parts[-3]
            for rec in json.loads(summary.read_text()).get("per_state", []):
                strict_triples.add((policy_id, seed_index, str(rec["state_key"])))

    base_exclusions: set[tuple[str, int, str]] = set()
    if args.base_exclusions is not None:
        for entry in read_json(args.base_exclusions)["excluded"]:
            base_exclusions.add((str(entry[0]), int(entry[1]), str(entry[2])))

    pending: list[dict] = []
    reproducible: list[dict] = []
    step_diff: list[dict] = []
    cost_variability: list[dict] = []
    label_variability: list[dict] = []
    needs_third: list[dict] = []
    success_flip: list[dict] = []
    seed_mismatch: list[dict] = []
    for triple in sorted(by_triple):
        replicas = sorted(by_triple[triple], key=lambda r: r["rollout_index"])
        policy, seed_index, state = triple
        outcomes = [r["source_success"] for r in replicas]
        steps = [r["source_steps"] for r in replicas]
        rollout_seeds = [r["rollout_seed"] for r in replicas]
        record = {
            "key": [policy, seed_index, state],
            "replicas": replicas,
            "n_replicas": len(replicas),
            "has_strict_atlas_reference": triple in strict_triples,
        }
        if len(replicas) < 2:
            pending.append({**record, "note": "fewer than 2 replicas; repro audit incomplete"})
            continue
        # Success labels and teacher cost are deliberately adjudicated
        # separately.  Exact simulator/OFT execution can vary by one or two
        # terminal teacher calls while producing the same success labels.  The
        # protocol models that cost spread with replica-level quantiles; it
        # must not multiply the collection with unnecessary rep2 rollouts.
        label_signatures = [json.dumps([
            {
                "elapsed": boundary["elapsed"],
                "source_final_success": boundary["source_final_success"],
                "persistent_success": boundary["persistent_success"],
            }
            for boundary in r["boundary_signature"]
        ], sort_keys=True) for r in replicas]
        cost_signatures = [tuple(
            boundary["persistent_teacher_steps"] for boundary in r["boundary_signature"]
        ) for r in replicas]
        label_disagrees = len(set(label_signatures)) > 1
        cost_disagrees = len(set(cost_signatures)) > 1
        requires_adjudication = (len(set(rollout_seeds)) > 1
                                 or len(set(outcomes)) > 1
                                 or len(set(steps)) > 1
                                 or label_disagrees)
        if requires_adjudication and len(replicas) < 3:
            needs_third.append({**record, "outcomes": outcomes, "steps": steps,
                                "label_disagrees": label_disagrees,
                                "cost_disagrees": cost_disagrees,
                                "note": ("source seed/outcome/terminal step or boundary success "
                                         "labels disagree; collect rollout_index=2 before adjudication")})
            continue
        if len(set(rollout_seeds)) > 1:
            seed_mismatch.append({**record, "rollout_seeds": rollout_seeds,
                                  "note": "replicas do not share an exact rollout_seed -> exclude"})
            continue
        success_flipped = len(set(outcomes)) > 1
        if success_flipped:
            success_flip.append({**record, "outcomes": outcomes, "steps": steps,
                                 "note": "source success remains mixed after third replica -> exclude from hard-label dataset"})
            continue
        if label_disagrees:
            label_variability.append({**record, "outcomes": outcomes, "steps": steps,
                                      "note": ("source outcome stable but OFT boundary labels/cost vary; "
                                               "keep replica-level probabilistic trials")})
            continue
        if len(set(steps)) > 1:
            step_diff.append({**record, "outcomes": outcomes, "steps": steps,
                              "boundary_disagree": label_disagrees,
                              "cost_disagrees": cost_disagrees,
                              "note": "same labels, different terminal steps -> keep replica-level cost trials"})
            continue
        if cost_disagrees:
            cost_variability.append({**record, "outcomes": outcomes, "steps": steps,
                                     "cost_signatures": [list(x) for x in cost_signatures],
                                     "note": ("hard labels agree; teacher cost varies -> keep both "
                                              "replicas and supervise cost by median/quantile")})
            continue
        reproducible.append({**record, "outcomes": outcomes, "steps": steps})

    # Extended exclusion manifest = base + success-flip triples.
    flip_triples = {tuple(e["key"]) for e in success_flip}
    seed_mismatch_triples = {tuple(e["key"]) for e in seed_mismatch}
    all_excluded = sorted(base_exclusions | flip_triples | seed_mismatch_triples)
    manifest = {
        "schema_version": "rase-r6c1b-repro-exclusions/v1",
        "status": "incomplete_needs_third" if pending or needs_third else "frozen",
        "date": "2026-08-10",
        "purpose": ("R6-C.1B reproducibility protocol: success-flip triples are excluded "
                    "from the hard-label dataset (they may be revisited as probabilistic-label groups)"),
        "strict_parity_scope": "existing (state, policy, seed) with R6-A reference only",
        "base_exclusions": str(args.base_exclusions.resolve()) if args.base_exclusions else None,
        "excluded": [list(t) for t in all_excluded],
        "repro_summary": {
            "n_collected_triples": len(by_triple),
            "n_strict_reference_triples": sum(t in strict_triples for t in by_triple),
            "n_pending": len(pending),
            "n_reproducible": len(reproducible),
            "n_step_diff": len(step_diff),
            "n_cost_variability": len(cost_variability),
            "n_label_variability": len(label_variability),
            "n_needs_third": len(needs_third),
            "n_success_flip": len(success_flip),
            "n_seed_mismatch": len(seed_mismatch),
        },
        "needs_third": needs_third,
        "label_variability": label_variability,
        "cost_variability": cost_variability,
    }
    args.exclusions_output.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "repro_summary": manifest["repro_summary"],
        "excluded": manifest["excluded"],
        "step_diff_triples": [e["key"] for e in step_diff],
        "cost_variability_triples": [e["key"] for e in cost_variability],
        "label_variability_triples": [e["key"] for e in label_variability],
        "needs_third_triples": [e["key"] for e in needs_third],
        "seed_mismatch_triples": [e["key"] for e in seed_mismatch],
        "pending_triples": [e["key"] for e in pending],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

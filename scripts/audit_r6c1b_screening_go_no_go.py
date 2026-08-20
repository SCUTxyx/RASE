#!/usr/bin/env python3
"""Audit source-only R6-C.1B screening before any OFT-labelled collection.

This is an information/value gate, not a method-success gate: source-only
screening can establish completeness and the number/diversity of hard source
failures, but it cannot establish whether OFT rescues those states.  A PASS
therefore authorizes only the next label-collection stage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SEED_PLAN = {"pi05_libero": [2, 3], "pi0fast_libero": [1]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pi05-enrichment-failures", type=int, default=30)
    parser.add_argument("--min-pi05-failure-suites", type=int, default=4)
    parser.add_argument("--min-pi05-failure-tasks", type=int, default=12)
    parser.add_argument("--min-pi0fast-failures", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(args.initial_keys.read_text())
    role_states = Counter(str(row["role"]) for row in manifest["records"])
    expected = {
        f"{policy}:{role}": len(seeds) * role_states[role]
        for policy, seeds in SEED_PLAN.items()
        for role in ("natural_development_eval", "train_enrichment")
    }

    paths = sorted(glob.glob(str(args.screen_root / "suite_*" / "*" / "*" /
                                  "seed_*" / "*__seed*.json")))
    records = []
    seen = set()
    for value in paths:
        path = Path(value)
        data = json.loads(path.read_text())
        if not data.get("rows"):
            continue
        row = data["rows"][0]
        role = "train_enrichment" if "train_enrichment" in path.parts else "natural_development_eval"
        key = (str(row["policy_id"]), int(row["seed_index"]), str(row["state_key"]), role)
        if key in seen:
            raise ValueError(f"duplicate screening trajectory: {key}")
        seen.add(key)
        records.append({
            "policy_id": key[0], "seed_index": key[1], "state_key": key[2],
            "role": role, "suite": str(row["suite"]), "task_id": str(row["task_id"]),
            "source_success": bool(data.get("source_success", row["source_final_success"])),
            "source_steps": int(data.get("source_steps", row["source_total_steps"])),
            "hard_for_collection": ((not bool(data.get("source_success", row["source_final_success"])))
                                    or any(bool(value["source_final_success"])
                                           and not bool(value["source_success_within_16"])
                                           for value in data["rows"])),
        })

    observed = Counter(f'{r["policy_id"]}:{r["role"]}' for r in records)
    completeness = {key: {"expected": count, "observed": observed.get(key, 0)}
                    for key, count in expected.items()}
    complete = all(v["observed"] == v["expected"] for v in completeness.values())

    summaries = {}
    for policy in SEED_PLAN:
        subset = [r for r in records if r["policy_id"] == policy]
        failures = [r for r in subset if not r["source_success"]]
        enrich_failures = [r for r in failures if r["role"] == "train_enrichment"]
        hard_enrichment = [r for r in subset
                           if r["role"] == "train_enrichment" and r["hard_for_collection"]]
        summaries[policy] = {
            "trajectories": len(subset),
            "failures": len(failures),
            "failure_rate": len(failures) / max(1, len(subset)),
            "natural_failures": sum(r["role"] == "natural_development_eval" for r in failures),
            "enrichment_failures": len(enrich_failures),
            "hard_enrichment_states": len({r["state_key"] for r in hard_enrichment}),
            "hard_enrichment_trajectories": len(hard_enrichment),
            "failure_suites": sorted({r["suite"] for r in failures}),
            "failure_tasks": sorted({r["task_id"] for r in failures}),
            "enrichment_failure_suites": sorted({r["suite"] for r in enrich_failures}),
            "enrichment_failure_tasks": sorted({r["task_id"] for r in enrich_failures}),
        }

    pi05 = summaries["pi05_libero"]
    pi0fast = summaries["pi0fast_libero"]
    reasons = []
    if not complete:
        reasons.append("screening collection is incomplete")
    if pi05["enrichment_failures"] < args.min_pi05_enrichment_failures:
        reasons.append("Pi0.5 enrichment failures below the predeclared target")
    if len(pi05["enrichment_failure_suites"]) < args.min_pi05_failure_suites:
        reasons.append("Pi0.5 enrichment failures do not cover four suites")
    if len(pi05["enrichment_failure_tasks"]) < args.min_pi05_failure_tasks:
        reasons.append("Pi0.5 enrichment failures do not cover twelve tasks")
    if pi0fast["failures"] < args.min_pi0fast_failures:
        reasons.append("Pi0Fast screening failures are too sparse")
    passed = not reasons
    collector_hashes = set()
    for report_path in glob.glob(str(args.screen_root / "suite_*" / "*" / "*" /
                                          "seed_*" / "report.json")):
        value = json.loads(Path(report_path).read_text()).get("collector_sha256")
        if value:
            collector_hashes.add(str(value))
    result = {
        "schema_version": "rase-r6c1b-screening-go-no-go/v1",
        "status": "complete" if complete else "incomplete",
        "scientific_scope": ("source-only data-value audit; PASS authorizes OFT label collection "
                             "only and is not evidence that the selector gate passes"),
        "screen_root": str(args.screen_root.resolve()),
        "initial_keys": str(args.initial_keys.resolve()),
        "initial_keys_sha256": sha256(args.initial_keys),
        "reported_collector_sha256s": sorted(collector_hashes),
        "collector_provenance_note": ("screening used rollout_index=0 only; the in-flight "
                                      "collector update changed replica handling but leaves "
                                      "rollout_index=0 seed derivation unchanged"),
        "completeness": completeness,
        "policy_summaries": summaries,
        "label_collection_gate_passed": passed,
        "decision": "GO_LABEL_COLLECTION" if passed else "STOP_OR_REDESIGN_ENRICHMENT",
        "reasons": reasons,
        "important_limitation": "OFT rescueability is unknown until counterfactual labels are collected.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

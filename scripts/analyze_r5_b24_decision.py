#!/usr/bin/env python3
"""Produce the frozen R5-B24 stop/go decision from completed probability data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_all_success_trials_for_wilson_lcb(threshold: float, z: float) -> int:
    """Smallest n whose one-sided Wilson LCB reaches threshold when x=n."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if z <= 0.0:
        raise ValueError("z must be positive")
    return math.ceil(threshold * z * z / (1.0 - threshold))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z", type=float, default=1.6448536269514722)
    args = parser.parse_args()

    summary: dict[str, Any] = json.loads(args.summary.read_text())
    collection: dict[str, Any] = json.loads(args.collection_report.read_text())
    if summary.get("source_collection_report_sha256") != sha256(args.collection_report):
        raise ValueError("collection report changed after summary was written")

    curves = summary["exploratory_probability_opportunity"]["state_curves"]
    source_safe = []
    recovery_created = []
    for state, curve in curves.items():
        by_step = {int(point["elapsed_oft_steps"]): float(point["success_probability"]) for point in curve}
        if by_step.get(0) == 1.0:
            source_safe.append(state)
        if by_step.get(0, 0.0) < 1.0 and any(step > 0 and probability == 1.0 for step, probability in by_step.items()):
            recovery_created.append(state)

    compared = int(collection["historical_handback_labels_compared"])
    matched = int(collection["historical_handback_label_matches"])
    opportunity_ready = summary.get("probability_opportunity_gate_status") == "ready"
    protocol_ready = summary.get("protocol_gate_status") == "ready"
    if not protocol_ready:
        decision = "repair_protocol_before_any_model"
    elif not opportunity_ready:
        decision = "stop_safe_handback_model"
    else:
        decision = "safe_handback_model_may_train"

    output = {
        "schema_version": "rase-r5-b24-decision/v1",
        "summary": str(args.summary.resolve()),
        "summary_sha256": sha256(args.summary),
        "collection_report": str(args.collection_report.resolve()),
        "collection_report_sha256": sha256(args.collection_report),
        "protocol_gate_status": summary["protocol_gate_status"],
        "protocol_gate_reasons": summary["protocol_gate_reasons"],
        "opportunity_gate_status": summary["probability_opportunity_gate_status"],
        "opportunity_gate_reasons": summary["probability_opportunity_gate_reasons"],
        "decision": decision,
        "safe_handback_training_authorized": protocol_ready and opportunity_ready,
        "second_vla_authorized": False,
        "world_model_authorized": False,
        "test_authorized": False,
        "n_states": int(summary["n_states"]),
        "n_tasks": int(summary["n_tasks"]),
        "n_boundaries": int(summary["n_rows"]),
        "n_continuation_trials": int(summary["n_trials"]),
        "live_finite_safe_states": int(collection["live_finite_safe_states"]),
        "source_safe_all_k_states": len(source_safe),
        "recovery_created_all_k_states": len(recovery_created),
        "recovery_created_state_keys": sorted(recovery_created),
        "historical_binary_vs_all_k_agreement": matched / max(1, compared),
        "historical_labels_compared": compared,
        "nondegenerate_boundary_fraction": float(summary["nondegenerate_boundary_fraction"]),
        "nonmonotonic_state_fraction": float(
            summary["exploratory_probability_opportunity"]["nonmonotonic_state_fraction"]
        ),
        "confidence_separated_downward_transitions": len(
            summary["exploratory_probability_opportunity"]["confidence_separated_downward_transitions"]
        ),
        "minimum_zero_failure_trials_for_wilson_lcb": {
            str(threshold): minimum_all_success_trials_for_wilson_lcb(threshold, args.z)
            for threshold in (0.8, 0.9, 0.95)
        },
        "scope": (
            "development stop/go audit; B24 is outcome-enriched and cannot support a "
            "population-level performance claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["safe_handback_training_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

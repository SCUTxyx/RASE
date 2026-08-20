#!/usr/bin/env python3
"""Freeze the independent 48-task K5 confirmation cohort after discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.confirmation import build_confirmation_manifest
from scripts.validate_rase_vnext_protocol import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-catalog", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--discovery-complete", type=Path, required=True)
    parser.add_argument("--discovery-branches", type=Path, required=True)
    parser.add_argument("--shadow-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots-per-task", type=int, default=1)
    parser.add_argument("--salt", default="rase-vnext-confirmation-v1")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    errors = validate(protocol, allow_draft=False)
    if errors:
        raise SystemExit("protocol is not frozen and valid: " + "; ".join(errors))
    discovery = json.loads(args.discovery_manifest.read_text())
    complete = json.loads(args.discovery_complete.read_text())
    shadow = json.loads(args.shadow_audit.read_text())
    if discovery.get("status") != "frozen_discovery":
        raise SystemExit("discovery manifest is not frozen")
    if complete.get("status") != "DISCOVERY_COMPLETE" or complete.get("feasibility_gate") != "PASS":
        raise SystemExit("frozen discovery feasibility did not pass")
    if shadow.get("status") != "GO_CONFIRMATION":
        raise SystemExit("non-abort scientific shadow did not authorize confirmation")
    if shadow.get("branches_sha256") != sha256(args.discovery_branches):
        raise SystemExit("shadow audit is not bound to supplied discovery branches")
    if shadow.get("protocol_sha256") != sha256(args.protocol):
        raise SystemExit("shadow audit is not bound to supplied protocol")

    # Discovery established that Pi0Fast's two native resample candidates were
    # identical in every trial. This is an interface capability mask, not an
    # outcome-derived selection. Keep all scheduled rows but avoid a fake arm.
    candidate = shadow.get("candidate_diagnostics", {}).get("pi0fast.libero", {})
    masks = {}
    if float(candidate.get("distinct_candidate_pair_fraction", 0.0)) == 0.0:
        masks[("pi0fast.libero", "resample.source")] = (
            "discovery_capability_audit:no_native_candidate_diversity"
        )

    catalog = json.loads(args.root_catalog.read_text())
    records = catalog["records"] if isinstance(catalog, dict) else catalog
    discovery_roots = {str(row["root_id"]) for row in discovery["roots"]}
    manifest = build_confirmation_manifest(
        records, protocol, discovery_root_ids=discovery_roots,
        roots_per_task=args.roots_per_task, salt=args.salt,
        operator_masks=masks,
    )
    manifest.update({
        "protocol_sha256": sha256(args.protocol),
        "root_catalog_sha256": sha256(args.root_catalog),
        "root_catalog_pool": str(catalog["pool"]),
        "discovery_manifest_sha256": sha256(args.discovery_manifest),
        "discovery_complete_sha256": sha256(args.discovery_complete),
        "discovery_branches_sha256": sha256(args.discovery_branches),
        "discovery_shadow_sha256": sha256(args.shadow_audit),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "roots": manifest["expected_roots"],
        "tasks": len(manifest["tasks"]), "suites": manifest["suites"],
        "jobs": manifest["expected_jobs"],
        "available_jobs": manifest["expected_available_jobs"],
        "operator_masks": manifest["operator_masks"],
        "sha256": sha256(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


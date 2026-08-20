#!/usr/bin/env python3
"""Freeze the fixed-K discovery schedule after G0 and protocol finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.vnext.discovery import build_discovery_manifest
from scripts.validate_rase_vnext_protocol import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-catalog", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--unlock-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", default="rase-vnext-discovery-v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    errors = validate(protocol, allow_draft=False)
    if errors:
        raise SystemExit("protocol is not frozen and valid: " + "; ".join(errors))
    audit = json.loads(args.unlock_audit.read_text())
    if audit.get("schema_version") != protocol["activation_gate"]["required_schema"]:
        raise SystemExit("unlock audit schema does not match protocol")
    expected_audit_hash = protocol["activation_gate"].get("audit_sha256")
    if expected_audit_hash and sha256(args.unlock_audit) != expected_audit_hash:
        raise SystemExit("unlock audit hash does not match frozen protocol")
    accepted_status = protocol["activation_gate"].get("accepted_status")
    if accepted_status and audit.get("status") != accepted_status:
        raise SystemExit("unlock audit status does not match frozen protocol")
    if audit.get("status") == "FAIL_CONTRACT" or audit.get("errors"):
        raise SystemExit("G0 audit is contract-invalid")
    catalog = json.loads(args.root_catalog.read_text())
    records = catalog["records"] if isinstance(catalog, dict) else catalog
    manifest = build_discovery_manifest(records, protocol, salt=args.salt)
    manifest["protocol_sha256"] = sha256(args.protocol)
    manifest["unlock_audit_sha256"] = sha256(args.unlock_audit)
    manifest["root_catalog_sha256"] = sha256(args.root_catalog)
    manifest["root_catalog_pool"] = str(catalog["pool"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"roots", "jobs"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

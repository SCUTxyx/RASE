#!/usr/bin/env bash
# Freeze the outcome-independent roots and fixed-K discovery schedule after G0.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT=runs/rase_vnext/frozen
POOL=runs/pre_c0_r7/r7a_pi0fast_reset_pool_v1
PROTOCOL=configs/rase_vnext_protocol_v1.json
AUDIT=runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json
CATALOG="$ROOT/root_catalog_v1.json"
MANIFEST="$ROOT/discovery_manifest_v1.json"
mkdir -p "$ROOT"

"$PY" scripts/validate_rase_vnext_protocol.py "$PROTOCOL"
"$PY" scripts/build_rase_vnext_root_catalog.py \
  --pool "$POOL" --output "$CATALOG" --minimum-roots-per-task 2
"$PY" scripts/freeze_rase_vnext_discovery_manifest.py \
  --root-catalog "$CATALOG" --protocol "$PROTOCOL" \
  --unlock-audit "$AUDIT" --output "$MANIFEST"
"$PY" - "$CATALOG" "$MANIFEST" <<'PY'
import json, sys
catalog, manifest = (json.load(open(path)) for path in sys.argv[1:])
assert catalog["n_records"] == 192 and catalog["n_tasks"] == 48
assert manifest["expected_roots"] == 16
assert manifest["expected_jobs"] == 960
assert {row["suite"] for row in manifest["roots"]} == {"Spatial", "Object", "Goal", "Long"}
assert all(key not in json.dumps(manifest["roots"]).lower() for key in ('"outcome"', '"success"', '"reward"'))
print(json.dumps({
    "status": "FROZEN", "catalog_records": catalog["n_records"],
    "catalog_tasks": catalog["n_tasks"], "discovery_roots": manifest["expected_roots"],
    "discovery_jobs": manifest["expected_jobs"],
}, indent=2, sort_keys=True))
PY

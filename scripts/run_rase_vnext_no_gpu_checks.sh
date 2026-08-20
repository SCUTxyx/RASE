#!/usr/bin/env bash
# Pure CPU/import/contract checks. This script never launches collection or training.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=${RASE_CPU_PYTHON:-/root/autodl-tmp/envs/smolvla/bin/python}
"$PY" scripts/validate_rase_vnext_protocol.py \
  configs/rase_vnext_protocol_v1.json --allow-draft
"$PY" -m py_compile \
  rase/vnext/*.py \
  scripts/audit_r10b_chunk_input_divergence.py \
  scripts/status_r10b_chunk_diagnostic.py \
  scripts/validate_rase_vnext_protocol.py \
  scripts/freeze_rase_vnext_discovery_manifest.py \
  scripts/audit_rase_vnext_opportunity.py
"$PY" -m pytest -q \
  tests/test_r10b_chunk_input_divergence.py \
  tests/test_r10b_chunk_status.py \
  tests/test_vnext_canonical_interface.py \
  tests/test_vnext_protocol.py \
  tests/test_vnext_discovery.py \
  tests/test_vnext_opportunity.py
"$PY" scripts/status_r10b_chunk_diagnostic.py

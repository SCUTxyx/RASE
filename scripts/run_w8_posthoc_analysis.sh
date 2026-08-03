#!/usr/bin/env bash
# CPU-only paired analysis and expected failure-only selector readiness audit.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${SMOLVLA_ENV:-smolvla}"

python scripts/summarize_direct_escalation_pairing.py \
  --matrix runs/ngc_w7_heldout24_policy_matrix.json \
  --direct-summary runs/ngc_w8_direct_oft_spatial_heldout24/summary.json \
  --direct-summary runs/ngc_w8_direct_oft_object_heldout24/summary.json \
  --direct-summary runs/ngc_w8_direct_oft_goal_heldout24/summary.json \
  --direct-summary runs/ngc_w8_direct_oft_10_heldout24/summary.json \
  --output-json runs/ngc_w8_direct_escalation_pairing.json \
  --output-md runs/ngc_w8_direct_escalation_pairing.md

python scripts/build_selector_splits.py \
  --dataset runs/ngc_w8_direct_escalation_failure.jsonl \
  --grouping episode \
  --seed 20260729 \
  --output runs/ngc_w8_failure_selector_splits.json

python scripts/audit_selector_pool_support.py \
  --pool-root pool \
  --output runs/ngc_w8_selector_pool_support.json

set +e
python scripts/train_lightweight_selector.py \
  --dataset runs/ngc_w8_direct_escalation_failure.jsonl \
  --splits runs/ngc_w8_failure_selector_splits.json \
  --output-dir runs/ngc_w8_failure_selector_audit \
  --min-train-states 30
audit_status=$?
set -e
if [[ "$audit_status" -ne 2 ]]; then
  echo "ERROR: expected scientific readiness rejection (exit 2), got ${audit_status}" >&2
  exit 1
fi

echo "W8_POSTHOC_ANALYSIS_DONE expected_selector_status=NOT_READY"

#!/usr/bin/env bash
# Frozen R5-A16 label-entropy pilot. This is development data, not validation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AUDIT="${AUDIT:-runs/pre_c0_r5/opportunity_audit_costaware_val_qc.json}"
export TARGET_MANIFEST="${TARGET_MANIFEST:-runs/pre_c0_r5/probability_pilot16_manifest_v1.json}"
export OUTPUT="${OUTPUT:-runs/pre_c0_r5/boundary_probability_pilot16_v2}"
export LOG="${LOG:-runs/pre_c0_r5/boundary_probability_pilot16_v2.log}"
export SPLIT_FILTER=val
export MAX_STATES=0
export ALLOW_CLOSED_OPPORTUNITY_FOR_EVAL=1
export HANDBACK_REPEATS=5
export BOUNDARIES=0,16,64,128

exec bash scripts/run_pre_c0_r5_probabilistic_collect.sh

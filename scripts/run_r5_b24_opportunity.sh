#!/usr/bin/env bash
# Frozen paired-repeat R5-B24 model-free opportunity screen.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AUDIT="${AUDIT:-runs/pre_c0_r5/opportunity_audit_costaware_train_qc.json}"
export TARGET_MANIFEST="${TARGET_MANIFEST:-runs/pre_c0_r5/b24_opportunity_manifest_v1.json}"
export OUTPUT="${OUTPUT:-runs/pre_c0_r5/boundary_probability_b24_v1}"
export LOG="${LOG:-runs/pre_c0_r5/boundary_probability_b24_v1.log}"
export SPLIT_FILTER=train
export MAX_STATES=0
export ALLOW_CLOSED_OPPORTUNITY_FOR_EVAL=0
export HANDBACK_REPEATS=5
export PAIRED_REPEAT_SEEDS=1
export BOUNDARIES=0,16,32,64,96,128

exec bash scripts/run_pre_c0_r5_probabilistic_collect.sh

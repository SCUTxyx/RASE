#!/usr/bin/env bash
# PRE-C1.2 Phase 1: same-H base vs adapted horizon sweep + freeze H.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
OUT="${OUT:-runs/rase_pre_c1_2_horizon_sweep_v1.json}"

"$PY" scripts/eval_pre_c1_2_horizon_sweep.py \
  --protocol-lock "$PROTOCOL" \
  --config "$CONFIG" \
  --adapter-dir "$ADAPTER" \
  --failure-rollout-dir "$FAILURES" \
  --output "$OUT" \
  --freeze-protocol

echo PRE_C1_2_HORIZON_SWEEP_PIPELINE_DONE

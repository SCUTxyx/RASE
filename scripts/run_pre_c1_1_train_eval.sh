#!/usr/bin/env bash
# PRE-C1.1: train recovery LoRA + eval + gate (after successful teacher QC).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_1_protocol_lock.yaml}"
DATASET="${DATASET:-runs/rase_pre_c1_1_distill_dataset_v1.jsonl}"
SPLITS="${SPLITS:-runs/rase_pre_c1_1_distill_dataset_v1.benchmark-splits.json}"
TRAIN_OUT="${TRAIN_OUT:-runs/rase_pre_c1_1_lora_train_v1}"
CACHE="${CACHE:-runs/rase_pre_c1_1_tensor_cache_v1}"
EVAL_OUT="${EVAL_OUT:-runs/rase_pre_c1_1_eval_v1.json}"
DECISION="${DECISION:-runs/rase_pre_c1_1_decision_v1.json}"
AUDIT="${AUDIT:-runs/rase_pre_c1_1_gate_audit_v1.json}"
PROGRESS="${PROGRESS:-progress/2026-08-04_pre_c1_1_gate_results.md}"
ARTIFACT="${ARTIFACT:-artifacts/pre_c1/pre_c1_1_gate_results.json}"

"$PY" scripts/train_smolvla_recovery_lora.py \
  --protocol-lock "$PROTOCOL" \
  --dataset-jsonl "$DATASET" \
  --splits-json "$SPLITS" \
  --config "$CONFIG" \
  --output-dir "$TRAIN_OUT" \
  --cache-dir "$CACHE"

"$PY" scripts/eval_pre_c1_recovery_lora.py \
  --dataset-jsonl "$DATASET" \
  --splits-json "$SPLITS" \
  --config "$CONFIG" \
  --adapter-dir "$TRAIN_OUT/adapter_final" \
  --output "$EVAL_OUT" \
  --failure-rollout-dir runs/rase_pre_c0_same_policy_pilot48_v1

"$PY" scripts/analyze_pre_c1_recovery_gate.py \
  --protocol-lock "$PROTOCOL" \
  --eval-json "$EVAL_OUT" \
  --output "$AUDIT" \
  --decision-output "$DECISION" \
  --progress-md "$PROGRESS" \
  --artifact-json "$ARTIFACT"

# Retitle progress for C1.1
if [[ -f "$PROGRESS" ]]; then
  sed -i '1s/.*/# PRE-C1.1 recovery LoRA gate results/' "$PROGRESS"
fi

echo PRE_C1_1_TRAIN_EVAL_DONE

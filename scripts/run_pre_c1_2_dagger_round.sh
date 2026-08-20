#!/usr/bin/env bash
# PRE-C1.2 Phase 2: one DAgger round (requires frozen H + OFT oracle).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
ROUND="${ROUND:-1}"
OUT="${OUT:-runs/rase_pre_c1_2_dagger_r${ROUND}_v1}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
ORIGINAL="${ORIGINAL:-runs/rase_pre_c1_1_distill_dataset_v1.jsonl}"

"$PY" scripts/collect_pre_c1_2_student_state_oft_relabel.py \
  --protocol-lock "$PROTOCOL" \
  --config "$CONFIG" \
  --adapter-dir "$ADAPTER" \
  --failure-rollout-dir "$FAILURES" \
  --endpoint "$ENDPOINT" \
  --round-id "$ROUND" \
  --output-dir "$OUT" \
  --resume

"$PY" scripts/build_pre_c1_2_dagger_dataset.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$OUT" \
  --original-dataset-jsonl "$ORIGINAL" \
  --output-jsonl "runs/rase_pre_c1_2_distill_dataset_r${ROUND}_v1.jsonl" \
  --splits-output "runs/rase_pre_c1_2_distill_dataset_r${ROUND}_v1.benchmark-splits.json" \
  --qc-json "artifacts/pre_c1/pre_c1_2_dataset_qc_r${ROUND}.json"

echo PRE_C1_2_DAGGER_ROUND_DONE round="$ROUND"

#!/usr/bin/env bash
# PRE-C1.2 Phase 3: E3 (schedule) and/or E4 (prefix weighting) + dual gate.
#
# LEGACY PATH (paused by default after R0 pivot):
#   Full OFT-action BC / E3→E4 is no longer the automatic next step after DAgger R1.
#   Unlock only with ALLOW_LEGACY_E3_E4=1 or artifacts/pre_c1/ALLOW_LEGACY_E3_E4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
DATASET="${DATASET:-runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl}"
SPLITS="${SPLITS:-runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json}"
STAGE="${STAGE:-e4}"  # e3 | e4
TRAIN_OUT="${TRAIN_OUT:-runs/rase_pre_c1_2_lora_${STAGE}_v1}"
CACHE="${CACHE:-runs/rase_pre_c1_2_tensor_cache_${STAGE}_v1}"
EVAL_OUT="${EVAL_OUT:-runs/rase_pre_c1_2_eval_${STAGE}_v1.json}"
DECISION="${DECISION:-runs/rase_pre_c1_2_decision_${STAGE}_v1.json}"
AUDIT="${AUDIT:-runs/rase_pre_c1_2_gate_audit_${STAGE}_v1.json}"
PROGRESS="${PROGRESS:-progress/2026-08-05_pre_c1_2_gate_${STAGE}.md}"
ARTIFACT="${ARTIFACT:-artifacts/pre_c1/pre_c1_2_gate_${STAGE}.json}"
UNLOCK_FILE="${UNLOCK_FILE:-artifacts/pre_c1/ALLOW_LEGACY_E3_E4}"

if [[ "${ALLOW_LEGACY_E3_E4:-0}" != "1" && ! -f "$UNLOCK_FILE" ]]; then
  mkdir -p runs artifacts/pre_c1 progress
  cat > "runs/rase_pre_c1_2_legacy_${STAGE}_blocked.json" <<EOF
{
  "schema_version": "rase-pre-c1-2-legacy-train-block/v1",
  "stage": "${STAGE}",
  "blocked": true,
  "reason": "R0 pivot: refuse automatic E3/E4 full OFT-action BC before recoverability diagnostics",
  "required_path": "DAgger R1 -> global QC -> R0 (teacher-forced / one-step / R(k)) -> branch decision",
  "unlock": "ALLOW_LEGACY_E3_E4=1 or create ${UNLOCK_FILE}",
  "revised_entrypoints": [
    "scripts/run_pre_c1_2_r0.sh",
    "scripts/run_pre_c1_2_revised_train.sh"
  ]
}
EOF
  echo "LEGACY_E3_E4_BLOCKED stage=${STAGE}" >&2
  echo "Refusing legacy E3/E4. Complete R0 diagnostics first." >&2
  echo "Unlock only if intentionally running paused legacy path:" >&2
  echo "  ALLOW_LEGACY_E3_E4=1 bash scripts/run_pre_c1_2_train_eval.sh" >&2
  echo "Blocked marker: runs/rase_pre_c1_2_legacy_${STAGE}_blocked.json" >&2
  exit 42
fi

echo "WARNING: running paused legacy E3/E4 path (ALLOW_LEGACY_E3_E4 unlocked)" >&2

WEIGHT_FLAG=()
if [[ "$STAGE" == "e4" ]]; then
  WEIGHT_FLAG=(--enable-horizon-weighting)
fi

"$PY" scripts/train_smolvla_recovery_lora_c1_2.py \
  --protocol-lock "$PROTOCOL" \
  --dataset-jsonl "$DATASET" \
  --splits-json "$SPLITS" \
  --config "$CONFIG" \
  --output-dir "$TRAIN_OUT" \
  --cache-dir "$CACHE" \
  "${WEIGHT_FLAG[@]}"

"$PY" scripts/eval_pre_c1_2_recovery_lora.py \
  --protocol-lock "$PROTOCOL" \
  --dataset-jsonl "$DATASET" \
  --splits-json "$SPLITS" \
  --config "$CONFIG" \
  --adapter-dir "$TRAIN_OUT/adapter_final" \
  --output "$EVAL_OUT" \
  --failure-rollout-dir runs/rase_pre_c0_same_policy_pilot48_v1

"$PY" scripts/analyze_pre_c1_2_recovery_gate.py \
  --protocol-lock "$PROTOCOL" \
  --eval-json "$EVAL_OUT" \
  --output "$AUDIT" \
  --decision-output "$DECISION" \
  --progress-md "$PROGRESS" \
  --artifact-json "$ARTIFACT"

echo PRE_C1_2_TRAIN_EVAL_DONE stage="$STAGE"

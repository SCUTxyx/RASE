#!/usr/bin/env bash
# PRE-C1.2 R0: global QC assumptions + teacher-forced + recoverability grid + decision.
# Does not unlock legacy E3/E4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
KEYS="${KEYS:-artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
LOG_DIR="${LOG_DIR:-runs/rase_pre_c1_2_pipeline_logs}"
DATASET="${DATASET:-runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl}"
SPLITS="${SPLITS:-runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json}"
DAGGER_OUT="${DAGGER_OUT:-runs/rase_pre_c1_2_dagger_r1_v1}"
TF_OUT="${TF_OUT:-runs/rase_pre_c1_2_r0_teacher_forced_v1.json}"
REC_OUT="${REC_OUT:-runs/rase_pre_c1_2_r0_recoverability_v1}"
DECISION_OUT="${DECISION_OUT:-runs/rase_pre_c1_2_r0_decision_v1.json}"
SMOKE="${SMOKE:-0}"
SKIP_TF="${SKIP_TF:-0}"
SKIP_REC="${SKIP_REC:-0}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

mkdir -p "$LOG_DIR" runs artifacts/pre_c1 progress

kill_oft() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

start_oft() {
  local short="$1"
  local suite="$2"
  local ckpt="$3"
  kill_oft
  echo "=== Start OFT server suite=${suite} ckpt=${ckpt} ==="
  (
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "$LOG_DIR/oft_server_r0_${short}.log" 2>&1
  ) &
  local ready=0
  for _ in $(seq 1 150); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server not ready for $suite" >&2
    tail -80 "$LOG_DIR/oft_server_r0_${short}.log" >&2 || true
    exit 1
  fi
  echo "OFT ready suite=$suite"
}

echo "=== PRE-C1.2 R0 START $(date -Is) ==="

# Global QC (safe to re-run; uses root run summaries)
"$PY" scripts/analyze_pre_c1_2_dagger_global_qc.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$DAGGER_OUT" \
  --state-keys-json "$KEYS" \
  --dataset-jsonl "$DATASET" \
  --splits-json "$SPLITS" \
  --output artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json \
  --progress-md progress/2026-08-05_pre_c1_2_dagger_r1_global_qc.md

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset missing: $DATASET (build after DAgger R1 finishes)" >&2
  exit 2
fi

if [[ "$SKIP_TF" != "1" ]]; then
  echo "=== R0-A teacher-forced $(date -Is) ==="
  TF_FLAGS=()
  if [[ "$SMOKE" == "1" ]]; then
    TF_FLAGS+=(--smoke)
  fi
  "$PY" scripts/eval_pre_c1_2_teacher_forced_fit.py \
    --protocol-lock "$PROTOCOL" \
    --config "$CONFIG" \
    --dataset-jsonl "$DATASET" \
    --splits-json "$SPLITS" \
    --adapter-dir "$ADAPTER" \
    --output "$TF_OUT" \
    "${TF_FLAGS[@]}"
else
  echo "SKIP teacher-forced"
fi

if [[ "$SKIP_REC" != "1" ]]; then
  echo "=== R0-B/C recoverability grid $(date -Is) ==="
  SUITE_SHORTS=(spatial object goal 10)
  SUITE_LABELS=(Spatial Object Goal Long)
  SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
  CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)
  REC_FLAGS=(--resume)
  if [[ "$SMOKE" == "1" ]]; then
    REC_FLAGS+=(--smoke)
  fi
  for idx in "${!SUITE_SHORTS[@]}"; do
    short="${SUITE_SHORTS[$idx]}"
    label="${SUITE_LABELS[$idx]}"
    suite="${SUITE_NAMES[$idx]}"
    ckpt="${CKPTS[$idx]}"
    n_keys="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$KEYS").read_text())
print(len(d.get("by_suite",{}).get("$label",[])))
PY
)"
    if [[ "$n_keys" == "0" ]]; then
      echo "SKIP recoverability suite=$label"
      continue
    fi
    start_oft "$short" "$suite" "$ckpt"
    "$PY" scripts/eval_pre_c1_2_student_prefix_teacher_handover.py \
      --protocol-lock "$PROTOCOL" \
      --config "$CONFIG" \
      --adapter-dir "$ADAPTER" \
      --failure-rollout-dir "$FAILURES" \
      --state-keys-json "$KEYS" \
      --suite "$label" \
      --endpoint "$ENDPOINT" \
      --output-dir "$REC_OUT" \
      "${REC_FLAGS[@]}"
  done
  kill_oft
else
  echo "SKIP recoverability"
fi

echo "=== R0 decision $(date -Is) ==="
DEC_FLAGS=()
if [[ "$ALLOW_INCOMPLETE" == "1" || "$SMOKE" == "1" ]]; then
  DEC_FLAGS+=(--allow-incomplete)
fi
"$PY" scripts/analyze_pre_c1_2_r0.py \
  --protocol-lock "$PROTOCOL" \
  --teacher-forced-json "$TF_OUT" \
  --recoverability-json "$REC_OUT/summary.json" \
  --dagger-qc-json artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json \
  --state-keys-json "$KEYS" \
  --output "$DECISION_OUT" \
  --progress-md progress/2026-08-05_pre_c1_2_r0_decision.md \
  "${DEC_FLAGS[@]}"

echo "=== PRE-C1.2 R0 DONE $(date -Is) ==="
echo "Teacher-forced: $TF_OUT"
echo "Recoverability: $REC_OUT/summary.json"
echo "Decision: $DECISION_OUT"
echo "Legacy E3/E4 remains blocked unless ALLOW_LEGACY_E3_E4=1"

#!/usr/bin/env bash
# Robust suite-by-suite OFT teacher collection for PRE-C1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ROLLOUT="${ROLLOUT:-runs/rase_pre_c0_same_policy_pilot48_v1}"
TEACHER_DIR="${TEACHER_DIR:-runs/rase_pre_c1_oft_teacher_v1}"
CHUNK_STEPS="${CHUNK_STEPS:-10}"
LIMIT="${LIMIT:-0}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
source /root/miniconda3/etc/profile.d/conda.sh

SUITE_SHORTS=(spatial object goal 10)
SUITE_LABELS=(Spatial Object Goal Long)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

mkdir -p "$TEACHER_DIR" runs artifacts/pre_c1 progress

kill_server() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  label="${SUITE_LABELS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"
  suite_out="${TEACHER_DIR}/suite_${short}"
  mkdir -p "$suite_out"
  echo "=== PRE-C1 OFT server suite=${suite} ==="
  kill_server
  (
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_pre_c1_${short}.log" 2>&1
  ) &
  ready=0
  for _ in $(seq 1 120); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server not ready for $suite" >&2
    tail -50 "runs/oft_server_pre_c1_${short}.log" >&2 || true
    exit 1
  fi
  limit_args=()
  if [[ "$LIMIT" != "0" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  "$PY" scripts/collect_pre_c1_oft_teacher_chunks.py \
    --config "$CONFIG" \
    --rollout-dir "$ROLLOUT" \
    --suite "$label" \
    --endpoint "$ENDPOINT" \
    --output-dir "$suite_out" \
    --chunk-steps "$CHUNK_STEPS" \
    --resume \
    "${limit_args[@]}"
  cp -f "$suite_out"/*.json "$TEACHER_DIR"/ 2>/dev/null || true
done

kill_server

"$PY" scripts/build_pre_c1_recovery_distill_dataset.py \
  --protocol-lock artifacts/pre_c1/pre_c1_protocol_lock.yaml \
  --rollout-dir "$ROLLOUT" \
  --oft-teacher-dir "$TEACHER_DIR" \
  --output-jsonl runs/rase_pre_c1_distill_dataset_v1.jsonl \
  --splits-output runs/rase_pre_c1_distill_dataset_v1.benchmark-splits.json \
  --qc-json artifacts/pre_c1/dataset_qc.json \
  --qc-md progress/2026-08-04_pre_c1_dataset_qc.md

echo PRE_C1_DATASET_DONE

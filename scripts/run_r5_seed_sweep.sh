#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET="${DATASET:-runs/pre_c0_r4/boundary_train_v4/boundary_transitions.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r5}"
SEEDS="${SEEDS:-20260810 20260811 20260812 20260813}"
EPOCHS="${EPOCHS:-120}"
STATUS_FILE="$OUTPUT_ROOT/seed_sweep.status"

mkdir -p "$OUTPUT_ROOT"
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/light_student_baseline_seed_${seed}"
  mkdir -p "$output"
  "$PYTHON_BIN" scripts/train_r4d_light_risk_student_v2.py \
    --dataset "$DATASET" \
    --output-dir "$output" \
    --feature-mode baseline \
    --folds 5 \
    --ensemble-size 3 \
    --epochs "$EPOCHS" \
    --device cuda \
    --seed "$seed" \
    > "${output}.log" 2>&1
  echo "DONE $seed" >> "$STATUS_FILE"
done

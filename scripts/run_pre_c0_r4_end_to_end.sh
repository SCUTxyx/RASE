#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AUDIT="${AUDIT:-runs/pre_c0_r4/opportunity_audit_costaware_qc.json}"
BOUNDARY_OUTPUT="${BOUNDARY_OUTPUT:-runs/pre_c0_r4/boundary_train_v3}"
BOUNDARY_LOG="${BOUNDARY_LOG:-runs/pre_c0_r4/boundary_train_v3.log}"
MODEL_OUTPUT="${MODEL_OUTPUT:-runs/pre_c0_r4/world_model_oof_v1}"
MODEL_LOG="${MODEL_LOG:-runs/pre_c0_r4/world_model_oof_v1.log}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"

AUDIT="$AUDIT" OUTPUT="$BOUNDARY_OUTPUT" LOG="$BOUNDARY_LOG" \
  bash scripts/run_pre_c0_r4_collect.sh

"$CONDA_ROOT/bin/conda" run -n smolvla \
  python -u scripts/analyze_r4_live_boundary_opportunity.py \
  --dataset "$BOUNDARY_OUTPUT/boundary_transitions.jsonl" \
  --collection-report "$BOUNDARY_OUTPUT/report.json" \
  --output "$BOUNDARY_OUTPUT/live_opportunity.json"

mkdir -p "$MODEL_OUTPUT" "$(dirname "$MODEL_LOG")"
set +e
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA:-0}" "$CONDA_ROOT/bin/conda" run -n smolvla \
  python -u scripts/train_r4_safe_handback_world_model.py \
  --dataset "$BOUNDARY_OUTPUT/boundary_transitions.jsonl" \
  --collection-report "$BOUNDARY_OUTPUT/report.json" \
  --output-dir "$MODEL_OUTPUT" \
  --folds "${FOLDS:-6}" \
  --ensemble-size "${ENSEMBLE_SIZE:-3}" \
  --epochs "${EPOCHS:-250}" \
  --patience "${PATIENCE:-30}" \
  --hidden-dim "${HIDDEN_DIM:-256}" \
  --device "${TRAIN_DEVICE:-cuda}" 2>&1 | tee -a "$MODEL_LOG"
train_code=${PIPESTATUS[0]}
set -e

# Exit 2 is a valid scientific gate failure, not an infrastructure crash.
if [[ "$train_code" != "0" && "$train_code" != "2" ]]; then
  exit "$train_code"
fi
exit "$train_code"

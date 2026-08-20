#!/usr/bin/env bash
# R6-C candidate-arm no-world-model baseline: five frozen task-held-out OOF
# training seeds, per-VLA stage gate.  Uses the candidate-arm dataset schema
# (arm_success / arm_teacher_steps) and the CandidateArmStudent trainer.
# Gated: run only after the R6-B1.2 collection passes the source-parity audit.
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-runs/pre_c0_r6/r6b1_b1p2_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r6/r6c_candidate_arm_oof_v1}"
DATASET="${DATASET:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz}"
DATASET_REPORT="${DATASET_REPORT:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz.report.json}"
PROTOCOL="${PROTOCOL:-configs/r6b1_dynamic_boundary_protocol_v1.json}"
ATLAS="${ATLAS:-runs/pre_c0_r6/policy_pair_atlas_v1}"
EXCLUSIONS="${EXCLUSIONS:-runs/pre_c0_r6/r6b1_b12_exclusions_v1.json}"
SEEDS="${SEEDS:-10 11 12 13 14}"
FOLD_SEED="${FOLD_SEED:-20260810}"
EPOCHS="${EPOCHS:-60}"

mkdir -p "$OUTPUT_ROOT"

# Build the candidate-arm dataset once from the gated B1.2 collection.
# --atlas-root enables the R6-A source-parity hard gate inside the build.
if [[ ! -f "$DATASET" ]]; then
  "$PYTHON_BIN" scripts/build_candidate_arm_dataset.py \
    --input-root "$DATASET_ROOT" \
    --protocol "$PROTOCOL" \
    --output "$DATASET" \
    --atlas-root "$ATLAS" \
    --exclusions "$EXCLUSIONS"
fi
if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DATASET_REPORT")" != "complete" ]]; then
  echo "R6C ERROR: candidate-arm dataset report is not complete" >&2
  exit 2
fi

: > "$OUTPUT_ROOT/status.txt"
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$output"
  # Per-VLA: each qualified source policy gets its own model (the stage gate).
  for policy in pi0fast_libero pi05_libero; do
    "$PYTHON_BIN" scripts/train_r6c_candidate_arm_student.py \
      --dataset "$DATASET" \
      --dataset-report "$DATASET_REPORT" \
      --output "$output/per_vla_${policy}.json" \
      --mode per_vla --target-policy "$policy" \
      --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
      --members 3 --epochs "$EPOCHS" \
      --dwell 2 --lcb-z 1.6448536269514722 \
      --device cuda \
      > "$output/per_vla_${policy}.log" 2>&1
    echo "DONE $seed per_vla $policy" | tee -a "$OUTPUT_ROOT/status.txt"
  done
  # Zero-shot: train on one source VLA, evaluate on the other.
  "$PYTHON_BIN" scripts/train_r6c_candidate_arm_student.py \
    --dataset "$DATASET" \
    --dataset-report "$DATASET_REPORT" \
    --output "$output/zero_pi0fast_to_pi05.json" \
    --mode zero_shot --source-policy pi0fast_libero --target-policy pi05_libero \
    --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
    --members 3 --epochs "$EPOCHS" \
    --dwell 2 --lcb-z 1.6448536269514722 \
    --device cuda \
    > "$output/zero_pi0fast_to_pi05.log" 2>&1
  echo "DONE $seed zero_pi0fast_to_pi05" | tee -a "$OUTPUT_ROOT/status.txt"
  "$PYTHON_BIN" scripts/train_r6c_candidate_arm_student.py \
    --dataset "$DATASET" \
    --dataset-report "$DATASET_REPORT" \
    --output "$output/zero_pi05_to_pi0fast.json" \
    --mode zero_shot --source-policy pi05_libero --target-policy pi0fast_libero \
    --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
    --members 3 --epochs "$EPOCHS" \
    --dwell 2 --lcb-z 1.6448536269514722 \
    --device cuda \
    > "$output/zero_pi05_to_pi0fast.log" 2>&1
  echo "DONE $seed zero_pi05_to_pi0fast" | tee -a "$OUTPUT_ROOT/status.txt"
  # Leave-one-VLA-out: train on all but the target, evaluate on the target.
  for policy in pi0fast_libero pi05_libero; do
    "$PYTHON_BIN" scripts/train_r6c_candidate_arm_student.py \
      --dataset "$DATASET" \
      --dataset-report "$DATASET_REPORT" \
      --output "$output/loo_${policy}.json" \
      --mode loo --target-policy "$policy" \
      --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
      --members 3 --epochs "$EPOCHS" \
      --dwell 2 --lcb-z 1.6448536269514722 \
      --device cuda \
      > "$output/loo_${policy}.log" 2>&1
    echo "DONE $seed loo $policy" | tee -a "$OUTPUT_ROOT/status.txt"
  done
done

report_args=()
for seed in $SEEDS; do
  report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/per_vla_pi0fast_libero.json")
  report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/per_vla_pi05_libero.json")
done
"$PYTHON_BIN" scripts/audit_r6c_dynamic_stability.py \
  "${report_args[@]}" \
  --mode per_vla \
  --required-passing-seeds 4 \
  --output "$OUTPUT_ROOT/stability.json"
echo complete > "$OUTPUT_ROOT/COMPLETE"

#!/usr/bin/env bash
# R6-C.1C early-window stratified selector: five frozen task-held-out OOF
# training seeds across the R6-C.2 configuration ladder.
#
# Configurations:
#   per_vla       per-VLA model (no policy condition)
#   shared        shared core, no policy condition
#   shared_id     shared core + VLA identity embedding
#   shared_desc   shared core + deployable behavior descriptor
#   shared_calib  shared core + descriptor + small per-VLA FiLM calibration
#   loo           leave-one-VLA-out (descriptor from few-shot calibration split)
#   zero_shot     train on source, eval on target (challenge metric only)
#
# Gate (per VLA, >=4/5 seeds): fold-correct success gap >= -5pp, original
# false-continue <= 5%, absolute paired harm <= 5%, savings >= 20%, no
# concentrated suite harm.  Conditional missed-rescue reported with intervals.
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-runs/pre_c0_r6/r6c1b_merged_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r6/r6c1_early_selector_oof_v1}"
DATASET="${DATASET:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz}"
DATASET_REPORT="${DATASET_REPORT:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz.report.json}"
PROTOCOL="${PROTOCOL:-configs/r6c1b_dynamic_boundary_protocol_v1.json}"
ATLAS="${ATLAS:-runs/pre_c0_r6/policy_pair_atlas_v1}"
EXCLUSIONS="${EXCLUSIONS:-runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json}"
SEEDS="${SEEDS:-10 11 12 13 14}"
FOLD_SEED="${FOLD_SEED:-20260810}"
EPOCHS="${EPOCHS:-60}"
MODE="${MODE:-shared_calib}"
TARGET_POLICIES="${TARGET_POLICIES:-pi0fast_libero pi05_libero}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$DATASET" ]]; then
  echo "R6C1C ERROR: dataset not found: $DATASET (build it with build_candidate_arm_dataset.py first)" >&2
  exit 2
fi
if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DATASET_REPORT")" != "complete" ]]; then
  echo "R6C1C ERROR: dataset report is not complete" >&2
  exit 2
fi

: > "$OUTPUT_ROOT/status.txt"
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$output"
  case "$MODE" in
    per_vla)
      for policy in $TARGET_POLICIES; do
        "$PYTHON_BIN" scripts/train_r6c1_early_selector.py \
          --dataset "$DATASET" \
          --dataset-report "$DATASET_REPORT" \
          --protocol "$PROTOCOL" \
          --output "$output/${MODE}_${policy}.json" \
          --mode "$MODE" --target-policy "$policy" \
          --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
          --members 3 --epochs "$EPOCHS" \
          --device cuda \
          > "$output/${MODE}_${policy}.log" 2>&1
        echo "DONE $seed $MODE $policy" | tee -a "$OUTPUT_ROOT/status.txt"
      done
      ;;
    shared|shared_id|shared_desc|shared_calib)
      # One report per seed; the per-VLA gate reads each policy's own
      # fold-correct metrics from metrics_by_policy / metrics_by_policy_suite.
      "$PYTHON_BIN" scripts/train_r6c1_early_selector.py \
        --dataset "$DATASET" \
        --dataset-report "$DATASET_REPORT" \
        --protocol "$PROTOCOL" \
        --output "$output/${MODE}.json" \
        --mode "$MODE" \
        --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
        --members 3 --epochs "$EPOCHS" \
        --device cuda \
        > "$output/${MODE}.log" 2>&1
      echo "DONE $seed $MODE all_policies" | tee -a "$OUTPUT_ROOT/status.txt"
      ;;
    loo)
      for policy in pi0fast_libero pi05_libero; do
        "$PYTHON_BIN" scripts/train_r6c1_early_selector.py \
          --dataset "$DATASET" \
          --dataset-report "$DATASET_REPORT" \
          --protocol "$PROTOCOL" \
          --output "$output/loo_${policy}.json" \
          --mode loo --target-policy "$policy" \
          --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
          --members 3 --epochs "$EPOCHS" \
          --device cuda \
          > "$output/loo_${policy}.log" 2>&1
        echo "DONE $seed loo $policy" | tee -a "$OUTPUT_ROOT/status.txt"
      done
      ;;
    zero_shot)
      "$PYTHON_BIN" scripts/train_r6c1_early_selector.py \
        --dataset "$DATASET" \
        --dataset-report "$DATASET_REPORT" \
        --protocol "$PROTOCOL" \
        --output "$output/zero_pi0fast_to_pi05.json" \
        --mode zero_shot --source-policy pi0fast_libero --target-policy pi05_libero \
        --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
        --members 3 --epochs "$EPOCHS" \
        --device cuda \
        > "$output/zero_pi0fast_to_pi05.log" 2>&1
      echo "DONE $seed zero_pi0fast_to_pi05" | tee -a "$OUTPUT_ROOT/status.txt"
      "$PYTHON_BIN" scripts/train_r6c1_early_selector.py \
        --dataset "$DATASET" \
        --dataset-report "$DATASET_REPORT" \
        --protocol "$PROTOCOL" \
        --output "$output/zero_pi05_to_pi0fast.json" \
        --mode zero_shot --source-policy pi05_libero --target-policy pi0fast_libero \
        --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
        --members 3 --epochs "$EPOCHS" \
        --device cuda \
        > "$output/zero_pi05_to_pi0fast.log" 2>&1
      echo "DONE $seed zero_pi05_to_pi0fast" | tee -a "$OUTPUT_ROOT/status.txt"
      ;;
    *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
  esac
done

report_args=()
for seed in $SEEDS; do
  case "$MODE" in
    per_vla)
      for policy in $TARGET_POLICIES; do
        report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/per_vla_${policy}.json")
      done
      ;;
    shared|shared_id|shared_desc|shared_calib)
      report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/${MODE}.json")
      ;;
    loo)
      report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/loo_pi0fast_libero.json")
      report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/loo_pi05_libero.json")
      ;;
    zero_shot)
      report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/zero_pi0fast_to_pi05.json")
      report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/zero_pi05_to_pi0fast.json")
      ;;
  esac
done
"$PYTHON_BIN" scripts/audit_r6c1_selector_stability.py \
  "${report_args[@]}" \
  --mode "$MODE" \
  --required-passing-seeds 4 \
  --output "$OUTPUT_ROOT/stability.json"
echo complete > "$OUTPUT_ROOT/COMPLETE"

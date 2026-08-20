#!/usr/bin/env bash
# R6-C.1B/1C gated resume.  It never crosses an expensive/scientific gate
# merely because the preceding process completed.
# The current screening is already running under its own process; this script
# only waits for its COMPLETE marker (idempotent against a pre-existing screen).
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
SCREEN_ROOT=runs/pre_c0_r6/r6c1b_screen_v1
COLLECT_ROOT=runs/pre_c0_r6/r6c1b_collect_v1
# Never overwrite the legacy rep0-only artifact.  The v2 directory makes the
# replica-aware label semantics explicit in every downstream provenance chain.
MERGED_ROOT=runs/pre_c0_r6/r6c1b_replica_aggregated_v2
B12_ROOT=runs/pre_c0_r6/r6b1_b1p2_v1
DATASET="${MERGED_ROOT}/r6c_candidate_arm_dataset.npz"
DATASET_REPORT="${DATASET}.report.json"
PROTOCOL=configs/r6c1b_dynamic_boundary_protocol_v1.json
ATLAS=runs/pre_c0_r6/policy_pair_atlas_v1
REPRO_EXCLUSIONS=runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json
LABEL_SUPPORT=runs/pre_c0_r6/r6c1b_label_support.json
PRETRAIN_READINESS=runs/pre_c0_r6/r6c1b_pretrain_readiness.json
TRAIN_APPROVAL=runs/pre_c0_r6/APPROVE_R6C1_TRAIN
SCREEN_AUDIT=runs/pre_c0_r6/r6c1b_screen_v1/screening_go_no_go.json
OFT_SELECTION=runs/pre_c0_r6/r6c1b_oft_selection_v2.json
OFT_APPROVAL=runs/pre_c0_r6/r6c1b_screen_v1/APPROVE_OFT_LABEL_COLLECTION

echo "R6C1B_RESUME waiting for screening COMPLETE at $(date '+%F %T')"
while [[ ! -f "$SCREEN_ROOT/COMPLETE" ]]; do
  sleep 300
done
echo "R6C1B_RESUME screening complete at $(date '+%F %T')"

"$PYTHON_BIN" scripts/audit_r6c1b_screening_go_no_go.py \
  --screen-root "$SCREEN_ROOT" \
  --initial-keys runs/pre_c0_r6/r6c1b_initial_keys_v1.json \
  --output "$SCREEN_AUDIT"
"$PYTHON_BIN" scripts/freeze_r6c1b_oft_selection.py \
  --screen-root "$SCREEN_ROOT" \
  --screen-audit "$SCREEN_AUDIT" \
  --initial-keys runs/pre_c0_r6/r6c1b_initial_keys_v1.json \
  --output "$OFT_SELECTION"
"$PYTHON_BIN" scripts/audit_r6c1b_collection_plan.py \
  --selection "$OFT_SELECTION" \
  --initial-keys runs/pre_c0_r6/r6c1b_initial_keys_v1.json \
  --output runs/pre_c0_r6/r6c1b_collection_plan_v2.json
if [[ ! -f "$OFT_APPROVAL" ]]; then
  echo "R6C1B_RESUME STOP: screening audit passed, but expensive OFT collection requires $OFT_APPROVAL" >&2
  exit 20
fi
available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
minimum_kb=$((5 * 1024 * 1024))
if [[ "$available_kb" -lt "$minimum_kb" ]]; then
  echo "R6C1B_RESUME STOP: less than 5 GiB free on /root/autodl-tmp" >&2
  exit 22
fi

# Stage 2: targeted collection + reproducibility audit.
if [[ ! -f "$COLLECT_ROOT/COMPLETE" ]]; then
  echo "R6C1B_RESUME launching targeted collection at $(date '+%F %T')"
  bash scripts/run_r6c1b_collect.sh
fi
echo "R6C1B_RESUME collection complete at $(date '+%F %T')"
if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$REPRO_EXCLUSIONS")" != "frozen" ]]; then
  echo "R6C1B_RESUME STOP: reproducibility audit requires targeted third replicas" >&2
  exit 21
fi
"$PYTHON_BIN" scripts/audit_r6c1b_label_support.py \
  --input-root "$B12_ROOT" \
  --input-root "$COLLECT_ROOT" \
  --exclusions "$REPRO_EXCLUSIONS" \
  --output "$LABEL_SUPPORT" || true
"$PYTHON_BIN" scripts/audit_r6c1b_pretrain_readiness.py \
  --input-root "$B12_ROOT" \
  --input-root "$COLLECT_ROOT" \
  --exclusions "$REPRO_EXCLUSIONS" \
  --label-support "$LABEL_SUPPORT" \
  --output "$PRETRAIN_READINESS" || true
if [[ "$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["pi0fast_formal_training_ready"]).lower())' "$PRETRAIN_READINESS")" != "true" ]]; then
  echo "R6C1B_RESUME STOP: Pi0Fast pretrain-readiness gate failed" >&2
  exit 23
fi
if [[ ! -f "$TRAIN_APPROVAL" ]]; then
  echo "R6C1B_RESUME STOP: readiness complete; review $PRETRAIN_READINESS and create $TRAIN_APPROVAL before formal training" >&2
  exit 24
fi

# Stage 3: merged candidate-arm dataset.
mkdir -p "$MERGED_ROOT"
if [[ ! -f "$DATASET" ]]; then
  echo "R6C1B_RESUME building merged dataset at $(date '+%F %T')"
  "$PYTHON_BIN" scripts/build_r6c1_replica_aggregated_dataset.py \
    --input-root "$B12_ROOT" \
    --input-root "$COLLECT_ROOT" \
    --protocol "$PROTOCOL" \
    --output "$DATASET" \
    --atlas-root "$ATLAS" \
    --exclusions "$REPRO_EXCLUSIONS"
  echo "R6C1B_RESUME merged dataset built"
fi

# Stage 4: the decisive experiment is per-VLA first.  A failure is the
# predeclared kill point for the learned-selector method line.
mode=per_vla
MODE="$mode" OUTPUT_ROOT="runs/pre_c0_r6/r6c1_early_selector_oof_v1/$mode" \
  TARGET_POLICIES="pi0fast_libero" \
  DATASET_ROOT="$MERGED_ROOT" DATASET_REPORT="$DATASET_REPORT" PROTOCOL="$PROTOCOL" \
  bash scripts/run_r6c1_early_selector_oof.sh
PER_VLA_STABILITY=runs/pre_c0_r6/r6c1_early_selector_oof_v1/per_vla/stability.json
if [[ "$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["stage_gate_passed"]).lower())' "$PER_VLA_STABILITY")" != "true" ]]; then
  echo "R6C1B_RESUME STOP: per-VLA 1C gate failed; terminate learned-selector escalation" >&2
  echo per_vla_gate_failed > runs/pre_c0_r6/r6c1_early_selector_oof_v1/STOP
  exit 30
fi

# Only a passing per-VLA method may test the shared descriptor-conditioned
# calibration model.  Zero-shot remains a challenge metric, never a main gate.
mode=shared_calib
MODE="$mode" OUTPUT_ROOT="runs/pre_c0_r6/r6c1_early_selector_oof_v1/$mode" \
  DATASET_ROOT="$MERGED_ROOT" DATASET_REPORT="$DATASET_REPORT" PROTOCOL="$PROTOCOL" \
  bash scripts/run_r6c1_early_selector_oof.sh
echo "R6C1B_RESUME decisive OOF stages complete at $(date '+%F %T')"

R6C2_APPROVAL=runs/pre_c0_r6/r6c1_early_selector_oof_v1/APPROVE_R6C2_LADDER
if [[ ! -f "$R6C2_APPROVAL" ]]; then
  echo "R6C1B_RESUME STOP: create $R6C2_APPROVAL after reviewing shared_calib before optional ablations" >&2
  exit 40
fi
for mode in shared shared_id shared_desc loo zero_shot; do
  MODE="$mode" OUTPUT_ROOT="runs/pre_c0_r6/r6c1_early_selector_oof_v1/$mode" \
    DATASET_ROOT="$MERGED_ROOT" DATASET_REPORT="$DATASET_REPORT" PROTOCOL="$PROTOCOL" \
    bash scripts/run_r6c1_early_selector_oof.sh
done

# Stage 5: few-shot calibration curve for the generalization claim (R6-C.2).
echo "R6C1B_RESUME launching few-shot calibration curve at $(date '+%F %T')"
"$PYTHON_BIN" scripts/calibrate_r6c1_fewshot.py \
  --dataset "$DATASET" \
  --dataset-report "$DATASET_REPORT" \
  --protocol "$PROTOCOL" \
  --output runs/pre_c0_r6/r6c1_fewshot_curve.json
echo "R6C1B_RESUME few-shot calibration curve complete at $(date '+%F %T')"

# Stage 6: cross-configuration comparison (R6-C.2).
echo "R6C1B_RESUME launching config comparison at $(date '+%F %T')"
"$PYTHON_BIN" scripts/compare_r6c1_configs.py \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/shared_calib/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/per_vla/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/shared/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/shared_id/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/shared_desc/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/loo/stability.json \
  --stability runs/pre_c0_r6/r6c1_early_selector_oof_v1/zero_shot/stability.json \
  --output runs/pre_c0_r6/r6c1_config_comparison.json
echo "R6C1B_RESUME config comparison complete at $(date '+%F %T')"
echo complete > "$MERGED_ROOT/../r6c1b_resume.COMPLETE"

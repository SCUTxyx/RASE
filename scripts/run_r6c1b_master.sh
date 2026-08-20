#!/usr/bin/env bash
# R6-C.1B -> R6-C.1C master orchestration.
#
# Stages (each writes a COMPLETE marker before the next starts):
#  1. run_r6c1b_screen.sh       source-only screening (hard-case selection)
#  2. run_r6c1b_collect.sh      targeted OFT collection at t={0,8,16} + repro audit
#  3. build_candidate_arm_dataset.py  merge B1.2 + 1B collection into one dataset
#  4. run_r6c1_early_selector_oof.sh  5-seed OOF for the early-window selector
#
# Long-running: designed to run in the background.
#   nohup bash scripts/run_r6c1b_master.sh > runs/pre_c0_r6/r6c1b_master.log 2>&1 &
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
SCREEN_ROOT=runs/pre_c0_r6/r6c1b_screen_v1
COLLECT_ROOT=runs/pre_c0_r6/r6c1b_collect_v1
MERGED_ROOT=runs/pre_c0_r6/r6c1b_merged_v1
B12_ROOT=runs/pre_c0_r6/r6b1_b1p2_v1
DATASET="${MERGED_ROOT}/r6c_candidate_arm_dataset.npz"
PROTOCOL=configs/r6b1_dynamic_boundary_protocol_v1.json
ATLAS=runs/pre_c0_r6/policy_pair_atlas_v1
BASE_EXCLUSIONS=runs/pre_c0_r6/r6b1_b12_exclusions_v1.json
REPRO_EXCLUSIONS=runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json

wait_for() {
  local marker="$1" what="$2"
  echo "R6C1B_MASTER waiting for $what ($marker) at $(date '+%F %T')"
  while [[ ! -f "$marker" ]]; do
    sleep 300
  done
  echo "R6C1B_MASTER $what complete at $(date '+%F %T')"
}

# Stage 1: screening.  It may already be running from a previous launch.
if [[ ! -f "$SCREEN_ROOT/COMPLETE" ]]; then
  echo "R6C1B_MASTER launching screening at $(date '+%F %T')"
  bash scripts/run_r6c1b_screen.sh > "$SCREEN_ROOT/../r6c1b_screen_runner.log" 2>&1
fi
wait_for "$SCREEN_ROOT/COMPLETE" "source-only screening"

# Stage 2: targeted collection + reproducibility audit.
if [[ ! -f "$COLLECT_ROOT/COMPLETE" ]]; then
  echo "R6C1B_MASTER launching targeted collection at $(date '+%F %T')"
  bash scripts/run_r6c1b_collect.sh
fi
wait_for "$COLLECT_ROOT/COMPLETE" "targeted collection + repro audit"

# Stage 3: merged candidate-arm dataset (B1.2 + new collection).
mkdir -p "$MERGED_ROOT"
if [[ ! -f "$DATASET" ]]; then
  echo "R6C1B_MASTER building merged dataset at $(date '+%F %T')"
  "$PYTHON_BIN" scripts/build_candidate_arm_dataset.py \
    --input-root "$B12_ROOT" \
    --input-root "$COLLECT_ROOT" \
    --protocol "$PROTOCOL" \
    --output "$DATASET" \
    --atlas-root "$ATLAS" \
    --exclusions "$REPRO_EXCLUSIONS"
  echo "R6C1B_MASTER merged dataset built"
fi

# Stage 4: five-seed early-selector OOF.
echo "R6C1B_MASTER launching early-selector OOF at $(date '+%F %T')"
MODE=shared_calib bash scripts/run_r6c1_early_selector_oof.sh
echo "R6C1B_MASTER early-selector OOF complete"
echo complete > "$MERGED_ROOT/../r6c1b_master.COMPLETE"
echo "R6C1B_MASTER all stages complete at $(date '+%F %T')"

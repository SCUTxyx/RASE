#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Runs the frozen 16-state Phase0D Smol/OFT timing calibration."
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments; use --help" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_rase_ui_phase0c_factorial16.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
POOL="runs/rase_ui_phase0c_factorial16_pool"
KEYS="runs/rase_ui_phase0d_timing16_keys.json"
SMOL_RUN="runs/rase_ui_phase0d_timing16_smol_${TAG}"
ANALYSIS="runs/rase_ui_phase0d_timing16_analysis_${TAG}.json"
LOG="runs/rase_ui_phase0d_timing16_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ ! -f "$POOL/manifest.json" ]]; then
  echo "ERROR: missing frozen Phase0C pool: $POOL" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  for target in "$KEYS" "$SMOL_RUN" "$ANALYSIS"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh run target already exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PHASE0D TIMING PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PHASE0D FREEZE ONE STEP-2 STATE PER EPISODE ==="
if [[ "$FRESH_RUN" == "0" && -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_KEYS keys=$KEYS"
else
  "$PY" scripts/export_decision_context_keys.py \
    --pool "$POOL" \
    --output "$KEYS" \
    --step 2 \
    --one-per-episode \
    --expected-states 16
fi

echo "=== PHASE0D RUN SMOL ACTION-SELECTION TIMING ==="
smol_mode=(--resume)
if [[ "$FRESH_RUN" == "1" ]]; then
  smol_mode=(--fresh-run)
fi
"$PY" scripts/rollout_smol_interventions.py \
  --config "$CONFIG" \
  --state-keys-json "$KEYS" \
  --output-dir "$SMOL_RUN" \
  --continuation-seeds 1 \
  "${smol_mode[@]}"

echo "=== PHASE0D JOIN SMOL AND EXISTING OFT TIMING ==="
"$PY" scripts/analyze_intervention_timing.py \
  --state-keys-json "$KEYS" \
  --smol-summary "$SMOL_RUN/summary.json" \
  --oft-summary runs/rase_ui_phase0c_factorial16_oft_spatial_factorial_v1/summary.json \
  --oft-summary runs/rase_ui_phase0c_factorial16_oft_object_factorial_v1/summary.json \
  --oft-summary runs/rase_ui_phase0c_factorial16_oft_goal_factorial_v1/summary.json \
  --oft-summary runs/rase_ui_phase0c_factorial16_oft_10_factorial_v1/summary.json \
  --output "$ANALYSIS"

echo "PHASE0D_TIMING_DONE analysis=$ANALYSIS"

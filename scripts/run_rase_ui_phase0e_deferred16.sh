#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Runs immediate versus active-suffix-preserving OFT on the frozen 16 keys."
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
KEYS="${KEYS:-runs/rase_ui_phase0d_timing16_keys.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
OUTPUT_PREFIX="rase_ui_phase0e_deferred16"
ANALYSIS="runs/rase_ui_phase0e_deferred16_analysis_${TAG}.json"
CONTINUE_SUMMARY="${CONTINUE_SUMMARY:-runs/rase_ui_phase0d_timing16_smol_v2/summary.json}"
LOG="runs/rase_ui_phase0e_deferred16_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ ! -s "$KEYS" ]]; then
  echo "ERROR: missing frozen state-key artifact: $KEYS" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  if [[ -e "$ANALYSIS" ]]; then
    echo "ERROR: fresh analysis target exists: $ANALYSIS" >&2
    exit 1
  fi
  for short in spatial object goal 10; do
    target="runs/${OUTPUT_PREFIX}_${short}_${TAG}"
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output target exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PHASE0E RUN IMMEDIATE/DECISION-SUFFIX OFT, SUITE-SERIAL ==="
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=decision-suffix \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

echo "=== PHASE0E AUDIT AND ANALYZE DEFERRED SWITCH ==="
"$PY" scripts/analyze_deferred_switch.py \
  --state-keys-json "$KEYS" \
  --continue-summary "$CONTINUE_SUMMARY" \
  --summary "runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --summary "runs/${OUTPUT_PREFIX}_object_${TAG}/summary.json" \
  --summary "runs/${OUTPUT_PREFIX}_goal_${TAG}/summary.json" \
  --summary "runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --output "$ANALYSIS"

echo "PHASE0E_DEFERRED_DONE analysis=$ANALYSIS"

#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Runs the development-only 48-task replacement-audit pilot."
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments; use --help" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_rase_ui_phase1a_replacement48_source_v2.json}"
DESIGN="${DESIGN:-runs/rase_ui_phase0g_independent48_design.json}"
HANDOFF="${HANDOFF:-runs/rase_ui_phase0g_independent48_analysis_v2.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
POOL="runs/rase_ui_phase1a_replacement48_initial_pool_v2"
SOURCE_SUMMARY="runs/rase_ui_phase1a_replacement48_source_summary_${TAG}.json"
KEYS="runs/rase_ui_phase1a_replacement48_initial_keys_${TAG}.json"
OUTPUT_PREFIX="rase_ui_phase1a_replacement48_oft_only"
ANALYSIS="runs/rase_ui_phase1a_replacement48_analysis_${TAG}.json"
ANALYSIS_MD="runs/rase_ui_phase1a_replacement48_analysis_${TAG}.md"
LOG="runs/rase_ui_phase1a_replacement48_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  targets=("$POOL" "$SOURCE_SUMMARY" "$KEYS" "$ANALYSIS" "$ANALYSIS_MD")
  for short in spatial object goal 10; do
    targets+=("runs/${OUTPUT_PREFIX}_${short}_${TAG}")
  done
  for target in "${targets[@]}"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output target exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PHASE1A PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PHASE1A FULL-HORIZON SOURCE-ONLY + RESET SNAPSHOTS ==="
if [[ "$FRESH_RUN" == "0" && -f "$POOL/manifest.json" \
  && -f "$SOURCE_SUMMARY" ]]; then
  echo "SKIP_COMPLETED_SOURCE pool=$POOL summary=$SOURCE_SUMMARY"
else
  "$PY" scripts/collect_state_pool.py --config "$CONFIG" \
    --summary-output "$SOURCE_SUMMARY"
fi

echo "=== PHASE1A FREEZE AND AUDIT PRE-ACTION RESET STATES ==="
if [[ "$FRESH_RUN" == "0" && -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_INITIAL_KEYS keys=$KEYS"
else
  "$PY" scripts/export_initial_replacement_keys.py \
    --pool "$POOL" --design "$DESIGN" --output "$KEYS" \
    --expected-reset-simulator-timestep 10
fi

echo "=== PHASE1A OFT-ONLY FROM RESET, SUITE-SERIAL ==="
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

echo "=== PHASE1A REPLACEMENT AUDIT ==="
"$PY" scripts/analyze_replacement_audit.py \
  --initial-keys "$KEYS" --source-summary "$SOURCE_SUMMARY" \
  --oft-summary "libero_spatial=runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --oft-summary "libero_object=runs/${OUTPUT_PREFIX}_object_${TAG}/summary.json" \
  --oft-summary "libero_goal=runs/${OUTPUT_PREFIX}_goal_${TAG}/summary.json" \
  --oft-summary "libero_10=runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --handoff-analysis "$HANDOFF" --bootstrap-replicates 10000 \
  --bootstrap-seed 2026080201 --output "$ANALYSIS" --output-md "$ANALYSIS_MD"

echo "PHASE1A_DONE analysis=$ANALYSIS report=$ANALYSIS_MD"

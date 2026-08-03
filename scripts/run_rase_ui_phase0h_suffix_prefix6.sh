#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Runs the exploratory six-state k=0..5 active-suffix mechanism audit."
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments; use --help" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_rase_ui_phase0g_independent48.json}"
SOURCE_ANALYSIS="${SOURCE_ANALYSIS:-runs/rase_ui_phase0g_independent48_analysis_v2.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
KEYS="runs/rase_ui_phase0h_disagreement6_keys_${TAG}.json"
OUTPUT_PREFIX="rase_ui_phase0h_suffix_prefix6"
ANALYSIS="runs/rase_ui_phase0h_suffix_prefix6_analysis_${TAG}.json"
ANALYSIS_MD="runs/rase_ui_phase0h_suffix_prefix6_analysis_${TAG}.md"
LOG="runs/rase_ui_phase0h_suffix_prefix6_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  targets=("$KEYS" "$ANALYSIS" "$ANALYSIS_MD")
  for short in spatial goal 10; do
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

echo "=== PHASE0H PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PHASE0H FREEZE OUTCOME-SELECTED DISAGREEMENT COHORT ==="
if [[ "$FRESH_RUN" == "0" && -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_KEYS keys=$KEYS"
else
  "$PY" scripts/freeze_timing_disagreement_keys.py \
    --analysis "$SOURCE_ANALYSIS" --output "$KEYS" \
    --expected-direct-only 4 --expected-deferred-only 2
fi

echo "=== PHASE0H ACTIVE-SUFFIX PREFIX GRID k=0..5, SUITE-SERIAL ==="
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=suffix-prefix-grid \
OFT_SUITE_SHORTS=spatial,goal,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

echo "=== PHASE0H ENDPOINT IDENTITY AND CURVE ANALYSIS ==="
"$PY" scripts/analyze_suffix_prefix_mechanism.py \
  --cohort "$KEYS" --source-analysis "$SOURCE_ANALYSIS" \
  --summary "libero_spatial=runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --summary "libero_goal=runs/${OUTPUT_PREFIX}_goal_${TAG}/summary.json" \
  --summary "libero_10=runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --expected-suffix-steps 5 --output "$ANALYSIS" --output-md "$ANALYSIS_MD"

echo "PHASE0H_DONE analysis=$ANALYSIS report=$ANALYSIS_MD"

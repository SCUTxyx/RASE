#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Replays all Phase 0E immediate/deferred disagreement states."
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
SOURCE_KEYS="${SOURCE_KEYS:-runs/rase_ui_phase0d_timing16_keys.json}"
REFERENCE="${REFERENCE:-runs/rase_ui_phase0e_deferred16_analysis_v1.json}"
CONTINUE_SUMMARY="${CONTINUE_SUMMARY:-runs/rase_ui_phase0d_timing16_smol_v2/summary.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
KEYS="runs/rase_ui_phase0f_disagreement_keys_${TAG}.json"
OUTPUT_PREFIX="rase_ui_phase0f_disagreement_replay"
ANALYSIS="runs/rase_ui_phase0f_disagreement_analysis_${TAG}.json"
AUDIT="runs/rase_ui_phase0f_disagreement_replay_audit_${TAG}.json"
LOG="runs/rase_ui_phase0f_disagreement_replay_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  for target in "$KEYS" "$ANALYSIS" "$AUDIT" \
    "runs/${OUTPUT_PREFIX}_spatial_${TAG}" "runs/${OUTPUT_PREFIX}_10_${TAG}"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output target exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

"$PY" scripts/select_deferred_disagreement_keys.py \
  --state-keys-json "$SOURCE_KEYS" --analysis "$REFERENCE" --output "$KEYS"

OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=decision-suffix \
OFT_SUITE_SHORTS=spatial,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

"$PY" scripts/analyze_deferred_switch.py \
  --state-keys-json "$KEYS" \
  --continue-summary "$CONTINUE_SUMMARY" \
  --summary "runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --summary "runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --output "$ANALYSIS"

"$PY" scripts/audit_deferred_replay.py \
  --reference "$REFERENCE" --replay "$ANALYSIS" --output "$AUDIT"

echo "PHASE0F_REPLAY_DONE analysis=$ANALYSIS audit=$AUDIT"

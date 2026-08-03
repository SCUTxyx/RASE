#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/pre_a0_strict_resample12.yaml}"
KEYS="${KEYS:-runs/rase_pre_a0_strict_resample12_keys_v1.json}"
FALLBACK="${FALLBACK:-runs/rase_ui_phase1a_replacement48_analysis_v2.json}"
CHUNKS="runs/rase_pre_a1_oft_chunks12_v1"
OUTPUT="runs/rase_pre_a1_oft_prefix_to_smol12_v1"
ANALYSIS="runs/rase_pre_a1_replan_mechanism12_v1.json"
LOG="runs/rase_pre_a1_replan_mechanism12_v1.log"
FRESH_RUN="${FRESH_RUN:-1}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  for target in "$CHUNKS" "$OUTPUT" "$ANALYSIS"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PRE-A1 GENERATE ONE OFT CHUNK PER FROZEN STATE ==="
OFT_RUNNER=generate-prefix \
OUTPUT_PREFIX=rase_pre_a1_oft_chunk12 \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$CHUNKS" \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
./scripts/run_oft_verify_suites.sh "$CONFIG" v1

echo "=== PRE-A1 OFT PREFIX LENGTH -> FROZEN SMOL CONTINUATION ==="
run_mode=()
if [[ "$FRESH_RUN" == "1" ]]; then
  run_mode=(--fresh-run)
fi
"$PY" scripts/rollout_oft_prefix_to_smol.py \
  --config "$CONFIG" --state-keys-json "$KEYS" \
  --candidates-dir "$CHUNKS" --output-dir "$OUTPUT" \
  --prefix-length 0 --prefix-length 1 --prefix-length 4 --prefix-length 8 \
  "${run_mode[@]}"

echo "=== PRE-A1 MECHANISM DECISION ==="
"$PY" scripts/analyze_replan_mechanism.py \
  --prefix-summary "$OUTPUT/summary.json" \
  --fallback-analysis "$FALLBACK" --output "$ANALYSIS"

echo "PRE_A1_REPLAN_DONE analysis=$ANALYSIS"

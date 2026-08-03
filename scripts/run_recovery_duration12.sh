#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/pre_a0_strict_resample12.yaml}"
KEYS="${KEYS:-runs/rase_pre_a0_strict_resample12_keys_v1.json}"
FALLBACK="${FALLBACK:-runs/rase_ui_phase1a_replacement48_analysis_v2.json}"
TRAJECTORIES="runs/rase_pre_a2_oft_recovery_trajectories12_v1"
OUTPUT="runs/rase_pre_a2_recovery_duration12_v1"
ANALYSIS="runs/rase_pre_a2_recovery_duration_audit12_v1.json"
LOG="runs/rase_pre_a2_recovery_duration12_v1.log"
FRESH_RUN="${FRESH_RUN:-1}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  for target in "$TRAJECTORIES" "$OUTPUT" "$ANALYSIS"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PRE-A2 CAPTURE PERSISTENT CLOSED-LOOP OFT TRAJECTORIES ==="
OFT_RUNNER=generate-trajectory \
OUTPUT_PREFIX=rase_pre_a2_oft_trajectory12 \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$TRAJECTORIES" \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
./scripts/run_oft_verify_suites.sh "$CONFIG" v1

echo "=== PRE-A2 REPLAY OFT DURATION -> SAME-SEED SMOL HANDBACK ==="
run_mode=()
if [[ "$FRESH_RUN" == "1" ]]; then
  run_mode=(--fresh-run)
fi
"$PY" scripts/rollout_oft_prefix_to_smol.py \
  --config "$CONFIG" --state-keys-json "$KEYS" \
  --candidates-dir "$TRAJECTORIES" --output-dir "$OUTPUT" \
  --prefix-length 0 --prefix-length 8 --prefix-length 16 \
  --prefix-length 32 --prefix-length 64 "${run_mode[@]}"

echo "=== PRE-A2 DURATION AUDIT ==="
"$PY" scripts/analyze_recovery_duration.py \
  --duration-summary "$OUTPUT/summary.json" \
  --fallback-analysis "$FALLBACK" --output "$ANALYSIS"

echo "PRE_A2_DURATION_DONE analysis=$ANALYSIS"

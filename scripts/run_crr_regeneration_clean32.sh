#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="configs/crr_regeneration_clean32.yaml"
SOURCE_KEYS="runs/rase_continue_fallback_goal_long96_keys.json"
KEYS="runs/crr_regeneration_clean32_keys_v1.json"
CANDIDATES="runs/crr_regeneration_clean32_candidates_v1"
SCREEN="runs/crr_regeneration_clean32_screen_v1"
ANALYSIS="runs/crr_regeneration_clean32_opportunity_v1.json"
REPORT="runs/crr_regeneration_clean32_opportunity_v1.md"
LOG="runs/crr_regeneration_clean32_v1.log"
FRESH_RUN="${FRESH_RUN:-1}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  # The frozen, outcome-independent cohort is reusable across smoke/formal runs.
  for target in "$CANDIDATES" "$SCREEN" "$ANALYSIS" "$REPORT"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output target exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

if [[ ! -f "$KEYS" ]]; then
  "$PY" scripts/freeze_regeneration_keys.py \
    --source "$SOURCE_KEYS" --output "$KEYS" \
    --suite Goal --suite Long --dimension clean \
    --step 0 --step 2 --step 4 --step 6
fi

"$PY" scripts/generate_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS"

run_mode="--resume"
if [[ "$FRESH_RUN" == "1" ]]; then
  run_mode="--fresh-run"
fi
"$PY" scripts/rollout_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS" "$run_mode"

"$PY" scripts/analyze_regeneration_opportunity.py \
  --keys "$KEYS" \
  --resample-summary "$SCREEN/summary.json" \
  --source-summary runs/rase_continue_fallback_goal_long96_smol/summary.json \
  --fallback-summary runs/rase_continue_fallback_goal_long96_oft_goal_v1/summary.json \
  --fallback-summary runs/rase_continue_fallback_goal_long96_oft_10_v1/summary.json \
  --candidate-generation-summary "$CANDIDATES/summary.json" \
  --output "$ANALYSIS" --output-md "$REPORT"

echo "CRR_REGENERATION_CLEAN32_DONE analysis=$ANALYSIS report=$REPORT"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="configs/g1_multiscale_goal8.yaml"
SOURCE_KEYS="runs/rase_continue_fallback_goal_long96_keys.json"
KEYS="runs/g1_multiscale_goal8_keys_v1.json"
CANDIDATES="runs/g1_multiscale_goal8_candidates_v1"
SCREEN="runs/g1_multiscale_goal8_screen_v1"
ANALYSIS="runs/g1_multiscale_goal8_probe_v1.json"
REPORT="runs/g1_multiscale_goal8_probe_v1.md"
LOG="runs/g1_multiscale_goal8_v1.log"
FRESH_RUN="${FRESH_RUN:-1}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$FRESH_RUN" == "1" ]]; then
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
    --suite Goal --dimension clean --step 2 --step 4
fi
"$PY" scripts/generate_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS"

run_mode="--resume"
if [[ "$FRESH_RUN" == "1" ]]; then
  run_mode="--fresh-run"
fi
"$PY" scripts/rollout_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS" "$run_mode"

"$PY" scripts/analyze_multiscale_temperature_probe.py \
  --keys "$KEYS" \
  --rollout-summary "$SCREEN/summary.json" \
  --generation-summary "$CANDIDATES/summary.json" \
  --output "$ANALYSIS" --output-md "$REPORT"

echo "G1_MULTISCALE_GOAL8_DONE analysis=$ANALYSIS report=$REPORT"

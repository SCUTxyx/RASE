#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 $0"
  echo "Runs the development-only PRE-A0 12-state strict-resample opportunity pilot."
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments; use --help" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/pre_a0_strict_resample12.yaml}"
SOURCE_KEYS="${SOURCE_KEYS:-runs/rase_ui_phase1a_replacement48_initial_keys_v2.json}"
KEYS="${KEYS:-runs/rase_pre_a0_strict_resample12_keys_v1.json}"
CANDIDATES="runs/rase_pre_a0_strict_resample12_candidates_v1"
SCREEN="runs/rase_pre_a0_strict_resample12_screen_v1"
FALLBACK="${FALLBACK:-runs/rase_ui_phase1a_replacement48_analysis_v2.json}"
ANALYSIS="runs/rase_pre_a0_candidate_opportunity_v1.json"
ANALYSIS_MD="runs/rase_pre_a0_candidate_opportunity_v1.md"
LOG="runs/rase_pre_a0_strict_resample12_v1.log"
FRESH_RUN="${FRESH_RUN:-1}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  for target in "$CANDIDATES" "$SCREEN" "$ANALYSIS" "$ANALYSIS_MD"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh output target exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PRE-A0 PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PRE-A0 FREEZE OUTCOME-INDEPENDENT 12-STATE DEV KEYS ==="
if [[ -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_KEYS keys=$KEYS"
else
  "$PY" scripts/freeze_pre_a0_keys.py --source "$SOURCE_KEYS" --output "$KEYS"
fi

echo "=== PRE-A0 GENERATE K=4 STRICT SAME-PROFILE RESAMPLES ==="
"$PY" scripts/generate_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS"

echo "=== PRE-A0 ONE-SHOT SAME-STATE ROLLOUT SCREEN ==="
run_mode="--resume"
if [[ "$FRESH_RUN" == "1" ]]; then
  run_mode="--fresh-run"
fi
"$PY" scripts/rollout_pool_candidates.py \
  --config "$CONFIG" --state-keys-json "$KEYS" "$run_mode"

echo "=== PRE-A0 OPPORTUNITY AUDIT ==="
"$PY" scripts/analyze_pre_a0_candidate_opportunity.py \
  --keys "$KEYS" --strict-summary "$SCREEN/summary.json" \
  --fallback-analysis "$FALLBACK" --output "$ANALYSIS" \
  --output-md "$ANALYSIS_MD" --bootstrap-replicates 10000 \
  --bootstrap-seed 2026080302

echo "PRE_A0_DONE analysis=$ANALYSIS report=$ANALYSIS_MD"

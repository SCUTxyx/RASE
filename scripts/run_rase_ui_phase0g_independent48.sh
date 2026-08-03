#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: FRESH_RUN=0|1 TAG=<tag> $0"
  echo "Runs the frozen task/episode-disjoint 48-state timing-opportunity screen."
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
EXCLUDE_KEYS="${EXCLUDE_KEYS:-runs/rase_ui_phase0d_timing16_keys.json}"
TAG="${TAG:-v1}"
FRESH_RUN="${FRESH_RUN:-1}"
POOL="runs/rase_ui_phase0g_independent48_pool"
DESIGN="runs/rase_ui_phase0g_independent48_design.json"
POOL_AUDIT="runs/rase_ui_phase0g_independent48_pool_audit.json"
KEYS="runs/rase_ui_phase0g_independent48_keys.json"
KEYS_AUDIT="runs/rase_ui_phase0g_independent48_keys_audit.json"
SMOL_RUN="runs/rase_ui_phase0g_independent48_smol_${TAG}"
OFT_PREFIX="rase_ui_phase0g_independent48_oft"
ANALYSIS="runs/rase_ui_phase0g_independent48_analysis_${TAG}.json"
OPPORTUNITY="runs/rase_ui_phase0g_independent48_opportunity_${TAG}.json"
LOG="runs/rase_ui_phase0g_independent48_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  targets=(
    "$POOL" "$DESIGN" "$POOL_AUDIT" "$KEYS" "$KEYS_AUDIT"
    "$SMOL_RUN" "$ANALYSIS" "$OPPORTUNITY"
  )
  for short in spatial object goal 10; do
    targets+=("runs/${OFT_PREFIX}_${short}_${TAG}")
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

echo "=== PHASE0G PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PHASE0G FREEZE METADATA-ONLY INDEPENDENT DESIGN ==="
"$PY" scripts/freeze_independent_factorial_design.py \
  --config "$CONFIG" --exclude-keys "$EXCLUDE_KEYS" --output "$DESIGN"

echo "=== PHASE0G COLLECT 48 EPISODES ==="
if [[ "$FRESH_RUN" == "0" && -f "$POOL/manifest.json" ]]; then
  echo "SKIP_EXISTING_POOL manifest=$POOL/manifest.json"
else
  "$PY" scripts/collect_state_pool.py --config "$CONFIG" \
    --summary-output runs/rase_ui_phase0g_independent48_collection_summary.json
fi

echo "=== PHASE0G AUDIT FACTORIAL POOL ==="
"$PY" scripts/audit_factorial_pool.py \
  --config "$CONFIG" --pool "$POOL" --output "$POOL_AUDIT"

echo "=== PHASE0G FREEZE ONE STEP-2 STATE PER EPISODE ==="
if [[ "$FRESH_RUN" == "0" && -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_KEYS keys=$KEYS"
else
  "$PY" scripts/export_decision_context_keys.py \
    --pool "$POOL" --output "$KEYS" --step 2 --one-per-episode \
    --expected-states 48
fi

echo "=== PHASE0G AUDIT INDEPENDENT KEYS ==="
"$PY" scripts/audit_independent_keys.py \
  --state-keys-json "$KEYS" --design "$DESIGN" --config "$CONFIG" \
  --output "$KEYS_AUDIT"

echo "=== PHASE0G STRICT CONTINUE PRIMARY ARM ==="
if [[ "$FRESH_RUN" == "0" && -f "$SMOL_RUN/summary.json" ]]; then
  echo "SKIP_COMPLETED_SMOL summary=$SMOL_RUN/summary.json"
else
  smol_mode=(--resume)
  if [[ "$FRESH_RUN" == "1" ]]; then
    smol_mode=(--fresh-run)
  fi
  "$PY" scripts/rollout_smol_interventions.py \
    --config "$CONFIG" --state-keys-json "$KEYS" --output-dir "$SMOL_RUN" \
    --continuation-seeds 1 --profile continue-only "${smol_mode[@]}"
fi

echo "=== PHASE0G IMMEDIATE / DECISION-SUFFIX OFT, SUITE-SERIAL ==="
OUTPUT_PREFIX="$OFT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=decision-suffix \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

echo "=== PHASE0G EXACT THREE-OPERATOR ANALYSIS ==="
"$PY" scripts/analyze_deferred_switch.py \
  --state-keys-json "$KEYS" --continue-summary "$SMOL_RUN/summary.json" \
  --summary "runs/${OFT_PREFIX}_spatial_${TAG}/summary.json" \
  --summary "runs/${OFT_PREFIX}_object_${TAG}/summary.json" \
  --summary "runs/${OFT_PREFIX}_goal_${TAG}/summary.json" \
  --summary "runs/${OFT_PREFIX}_10_${TAG}/summary.json" \
  --output "$ANALYSIS"

echo "=== PHASE0G PREREGISTERED OPPORTUNITY GATE ==="
set +e
"$PY" scripts/audit_timing_opportunity.py \
  --analysis "$ANALYSIS" --output "$OPPORTUNITY" \
  --min-gap 0.05 --min-tasks-per-timing 2 \
  --bootstrap-replicates 10000 --bootstrap-seed 2026081807
gate_status=$?
set -e
if [[ "$gate_status" != "0" && "$gate_status" != "2" ]]; then
  echo "ERROR: timing opportunity audit failed with code $gate_status" >&2
  exit "$gate_status"
fi

echo "PHASE0G_DONE analysis=$ANALYSIS opportunity=$OPPORTUNITY gate_status=$gate_status"

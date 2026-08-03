#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_rase_ui_phase0c_factorial16.json}"
TAG="${TAG:-factorial_v1}"
FRESH_RUN="${FRESH_RUN:-1}"
POOL="runs/rase_ui_phase0c_factorial16_pool"
KEYS="runs/rase_ui_phase0c_factorial16_keys.json"
POOL_AUDIT="runs/rase_ui_phase0c_factorial16_pool_audit.json"
SMOL_RUN="runs/rase_ui_phase0c_factorial16_smol_${TAG}"
OFT_PREFIX="rase_ui_phase0c_factorial16_oft"
MATRIX="runs/rase_ui_phase0c_factorial16_matrix_${TAG}"
LOG="runs/rase_ui_phase0c_factorial16_${TAG}.log"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" ]]; then
  targets=("$POOL" "$KEYS" "$POOL_AUDIT" "$SMOL_RUN" "$MATRIX")
  for short in spatial object goal 10; do
    targets+=("runs/${OFT_PREFIX}_${short}_${TAG}")
  done
  for target in "${targets[@]}"; do
    if [[ -e "$target" ]]; then
      echo "ERROR: fresh run target already exists: $target" >&2
      exit 1
    fi
  done
fi

mkdir -p runs
exec > >(tee "$LOG") 2>&1

echo "=== PHASE0C PREFLIGHT ==="
"$PY" scripts/preflight_runner.py --min-free-gpu-mib 20000

echo "=== PHASE0C COLLECT BALANCED FACTORIAL16 ==="
if [[ "$FRESH_RUN" == "0" && -f "$POOL/manifest.json" ]]; then
  echo "SKIP_EXISTING_POOL manifest=$POOL/manifest.json"
else
  "$PY" scripts/collect_state_pool.py \
    --config "$CONFIG" \
    --summary-output "runs/rase_ui_phase0c_factorial16_collection_summary.json"
fi

echo "=== PHASE0C AUDIT FACTORIAL DESIGN ==="
"$PY" scripts/audit_factorial_pool.py \
  --config "$CONFIG" \
  --pool "$POOL" \
  --output "$POOL_AUDIT"

echo "=== PHASE0C FREEZE STRICT-CONTINUE KEYS ==="
if [[ "$FRESH_RUN" == "0" && -f "$KEYS" ]]; then
  echo "SKIP_FROZEN_KEYS keys=$KEYS"
else
  "$PY" scripts/export_decision_context_keys.py --pool "$POOL" --output "$KEYS"
fi

echo "=== PHASE0C STRICT CONTINUE / REPLAN ==="
if [[ "$FRESH_RUN" == "0" && -f "$SMOL_RUN/summary.json" ]]; then
  echo "SKIP_COMPLETED_SMOL summary=$SMOL_RUN/summary.json"
else
  smol_mode=(--resume)
  if [[ "$FRESH_RUN" == "1" ]]; then
    smol_mode=(--fresh-run)
  fi
  "$PY" scripts/rollout_smol_interventions.py \
    --config "$CONFIG" \
    --state-keys-json "$KEYS" \
    --output-dir "$SMOL_RUN" \
    --continuation-seeds 1 \
    "${smol_mode[@]}"
fi

echo "=== PHASE0C SWITCH_POLICY(OFT), SUITE-SERIAL ==="
OUTPUT_PREFIX="$OFT_PREFIX" \
STATE_KEYS_JSON="$KEYS" \
CANDIDATES_DIR="$KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
FRESH_RUN="$FRESH_RUN" \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh "$CONFIG" "$TAG"

echo "=== PHASE0C ASSEMBLE THREE-OPERATOR MATRIX ==="
assemble_mode=()
if [[ "$FRESH_RUN" == "1" ]]; then
  assemble_mode=(--fresh-run)
fi
"$PY" scripts/assemble_intervention_matrix.py \
  --smol-run "$SMOL_RUN" \
  --oft-summary "runs/${OFT_PREFIX}_spatial_${TAG}/summary.json" \
  --oft-summary "runs/${OFT_PREFIX}_object_${TAG}/summary.json" \
  --oft-summary "runs/${OFT_PREFIX}_goal_${TAG}/summary.json" \
  --oft-summary "runs/${OFT_PREFIX}_10_${TAG}/summary.json" \
  --output-dir "$MATRIX" \
  "${assemble_mode[@]}"

echo "=== PHASE0C SUCCESS-ONLY OPPORTUNITY GATE ==="
set +e
"$PY" scripts/audit_intervention_opportunity.py \
  --registry "$MATRIX/operators.json" \
  --snapshots "$MATRIX/snapshots.jsonl" \
  --outcomes "$MATRIX/outcomes.jsonl" \
  --output "$MATRIX/opportunity_audit_success_only.json" \
  --min-complete-snapshots 40 \
  --min-oracle-gap 0.05 \
  --min-winning-operators 3 \
  --min-tasks-per-winning-operator 2 \
  --allow-zero-harm \
  --allow-zero-futility
audit_status=$?
set -e
if [[ "$audit_status" != "0" && "$audit_status" != "2" ]]; then
  echo "ERROR: opportunity audit failed with code $audit_status" >&2
  exit "$audit_status"
fi

echo "=== PHASE0C STRATIFIED DIAGNOSTICS ==="
"$PY" scripts/analyze_intervention_matrix.py \
  --matrix-dir "$MATRIX" \
  --output "$MATRIX/analysis.json" \
  --bootstrap-replicates 10000

echo "=== PHASE0C EXPLORATORY COST SENSITIVITY ==="
"$PY" scripts/sweep_intervention_costs.py \
  --matrix-dir "$MATRIX" \
  --output "$MATRIX/cost_sensitivity.json" \
  --success-reward 1 \
  --base-penalty replan_smol=0.01 \
  --sweep-operator switch_oft \
  --sweep-values 0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5

echo "PHASE0C_DONE matrix=$MATRIX audit_status=$audit_status"

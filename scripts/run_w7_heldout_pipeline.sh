#!/usr/bin/env bash
# Resumable W7 held-out candidate, paired-policy, and summary pipeline.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
CFG="${CFG:-configs/ngc_w7_heldout24_screen.yaml}"
STATE_KEYS_JSON="${STATE_KEYS_JSON:-runs/ngc_w7_heldout24_state_keys.json}"
CANDIDATES_DIR="${CANDIDATES_DIR:-runs/ngc_w7_heldout24_candidates_t07}"
SMOL_OUTPUT="${SMOL_OUTPUT:-runs/ngc_w7_heldout24_smol_screen_t07}"
OFT_PREFIX="${OFT_PREFIX:-ngc_w7_heldout24_oft}"
OFT_TAG="${OFT_TAG:-heldout}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$SMOLVLA_ENV"

mkdir -p runs
exec 9>runs/ngc_w7_heldout24_pipeline.lock
if ! flock -n 9; then
  echo "ERROR: another W7 held-out pipeline is already running" >&2
  exit 1
fi

if [[ ! -f "$STATE_KEYS_JSON" ]]; then
  echo "ERROR: frozen held-out state keys missing: $STATE_KEYS_JSON" >&2
  exit 1
fi

if [[ -f "$CANDIDATES_DIR/summary.json" ]]; then
  echo "SKIP_COMPLETED stage=candidates"
else
  python scripts/generate_pool_candidates.py \
    --config "$CFG" \
    --state-keys-json "$STATE_KEYS_JSON" \
    --output-dir "$CANDIDATES_DIR"
fi

if [[ -f "$SMOL_OUTPUT/summary.json" ]]; then
  echo "SKIP_COMPLETED stage=smol-screen"
else
  python -u scripts/rollout_pool_candidates.py \
    --config "$CFG" \
    --mode smolvla-screen \
    --state-keys-json "$STATE_KEYS_JSON" \
    --candidates-dir "$CANDIDATES_DIR" \
    --output-dir "$SMOL_OUTPUT" \
    --resume
fi

OUTPUT_PREFIX="$OFT_PREFIX" \
STATE_KEYS_JSON="$STATE_KEYS_JSON" \
CANDIDATES_DIR="$CANDIDATES_DIR" \
./scripts/run_oft_verify_suites.sh "$CFG" "$OFT_TAG"

conda activate "$SMOLVLA_ENV"
python scripts/summarize_w6_policy_matrix.py \
  --title "W7 held-out L1-L2 paired one-shot policy matrix" \
  --state-keys "$STATE_KEYS_JSON" \
  --smol-summary "$SMOL_OUTPUT/summary.json" \
  --oft-summary Spatial="runs/${OFT_PREFIX}_spatial_${OFT_TAG}/summary.json" \
  --oft-summary Object="runs/${OFT_PREFIX}_object_${OFT_TAG}/summary.json" \
  --oft-summary Goal="runs/${OFT_PREFIX}_goal_${OFT_TAG}/summary.json" \
  --oft-summary Long="runs/${OFT_PREFIX}_10_${OFT_TAG}/summary.json" \
  --output-json runs/ngc_w7_heldout24_policy_matrix.json \
  --output-md runs/ngc_w7_heldout24_policy_matrix.md

echo "W7_HELDOUT_PIPELINE_DONE"

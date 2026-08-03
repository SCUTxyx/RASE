#!/usr/bin/env bash
# Direct OFT-from-snapshot arm on every frozen W7 state (24 rollouts total).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-configs/ngc_w7_heldout24_screen.yaml}"
STATE_KEYS_JSON="${STATE_KEYS_JSON:-runs/ngc_w7_heldout24_state_keys.json}"
CANDIDATES_DIR="${CANDIDATES_DIR:-runs/ngc_w7_heldout24_candidates_t07}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ngc_w8_direct_oft}"
TAG="${TAG:-heldout24}"

OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$STATE_KEYS_JSON" \
CANDIDATES_DIR="$CANDIDATES_DIR" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
./scripts/run_oft_verify_suites.sh "$CFG" "$TAG"

source "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${SMOLVLA_ENV:-smolvla}"
python scripts/export_direct_escalation_dataset.py \
  --smol-summary runs/ngc_w7_heldout24_smol_screen_t07/summary.json \
  --oft-direct-summary "runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --oft-direct-summary "runs/${OUTPUT_PREFIX}_object_${TAG}/summary.json" \
  --oft-direct-summary "runs/${OUTPUT_PREFIX}_goal_${TAG}/summary.json" \
  --oft-direct-summary "runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --pool pool/ngc_w5_l1_l2_camera_robot \
  --output runs/ngc_w8_direct_escalation_failure.jsonl

echo "W8_DIRECT_ESCALATION_DONE"

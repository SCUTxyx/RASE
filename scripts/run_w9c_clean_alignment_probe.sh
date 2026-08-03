#!/usr/bin/env bash
# Small matched SR probe: clean adapter vs preregistered baseline floors.
# Do NOT launch the 140-ep W9C collect until this exits 0.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${SMOLVLA_ENV:-/root/autodl-tmp/envs/smolvla}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
export LIBERO_ROOT="${LIBERO_ROOT:-/root/autodl-tmp/src/LIBERO}"

PROBE_CFG="${PROBE_CFG:-configs/collect_w9c_clean_probe.json}"
SUMMARY="${SUMMARY:-runs/ngc_w9c_clean_probe.json}"
AUDIT="${AUDIT:-runs/ngc_w9c_clean_probe_task_audit.json}"

pytest -q \
  tests/test_lerobot_collection_adapter.py \
  tests/test_w9c_schedule.py \
  tests/test_clean_task_identity.py \
  tests/test_task_fingerprint_stability.py

mkdir -p runs
python scripts/collect_state_pool.py \
  --config "$PROBE_CFG" \
  --summary-output "$SUMMARY"

python scripts/audit_w9c_clean_probe.py \
  --summary "$SUMMARY" \
  --config "$PROBE_CFG" \
  --output "$AUDIT"

echo "W9C_CLEAN_ALIGNMENT_PROBE_DONE summary=$SUMMARY audit=$AUDIT"

#!/usr/bin/env bash
# Gate-controlled continuation: reset pool -> provenance audit -> source labels.
# Intentionally stops before any model, OFT counterfactual, WM or validation run.
set -euo pipefail
cd /root/autodl-tmp/RASE

VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
SUMMARY=runs/pre_c0_r7/r7a_pi0fast_reset_pool_v1_summary.json
POOL=runs/pre_c0_r7/r7a_pi0fast_reset_pool_v1
DESIGN=runs/pre_c0_r7/r7a_multi_episode_design_v1.json
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json

echo "R7A_DRIVER waiting_for=$SUMMARY"
while [[ ! -f "$SUMMARY" ]]; do
  sleep 15
done

"$VLA_PY" scripts/freeze_r7a_reset_keys.py \
  --pool "$POOL" --design "$DESIGN" --output "$KEYS" \
  --expected-episodes 192 --collection-seed 2026081207

"$VLA_PY" - "$KEYS" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "frozen"
assert p["n_records"] == 192 and p["n_tasks"] == 48
assert p["pool_episode_outcome_is_label"] is False
print("R7A_RESET_GATE PASS records=192 tasks=48")
PY

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
"$VLA_PY" scripts/audit_r7a_restore_compat.py \
  --initial-keys "$KEYS" \
  --output runs/pre_c0_r7/r7a_restore_compat.json

scripts/run_r7a_source_labels.sh

# A complete label-support PASS is necessary but not sufficient.  Before the
# dataset is admitted for training, rerun a frozen hash-selected subset (two
# successes and two failures per suite) with the exact same rollout seeds and
# require outcome, terminal step, t0 feature and full action-trace parity.
# This adds source-only rollouts; it never starts OFT or model training.
scripts/run_r7a_exact_repeat.sh

scripts/run_r7a_build_source_dataset.sh
echo "R7A_DRIVER COMPLETE_SOURCE_LABELS_AND_EXACT_REPEAT"

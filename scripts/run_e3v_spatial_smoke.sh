#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
POOL=runs/rase_ui_phase0g_independent48_pool
PROTOCOL=runs/e3v_spatial_smoke4_protocol_v1.json
OUT=runs/e3v_spatial_smoke4_pi0fast_multiseed_v1
AUDIT=runs/e3v_spatial_smoke4_audit_v1.json

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PY" scripts/freeze_e3v_reference_roots.py \
  --opportunity runs/g2b_spatial16_opportunity_v1.json \
  --pool "$POOL" --output "$PROTOCOL" \
  --max-states 4 --rollouts-per-state 2

"$PY" -u scripts/collect_e3v_reference_oracle.py \
  --protocol "$PROTOCOL" \
  --policy-path ckpts/pi0fast_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
  --output-dir "$OUT"

"$PY" scripts/analyze_e3v_reference_oracle.py \
  --summary "$OUT/summary.json" --output "$AUDIT"

test -f "$AUDIT"

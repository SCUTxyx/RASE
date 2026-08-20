#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
PROTOCOL=configs/g2a_pi0fast_clean_long_v1.json
OUT=runs/oft_opportunity/g2a_pi0fast_clean_long_v1

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PY" -u scripts/eval_g2a_pi0fast_clean.py \
  --protocol "$PROTOCOL" \
  --output-dir "$OUT"

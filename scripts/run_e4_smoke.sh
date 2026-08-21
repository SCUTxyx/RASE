#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PY" -u scripts/e4_candidate_pool_audit.py \
  --config configs/g2a_pi0fast_clean_long_v1.json \
  --tasks 9 --episodes-per-task 1 --k 8 --temperature 0.7 \
  --smoke-k 2 --decision-step 10 \
  --output runs/e4_candidate_pool_audit_smoke

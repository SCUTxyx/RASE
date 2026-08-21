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
"$PY" -u scripts/e6_replan_freq.py \
  --config configs/g2a_pi0fast_clean_long_v1.json \
  --tasks 1,2,9 --init-states 0,1 \
  --sigmas 0.05 --ks 4,10 --horizon-cap 400 \
  --output runs/e6_replan_freq_v2

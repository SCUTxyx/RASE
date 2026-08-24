#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -u scripts/e7_stage0_pilot.py --pert 0.2 --suite libero_10 --roots 24 --k-new 4 \
  --output runs/e7_stage0_pilot_v1 > runs/e7_stage0_pilot.log 2>&1

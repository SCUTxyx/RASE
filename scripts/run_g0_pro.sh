#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -u scripts/g0_pro_baseline.py \
  --pert object --suite libero_object --tasks 10 --eps-per-task 8 \
  --policy ckpts/smolvla_libero --output runs/g0_pro_object_smolvla_v1

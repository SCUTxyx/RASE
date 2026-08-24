#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -u scripts/g0_pos_baseline.py --suite libero_object --tasks 10 --eps-per-task 8 --level 0.2 \
  --policy ckpts/smolvla_libero --tokenizer ckpts/SmolVLM2-500M-Instruct \
  --output runs/g0_pos_object_smolvla_l02_v1 > runs/g0_pos_object_smolvla_l02.log 2>&1
"$PY" -u scripts/g0_pos_baseline.py --suite libero_object --tasks 10 --eps-per-task 8 --level 0.3 \
  --policy ckpts/smolvla_libero --tokenizer ckpts/SmolVLM2-500M-Instruct \
  --output runs/g0_pos_object_smolvla_l03_v1 > runs/g0_pos_object_smolvla_l03.log 2>&1
echo ALL_POS_DONE

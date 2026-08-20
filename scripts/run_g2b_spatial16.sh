#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
POOL=runs/rase_ui_phase0g_independent48_pool
KEYS=runs/g2b_spatial16_keys_v1.json
SMOL=runs/g2b_spatial16_smol_continue_v1
PI0=runs/g2b_spatial16_pi0fast_direct_v1
OUT=runs/g2b_spatial16_opportunity_v1.json

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PY" scripts/export_decision_context_keys.py \
  --pool "$POOL" --output "$KEYS" --steps 0,2,4,6 \
  --task-id libero_spatial_000003 --task-id libero_spatial_000006 \
  --task-id libero_spatial_000007 --task-id libero_spatial_000008 \
  --expected-states 16

if [[ -d "$SMOL" ]]; then smol_mode=--resume; else smol_mode=--fresh-run; fi
"$PY" -u scripts/rollout_smol_interventions.py \
  --config configs/collect_rase_ui_phase0g_independent48.json \
  --state-keys-json "$KEYS" --output-dir "$SMOL" \
  --continuation-seeds 1 --profile continue-only "$smol_mode"

"$PY" -u scripts/rollout_lerobot_direct_from_pool.py \
  --pool "$POOL" --state-keys-json "$KEYS" \
  --policy-path ckpts/pi0fast_libero --policy-id pi0fast_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
  --output-dir "$PI0"

set +e
"$PY" scripts/analyze_continue_fallback_opportunity.py \
  --state-keys-json "$KEYS" --continue-summary "$SMOL/summary.json" \
  --fallback-summary "$PI0/summary.json" --output "$OUT" \
  --min-heterogeneity 0.05 --min-oracle-gain 0.05 \
  --bootstrap-replicates 10000 --bootstrap-seed 20260820
analysis_code=$?
set -e
if [[ "$analysis_code" != 0 && "$analysis_code" != 2 ]]; then
  exit "$analysis_code"
fi
test -f "$OUT"

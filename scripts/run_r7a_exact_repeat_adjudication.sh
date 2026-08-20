#!/usr/bin/env bash
# One diagnostic third repeat for the single formal v1 mismatch, then freeze its exclusion.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT="${R7_SOURCE_ROOT:-runs/pre_c0_r7/r7a_pi0fast_source_labels_v1}"
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json
BAD_KEY=sp1_5b2f2d114882fcce15f2a4be884ad084
REPEAT="$ROOT/exact_repeat"
ADJ="$ROOT/exact_repeat_adjudication_v1.json"
EXCLUSION="$ROOT/reproducibility_exclusions_v1.json"

if [[ ! -f "$REPEAT/suite_long/seed_0/${BAD_KEY}__seed0__rep2.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$KEYS" --policy-id pi0fast_libero \
    --policy-path ckpts/pi0fast_libero \
    --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
    --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
    --suite libero_10 --seed-index 0 --rollout-index 2 --no-oracle \
    --output-dir "$REPEAT/suite_long/seed_0" --boundary 0 --bookkeeping-mode full \
    --state-key "$BAD_KEY"
fi

"$PY" scripts/adjudicate_r7a_exact_repeat_failure.py \
  --original-audit "$ROOT/exact_repeat_audit.json" \
  --manifest "$ROOT/exact_repeat_manifest.json" \
  --label-audit "$ROOT/label_support.json" --input-root "$ROOT" \
  --repeat-root "$REPEAT" --output "$ADJ" --exclusion-output "$EXCLUSION"

#!/usr/bin/env bash
# Resume-safe A_PARTIAL π0-fast source-action pilot. No post-decision rollout.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OUT=runs/rase_vnext/phase_c_pi0fast_source_full_v1
COMMON=(
  --manifest runs/rase_vnext/frozen/confirmation_manifest_v1.json
  --branches runs/rase_vnext/confirmation_v1/branches.jsonl
  --output-dir "$OUT"
  --policy-id pi0fast.libero
  --policy-path ckpts/pi0fast_libero
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e
  --tasks-per-suite 0
  --max-replicas 3
  --hash-alignment-retries 5
  --operators continue.source requery.source
)

for suite in Spatial Object Goal Long; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -u scripts/export_rase_vnext_phase_c_features.py \
    "${COMMON[@]}" --suite "$suite"
done

set +e
"$PY" scripts/export_rase_vnext_phase_c_features.py "${COMMON[@]}" --summarize
summary_code=$?
"$PY" scripts/analyze_rase_vnext_phase_c_pilot.py \
  --feature-dir "$OUT" --output "$OUT/analysis.json" --bootstrap-replicates 10000
analysis_code=$?
set -e
printf 'PHASE_C_FULL summary_exit=%s analysis_exit=%s\n' "$summary_code" "$analysis_code"

# Non-zero analysis is an experimental verdict, not a runner integrity error.
test "$summary_code" -ne 3 || test -f "$OUT/collection_report.json"
test "$analysis_code" -ne 2 || test -f "$OUT/analysis.json"

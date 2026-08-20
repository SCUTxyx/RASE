#!/usr/bin/env bash
# Diagnostic only: verify post-source counterfactual collection preserves both
# the success and terminal step count of the R6-A Pi0Fast Spatial pair.
set -euo pipefail
cd /root/autodl-tmp/RASE

OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
ENDPOINT=tcp://127.0.0.1:5555
# Fresh immutable attempt directory (v1 recorded the pre-repair failure).
OUT=runs/pre_c0_r6/r6b1_smoke_pi0fast_spatial_pair_parity_v2
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
RASE_OFT_CHECKPOINT="$PWD/ckpts/oft_spatial" RASE_OFT_SUITE=libero_spatial \
"$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if "$VLA_PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite libero_spatial >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == 1 ]]

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
  --initial-keys runs/rase_ui_phase1a_replacement48_initial_keys_v2.json \
  --policy-path ckpts/pi0fast_libero --policy-id pi0fast_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
  --suite libero_spatial --seed-index 0 --endpoint "$ENDPOINT" \
  --output-dir "$OUT/data" --boundary 0 \
  --state-key sp1_0660d272e7256c6b204caf666e94c875 \
  --state-key sp1_0632d5ef6c45e2f304a01f2c133f0bfe \
  --bookkeeping-mode full

# Hard gate: every trajectory must reproduce the frozen R6-A reference exactly
# (rollout seed, final success, env steps) with finite features.
"$VLA_PY" scripts/audit_r6b1_source_parity.py \
  --atlas runs/pre_c0_r6/policy_pair_atlas_v1.json \
  --input-root "$OUT/data" --policy-id pi0fast_libero --seed-index 0 \
  --output "$OUT/data/parity_audit.json"

echo complete > "$OUT/COMPLETE"

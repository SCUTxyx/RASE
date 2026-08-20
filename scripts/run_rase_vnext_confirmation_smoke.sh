#!/usr/bin/env bash
# Two formal, resume-safe Spatial groups: one per source policy.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/confirmation_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/confirmation_v1
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
RASE_OFT_CHECKPOINT="$PWD/ckpts/oft_spatial" RASE_OFT_SUITE=libero_spatial \
"$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_spatial_smoke.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite libero_spatial >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  tail -100 "$OUT/oft_spatial_smoke.log" >&2 || true
  exit 31
fi

common=(
  --manifest "$MANIFEST" --protocol "$PROTOCOL" --output-dir "$OUT"
  --suite Spatial --endpoint tcp://127.0.0.1:5555 --max-groups 1
)
env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PY" -u scripts/collect_rase_vnext_discovery.py "${common[@]}" \
  --policy-id pi05.libero --policy-path ckpts/pi05_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46

env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PY" -u scripts/collect_rase_vnext_discovery.py "${common[@]}" \
  --policy-id pi0fast.libero --policy-path ckpts/pi0fast_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e


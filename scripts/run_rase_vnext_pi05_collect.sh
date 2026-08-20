#!/usr/bin/env bash
# RASE K3 formal collection: 432 operator slots / 360 simulator executions.
# Runs one suite at a time with its matched OFT oracle server; resumes are
# idempotent (existing group files are skipped by the collector).
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/pi05_challenge_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/pi05_collect_v1
mkdir -p "$OUT"

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
  esac
}
checkpoint() {
  case "$1" in
    Spatial) echo ckpts/oft_spatial ;;
    Object) echo ckpts/oft_object ;;
    Goal) echo ckpts/oft_goal ;;
    Long) echo ckpts/oft_10 ;;
  esac
}
cleanup_server() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/oft_${label,,}.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 90); do
    if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 \
      --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -100 "$server_log" >&2 || true
    exit 31
  fi
  echo "PI05 oracle ready: $label ($suite)"

  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/collect_rase_vnext_discovery.py \
    --manifest "$MANIFEST" \
    --protocol "$PROTOCOL" \
    --output-dir "$OUT" \
    --policy-path ckpts/pi05_libero \
    --policy-id pi05.libero \
    --suite "$label" \
    --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
     \
    --candidate-capture-dir "$OUT/captures"
  cleanup_server
done

echo "PI05 collection batch complete"

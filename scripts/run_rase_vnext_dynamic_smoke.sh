#!/usr/bin/env bash
# Phase 1 dynamic-boundary feasibility smoke (positive controls + controls).
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
OUT=runs/rase_vnext/dynamic_smoke_v1
mkdir -p "$OUT"

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

run_suite() {
  local suite=$1 ckpt=$2 actual=$3
  "$PY" - "$suite" <<'PY'
import json, sys
roots = json.load(open("/tmp/dyn_roots.json"))
suite = sys.argv[1]
sub = [r for r in roots if r["suite"] == suite]
json.dump(sub, open(f"/tmp/dyn_roots_{suite}.json", "w"))
print("suite", suite, "roots:", [(r["label"], r["task"]) for r in sub])
PY
  local count
  count=$("$PY" -c "import json; print(len(json.load(open('/tmp/dyn_roots_$suite.json'))))")
  if [[ "$count" == "0" ]]; then
    echo "no roots for $suite, skip"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$actual" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_${suite,,}.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 90); do
    if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 \
      --expect-suite "$actual" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -100 "$OUT/oft_${suite,,}.log" >&2 || true
    exit 31
  fi
  echo "oracle ready for $suite"

  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/run_rase_vnext_dynamic_smoke.py \
    --roots-file "/tmp/dyn_roots_$suite.json" \
    --output "$OUT/${suite,,}.json" \
    --pool runs/pre_c0_r7/r7a_pi0fast_reset_pool_v1 \
    --policy-path ckpts/pi0fast_libero \
    --policy-id pi0fast.libero \
    --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
    --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e
  cleanup_server
}

run_suite Goal ckpts/oft_goal libero_goal
run_suite Object ckpts/oft_object libero_object
run_suite Spatial ckpts/oft_spatial libero_spatial

echo "dynamic smoke complete"

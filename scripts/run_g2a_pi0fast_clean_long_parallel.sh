#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
PROTOCOL=configs/g2a_pi0fast_clean_long_v1.json
OUT=runs/oft_opportunity/g2a_pi0fast_clean_long_v1

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CLEAN_ROOT=/root/autodl-tmp/src/LIBERO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"
# Two workers fit the GPU. Stagger checkpoint loading to avoid simultaneous
# 7.7-GB reads making the SSH host temporarily unresponsive.
"$PY" -u scripts/eval_g2a_pi0fast_clean.py \
  --protocol "$PROTOCOL" --output-dir "$OUT" \
  --start-index 0 --end-index 40 > "$OUT/shard_0_40.log" 2>&1 &
pids=("$!")
for _ in $(seq 1 120); do
  grep -q "All keys loaded successfully" "$OUT/shard_0_40.log" && break
  sleep 5
done
if ! grep -q "All keys loaded successfully" "$OUT/shard_0_40.log"; then
  echo "G2A first worker did not finish loading; refusing concurrent launch" >&2
  wait "${pids[0]}" || true
  exit 1
fi
"$PY" -u scripts/eval_g2a_pi0fast_clean.py \
  --protocol "$PROTOCOL" --output-dir "$OUT" \
  --start-index 40 --end-index 80 > "$OUT/shard_40_80.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" != 0 ]]; then
  echo "G2A shard failure; inspect $OUT/shard_*.log" >&2
  exit "$status"
fi

"$PY" scripts/eval_g2a_pi0fast_clean.py \
  --protocol "$PROTOCOL" --output-dir "$OUT" --summary-only

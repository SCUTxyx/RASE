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
pids=()
for range in 0:27 27:54 54:80; do
  start=${range%%:*}
  end=${range##*:}
  "$PY" -u scripts/eval_g2a_pi0fast_clean.py \
    --protocol "$PROTOCOL" --output-dir "$OUT" \
    --start-index "$start" --end-index "$end" \
    > "$OUT/shard_${start}_${end}.log" 2>&1 &
  pids+=("$!")
done

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

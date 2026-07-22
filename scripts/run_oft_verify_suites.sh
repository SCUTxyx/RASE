#!/usr/bin/env bash
# Suite-serial OFT verify for a W3 pilot config (GPU0 server, GPU1 client).
# Usage: ./scripts/run_oft_verify_suites.sh configs/ngc_w3_pilot_adequate_early.yaml adequate_early
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="${1:?config yaml}"
TAG="${2:?output tag suffix}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"

kill_oft_server() {
  local pids
  pids=$(pgrep -f 'python -m rase.oracle.server' || true)
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill ${pids} || true
    sleep 2
  fi
}

run_suite() {
  local suite="$1" ckpt="$2" short="$3"
  echo "=== OFT suite=${suite} ckpt=${ckpt} ==="
  kill_oft_server
  (
    source /data/data2/yuxuan/miniconda3/etc/profile.d/conda.sh
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/data/data2/yuxuan/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_${short}_${TAG}.log" 2>&1
  ) &
  local server_pid=$!
  sleep 45
  (
    source /data/data2/yuxuan/miniconda3/etc/profile.d/conda.sh
    conda activate smolvla
    export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
    export HF_HOME=/data/data2/yuxuan/hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite"
    python -u scripts/rollout_pool_candidates.py \
      --config "$CFG" --mode oft-verify \
      --suite "$suite" --endpoint "$ENDPOINT" \
      --output-dir "runs/ngc_w3_oft_${short}_${TAG}" \
      --force-new-run
  )
  kill "$server_pid" 2>/dev/null || true
  kill_oft_server
}

run_suite libero_spatial ckpts/oft_spatial spatial
run_suite libero_object ckpts/oft_object object
run_suite libero_goal ckpts/oft_goal goal
run_suite libero_10 ckpts/oft_10 10
echo "ALL_OFT_SUITES_DONE tag=${TAG}"

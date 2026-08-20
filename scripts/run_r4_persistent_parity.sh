#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SUITE_SHORT="${SUITE_SHORT:?set SUITE_SHORT to spatial/object/goal/10}"
STATE_KEY="${STATE_KEY:?set STATE_KEY}"
REPEATS="${REPEATS:-3}"
OUTPUT="${OUTPUT:-runs/pre_c0_r4/parity_${SUITE_SHORT}_${STATE_KEY}.json}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

case "$SUITE_SHORT" in
  spatial) SUITE=libero_spatial; CKPT=ckpts/oft_spatial ;;
  object) SUITE=libero_object; CKPT=ckpts/oft_object ;;
  goal) SUITE=libero_goal; CKPT=ckpts/oft_goal ;;
  10) SUITE=libero_10; CKPT=ckpts/oft_10 ;;
  *) echo "invalid SUITE_SHORT=$SUITE_SHORT" >&2; exit 2 ;;
esac

(
  conda activate "${OFT_ENV:-oft}"
  export CUDA_VISIBLE_DEVICES="${SERVER_CUDA:-0}"
  export PYTHONPATH="${PYTHONPATH_OFT:-/root/autodl-tmp/src/openvla-oft}:${PYTHONPATH:-}"
  export RASE_OFT_CHECKPOINT="$ROOT/$CKPT"
  export RASE_OFT_SUITE="$SUITE"
  exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    > "${OUTPUT%.json}_server.log" 2>&1
) &
server_pid=$!
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if conda run -n "${SMOLVLA_ENV:-smolvla}" python scripts/probe_oracle.py \
    --endpoint "$ENDPOINT" --expect-suite "$SUITE" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  echo "oracle server not ready" >&2
  exit 1
fi

(
  conda activate "${SMOLVLA_ENV:-smolvla}"
  export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA:-0}"
  export MUJOCO_EGL_DEVICE_ID="${CLIENT_CUDA:-0}"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
  python -u scripts/replay_r4_persistent_parity.py \
    --config configs/pre_a3_recovery_duration120.yaml \
    --state-key "$STATE_KEY" \
    --suite "$SUITE" \
    --endpoint "$ENDPOINT" \
    --repeats "$REPEATS" \
    --output "$OUTPUT"
)

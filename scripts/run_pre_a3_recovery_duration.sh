#!/usr/bin/env bash
# PRE-A3 live closed-loop recovery-duration confirmatory pipeline.
# Requires frozen keys at KEYS and a collected pool referenced by CONFIG.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/pre_a3_recovery_duration120.yaml}"
KEYS="${KEYS:-runs/rase_pre_a3_keys120_v1.json}"
OUTPUT="${OUTPUT:-runs/rase_pre_a3_recovery_duration120_v1}"
ANALYSIS="${ANALYSIS:-runs/rase_pre_a3_recovery_duration_audit120_v1}"
LOG="${LOG:-runs/rase_pre_a3_recovery_duration120_v1.log}"
FRESH_RUN="${FRESH_RUN:-1}"
SPLIT_FILTER="${SPLIT_FILTER:-}"  # empty = all splits; or train/val/test
PREFIX_ARGS=(--prefix-length 0 --prefix-length 8 --prefix-length 16 --prefix-length 32 --prefix-length 64 --prefix-length 96 --prefix-length 128)
SMOKE_N="${SMOKE_N:-0}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ ! -f "$KEYS" ]]; then
  echo "ERROR: missing frozen keys: $KEYS" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" && -e "$OUTPUT" ]]; then
  echo "ERROR: fresh output exists: $OUTPUT" >&2
  exit 1
fi

mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

echo "=== PRE-A3 LIVE CLOSED-LOOP DURATION ==="
echo "config=$CONFIG keys=$KEYS output=$OUTPUT"

SUITE_SHORTS=(spatial object goal 10)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
OFT_ENV="${OFT_ENV:-oft}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
PYTHONPATH_OFT="${PYTHONPATH_OFT:-/root/autodl-tmp/src/openvla-oft}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
SERVER_CUDA="${SERVER_CUDA:-0}"
CLIENT_CUDA="${CLIENT_CUDA:-0}"

summaries=()
for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"
  suite_out="${OUTPUT}/suite_${short}"
  if [[ "$FRESH_RUN" == "1" && -e "$suite_out" ]]; then
    echo "ERROR: fresh suite output exists: $suite_out" >&2
    exit 1
  fi

  echo "=== start OFT server suite=${suite} ==="
  (
    conda activate "${OFT_ENV}"
    export CUDA_VISIBLE_DEVICES="${SERVER_CUDA}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="${PYTHONPATH_OFT}:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_pre_a3_${short}.log" 2>&1
  ) &
  server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  for _ in $(seq 1 60); do
    if conda run -n "$SMOLVLA_ENV" python scripts/probe_oracle.py \
      --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  conda run -n "$SMOLVLA_ENV" python scripts/probe_oracle.py \
    --endpoint "$ENDPOINT" --expect-suite "$suite"

  run_args=()
  if [[ "$FRESH_RUN" == "1" ]]; then
    run_args+=(--fresh-run)
  fi
  if [[ -n "$SPLIT_FILTER" ]]; then
    run_args+=(--split "$SPLIT_FILTER")
  fi

  (
    conda activate "${SMOLVLA_ENV}"
    export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA}"
    export MUJOCO_EGL_DEVICE_ID="${CLIENT_CUDA}"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_PLUS_ROOT
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    python -u scripts/rollout_live_oft_duration_to_smol.py \
      --config "$CONFIG" \
      --state-keys-json "$KEYS" \
      --suite "$suite" \
      --endpoint "$ENDPOINT" \
      --output-dir "$suite_out" \
      --include-persistent-oft \
      "${PREFIX_ARGS[@]}" \
      "${run_args[@]}"
  )
  summaries+=("$suite_out/summary_${suite}.json")
  cleanup
  trap - EXIT
done

echo "=== MERGE SUITE SUMMARIES ==="
merge_args=()
for path in "${summaries[@]}"; do
  merge_args+=(--input "$path")
done
"$PY" scripts/merge_live_duration_summaries.py \
  "${merge_args[@]}" \
  --output "$OUTPUT/summary.json"

echo "=== PRE-A3 AUDIT + METHOD GATE ==="
"$PY" scripts/analyze_pre_a3_recovery_duration.py \
  --duration-summary "$OUTPUT/summary.json" \
  --state-keys-json "$KEYS" \
  --output-dir "$ANALYSIS"

echo "PRE_A3_DURATION_DONE analysis=$ANALYSIS gate=$ANALYSIS/method_gate.json"

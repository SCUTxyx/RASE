#!/usr/bin/env bash
# Plumbing-only live closed-loop smoke for PRE-A3 runner validation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/pre_a3_smoke4.yaml}"
export KEYS="${KEYS:-runs/rase_pre_a3_smoke4_keys_v1.json}"
export OUTPUT="${OUTPUT:-runs/rase_pre_a3_smoke4_live_duration_v1}"
export ANALYSIS="${ANALYSIS:-runs/rase_pre_a3_smoke4_audit_v1}"
export LOG="${LOG:-runs/rase_pre_a3_smoke4_live_duration_v1.log}"
export FRESH_RUN="${FRESH_RUN:-1}"
export OFT_SUITE_SHORTS="${OFT_SUITE_SHORTS:-spatial,object}"

# Narrow duration set for smoke.
tmp_runner="$ROOT/scripts/run_pre_a3_recovery_duration.sh"
# Override prefix args via env consumed after patching local copy behavior:
export PREFIX_LENGTHS="${PREFIX_LENGTHS:-0,8,16,32}"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
if [[ "$FRESH_RUN" == "1" && -e "$OUTPUT" ]]; then
  echo "ERROR: fresh output exists: $OUTPUT" >&2
  exit 1
fi
mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

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

IFS=',' read -r -a lengths <<< "$PREFIX_LENGTHS"
prefix_args=()
for h in "${lengths[@]}"; do
  prefix_args+=(--prefix-length "$h")
done

declare -A SUITE_MAP=(
  [spatial]=libero_spatial
  [object]=libero_object
  [goal]=libero_goal
  [10]=libero_10
)
declare -A CKPT_MAP=(
  [spatial]=ckpts/oft_spatial
  [object]=ckpts/oft_object
  [goal]=ckpts/oft_goal
  [10]=ckpts/oft_10
)

IFS=',' read -r -a shorts <<< "$OFT_SUITE_SHORTS"
summaries=()
for short in "${shorts[@]}"; do
  suite="${SUITE_MAP[$short]}"
  ckpt="${CKPT_MAP[$short]}"
  suite_out="${OUTPUT}/suite_${short}"
  if [[ "$FRESH_RUN" == "1" && -e "$suite_out" ]]; then
    echo "ERROR: fresh suite output exists: $suite_out" >&2
    exit 1
  fi
  echo "=== smoke OFT server suite=${suite} ==="
  (
    conda activate "${OFT_ENV}"
    export CUDA_VISIBLE_DEVICES="${SERVER_CUDA}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="${PYTHONPATH_OFT}:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_pre_a3_smoke_${short}.log" 2>&1
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
      "${prefix_args[@]}" \
      "${run_args[@]}"
  )
  summaries+=("$suite_out/summary_${suite}.json")
  cleanup
  trap - EXIT
  FRESH_RUN=0  # subsequent suites resume into same OUTPUT tree
done

merge_args=()
for path in "${summaries[@]}"; do
  merge_args+=(--input "$path")
done
"$PY" scripts/merge_live_duration_summaries.py \
  "${merge_args[@]}" \
  --output "$OUTPUT/summary.json"

"$PY" scripts/analyze_pre_a3_recovery_duration.py \
  --duration-summary "$OUTPUT/summary.json" \
  --state-keys-json "$KEYS" \
  --output-dir "$ANALYSIS" \
  --bootstrap-replicates 1000

echo "PRE_A3_SMOKE_DONE output=$OUTPUT analysis=$ANALYSIS"

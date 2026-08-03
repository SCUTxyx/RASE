#!/usr/bin/env bash
# Suite-serial OFT verify for a W3/W4 pilot config.
# Usage:
#   ./scripts/run_oft_verify_suites.sh configs/ngc_w4_adequate_scale.yaml adequate
# Env overrides:
#   ENDPOINT, CONDA_ROOT, SMOLVLA_ENV, OFT_ENV, LIBERO_PLUS_ROOT, PYTHONPATH_OFT
#   OUTPUT_PREFIX (default: ngc_w4_oft) → runs/${OUTPUT_PREFIX}_${short}_${TAG}
#   STATE_KEYS_JSON, CANDIDATES_DIR, FRESH_RUN=1, HEALTH_RETRIES, HEALTH_INTERVAL
#   OFT_RUNNER=verify|prefix-ablation|generate-prefix
#   OFT_PREFIX_ARMS=full|direct|decision-suffix|suffix-prefix-grid
#   OFT_SUITE_SHORTS=spatial,object,goal,10
#   PREFLIGHT=0 skips the default read-only environment/artifact checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="${1:?config yaml}"
TAG="${2:?output tag suffix}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ngc_w4_oft}"
STATE_KEYS_JSON="${STATE_KEYS_JSON:-runs/ngc_w4_adequate_state_keys.json}"
CANDIDATES_DIR="${CANDIDATES_DIR:-runs/ngc_w4_adequate_candidates}"
FRESH_RUN="${FRESH_RUN:-0}"
HEALTH_RETRIES="${HEALTH_RETRIES:-30}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"
PREFLIGHT="${PREFLIGHT:-1}"
OFT_MIN_FREE_MIB="${OFT_MIN_FREE_MIB:-20000}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
OFT_RUNNER="${OFT_RUNNER:-verify}"
OFT_PREFIX_ARMS="${OFT_PREFIX_ARMS:-full}"
OFT_SUITE_SHORTS="${OFT_SUITE_SHORTS:-spatial,object,goal,10}"
# Single-GPU default: OFT server and lightweight environment client share GPU0.
# Dual-GPU: set CLIENT_CUDA=1 SERVER_CUDA=0.
SERVER_CUDA="${SERVER_CUDA:-0}"
CLIENT_CUDA="${CLIENT_CUDA:-0}"

# Prefer AutoDL layout; fall back to historical /data/data2/yuxuan paths.
CONDA_ROOT="${CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]]; then
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_ROOT=/root/miniconda3
  elif [[ -f /data/data2/yuxuan/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_ROOT=/data/data2/yuxuan/miniconda3
  else
    echo "ERROR: cannot find conda.sh; set CONDA_ROOT" >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
OFT_ENV="${OFT_ENV:-oft}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-}"
if [[ -z "${LIBERO_PLUS_ROOT}" ]]; then
  if [[ -d /root/autodl-tmp/src/LIBERO-plus ]]; then
    LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus
  elif [[ -d /root/autodl-tmp/LIBERO-plus ]]; then
    LIBERO_PLUS_ROOT=/root/autodl-tmp/LIBERO-plus
  elif [[ -d /data/data2/yuxuan/LIBERO-plus ]]; then
    LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
  fi
fi
PYTHONPATH_OFT="${PYTHONPATH_OFT:-}"
if [[ -z "${PYTHONPATH_OFT}" ]]; then
  if [[ -d /root/autodl-tmp/src/openvla-oft ]]; then
    PYTHONPATH_OFT=/root/autodl-tmp/src/openvla-oft
  elif [[ -d /data/data2/yuxuan/openvla-oft ]]; then
    PYTHONPATH_OFT=/data/data2/yuxuan/openvla-oft
  fi
fi

if [[ "$PREFLIGHT" == "1" ]]; then
  preflight_args=(
    scripts/preflight_runner.py
    --conda-root "$CONDA_ROOT" \
    --smolvla-env "$SMOLVLA_ENV" \
    --oft-env "$OFT_ENV" \
    --libero-plus-root "$LIBERO_PLUS_ROOT" \
    --checkpoints-root "$ROOT/ckpts" \
    --gpu-index "$SERVER_CUDA" \
    --min-free-gpu-mib "$OFT_MIN_FREE_MIB"
  )
  if [[ "$ALLOW_BUSY_GPU" == "1" ]]; then
    preflight_args+=(--allow-busy-gpu)
  elif [[ "$ALLOW_BUSY_GPU" != "0" ]]; then
    echo "ERROR: ALLOW_BUSY_GPU must be 0 or 1" >&2
    exit 1
  fi
  "${CONDA_ROOT}/bin/python" "${preflight_args[@]}"
elif [[ "$PREFLIGHT" != "0" ]]; then
  echo "ERROR: PREFLIGHT must be 0 or 1" >&2
  exit 1
fi

LOCK_FILE="${LOCK_FILE:-runs/${OUTPUT_PREFIX}_${TAG}.lock}"
mkdir -p runs
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  echo "ERROR: another OFT suite runner holds ${LOCK_FILE}" >&2
  exit 1
fi

if [[ ! -f "$STATE_KEYS_JSON" ]]; then
  echo "ERROR: frozen state key artifact missing: $STATE_KEYS_JSON" >&2
  exit 1
fi
if [[ "$OFT_RUNNER" == "verify" \
  || ( "$OFT_RUNNER" == "prefix-ablation" && "$OFT_PREFIX_ARMS" == "full" ) ]] \
  && [[ ! -d "$CANDIDATES_DIR" ]]; then
  echo "ERROR: candidates directory missing: $CANDIDATES_DIR" >&2
  exit 1
fi
if [[ "$OFT_RUNNER" != "verify" && "$OFT_RUNNER" != "prefix-ablation" \
  && "$OFT_RUNNER" != "generate-prefix" ]]; then
  echo "ERROR: OFT_RUNNER must be verify, prefix-ablation, or generate-prefix" >&2
  exit 1
fi
if [[ "$OFT_PREFIX_ARMS" != "full" && "$OFT_PREFIX_ARMS" != "direct" \
  && "$OFT_PREFIX_ARMS" != "decision-suffix" \
  && "$OFT_PREFIX_ARMS" != "suffix-prefix-grid" ]]; then
  echo "ERROR: OFT_PREFIX_ARMS must be full, direct, decision-suffix, or suffix-prefix-grid" >&2
  exit 1
fi

current_server_pid=""
cleanup_server() {
  local pid="${current_server_pid}"
  current_server_pid=""
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    local attempt
    for attempt in 1 2 3 4 5; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT TERM

wait_for_health() {
  local suite="$1" attempt
  for ((attempt = 1; attempt <= HEALTH_RETRIES; attempt++)); do
    if ! kill -0 "$current_server_pid" 2>/dev/null; then
      echo "ERROR: OFT server exited before becoming healthy (suite=${suite})" >&2
      return 1
    fi
    if (
      conda activate "${SMOLVLA_ENV}"
      python scripts/probe_oracle.py --endpoint "$ENDPOINT" \
        --expect-suite "$suite" --timeout-ms 2000 --skip-predict
    ) >/dev/null 2>&1; then
      echo "OFT_HEALTHY suite=${suite} attempt=${attempt}"
      return 0
    fi
    sleep "$HEALTH_INTERVAL"
  done
  echo "ERROR: OFT health check timed out for suite=${suite}" >&2
  return 1
}

run_suite() {
  local suite="$1" ckpt="$2" short="$3"
  local out_dir="runs/${OUTPUT_PREFIX}_${short}_${TAG}"
  if [[ -f "${out_dir}/summary.json" && "$FRESH_RUN" != "1" ]]; then
    echo "SKIP_COMPLETED suite=${suite} summary=${out_dir}/summary.json"
    return 0
  fi
  local suite_status=0
  (
    conda activate "${SMOLVLA_ENV}"
    python scripts/check_state_key_suite.py \
      --config "$CFG" --state-keys-json "$STATE_KEYS_JSON" --suite "$suite"
  ) || suite_status=$?
  if [[ "$suite_status" == "4" ]]; then
    echo "SKIP_EMPTY_SUITE suite=${suite}"
    return 0
  elif [[ "$suite_status" != "0" ]]; then
    echo "ERROR: suite membership audit failed for ${suite}" >&2
    return "$suite_status"
  fi
  echo "=== OFT suite=${suite} ckpt=${ckpt} out=${out_dir} ==="
  (
    conda activate "${OFT_ENV}"
    export CUDA_VISIBLE_DEVICES="${SERVER_CUDA}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="${PYTHONPATH_OFT}:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_${short}_${TAG}.log" 2>&1
  ) &
  current_server_pid=$!
  wait_for_health "$suite"
  local run_behavior=(--resume)
  if [[ "$FRESH_RUN" == "1" ]]; then
    run_behavior=(--fresh-run)
  fi
  (
    conda activate "${SMOLVLA_ENV}"
    export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA}"
    export MUJOCO_EGL_DEVICE_ID="${CLIENT_CUDA}"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_PLUS_ROOT
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    python scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite"
    if [[ "$OFT_RUNNER" == "verify" ]]; then
      python -u scripts/rollout_pool_candidates.py \
        --config "$CFG" --mode oft-verify \
        --suite "$suite" --endpoint "$ENDPOINT" \
        --state-keys-json "$STATE_KEYS_JSON" \
        --candidates-dir "$CANDIDATES_DIR" \
        --output-dir "${out_dir}" \
        "${run_behavior[@]}"
    elif [[ "$OFT_RUNNER" == "prefix-ablation" ]]; then
      python -u scripts/rollout_oft_prefix_ablation.py \
        --config "$CFG" \
        --suite "$suite" --endpoint "$ENDPOINT" \
        --state-keys-json "$STATE_KEYS_JSON" \
        --candidates-dir "$CANDIDATES_DIR" \
        --output-dir "${out_dir}" \
        --arms "$OFT_PREFIX_ARMS" \
        "${run_behavior[@]}"
    else
      python -u scripts/generate_oft_pool_candidates.py \
        --config "$CFG" \
        --suite "$suite" --endpoint "$ENDPOINT" \
        --state-keys-json "$STATE_KEYS_JSON" \
        --output-dir "$CANDIDATES_DIR" \
        --summary-output "${out_dir}/summary.json" \
        "${run_behavior[@]}"
    fi
  )
  cleanup_server
}

IFS=',' read -r -a selected_suite_shorts <<< "$OFT_SUITE_SHORTS"
if [[ "${#selected_suite_shorts[@]}" -eq 0 ]]; then
  echo "ERROR: OFT_SUITE_SHORTS must select at least one suite" >&2
  exit 1
fi

if [[ "$FRESH_RUN" == "1" ]]; then
  for short in "${selected_suite_shorts[@]}"; do
    out_dir="runs/${OUTPUT_PREFIX}_${short}_${TAG}"
    if [[ -e "$out_dir" ]]; then
      echo "ERROR: FRESH_RUN=1 requires new output directories; found $out_dir" >&2
      exit 1
    fi
  done
elif [[ "$FRESH_RUN" != "0" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 1
fi

for short in "${selected_suite_shorts[@]}"; do
  case "$short" in
    spatial) run_suite libero_spatial ckpts/oft_spatial spatial ;;
    object) run_suite libero_object ckpts/oft_object object ;;
    goal) run_suite libero_goal ckpts/oft_goal goal ;;
    10|long) run_suite libero_10 ckpts/oft_10 10 ;;
    *)
      echo "ERROR: unknown OFT suite short: $short" >&2
      exit 1
      ;;
  esac
done
echo "ALL_OFT_SUITES_DONE tag=${TAG} prefix=${OUTPUT_PREFIX}"

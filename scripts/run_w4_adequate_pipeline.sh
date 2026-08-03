#!/usr/bin/env bash
# End-to-end W4 ADEQUATE dual-oracle pipeline:
#   (1) SmolVLA primary  (2) OFT 4-suite verify  (3) yield + causal summary
#
# Usage:
#   ./scripts/run_w4_adequate_pipeline.sh
#   WAIT_PID=25646 ./scripts/run_w4_adequate_pipeline.sh   # wait for existing SmolVLA then continue
#   SKIP_SMOLVLA=1 ./scripts/run_w4_adequate_pipeline.sh   # only OFT + summarize
#   SUMMARY_ONLY=1 ./scripts/run_w4_adequate_pipeline.sh   # no rollout/model processes
#
# Env: CONDA_ROOT, SMOLVLA_ENV, LIBERO_PLUS_ROOT, HF_HOME, CUDA_VISIBLE_DEVICES, …
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-configs/ngc_w4_adequate_scale.yaml}"
TAG="${TAG:-adequate}"
KEYS_JSON="${KEYS_JSON:-runs/ngc_w4_adequate_state_keys.json}"
SMOL_OUT="${SMOL_OUT:-runs/ngc_w4_pilot_adequate}"
CAND_DIR="${CAND_DIR:-runs/ngc_w4_adequate_candidates}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ngc_w4_oft}"
LOG_DIR="${LOG_DIR:-runs/ngc_w4_pipeline_logs}"
mkdir -p "$LOG_DIR" runs

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
conda activate "${SMOLVLA_ENV}"

export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/pipeline.log"; }
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
if [[ "$SUMMARY_ONLY" != "0" && "$SUMMARY_ONLY" != "1" ]]; then
  echo "ERROR: SUMMARY_ONLY must be 0 or 1" >&2
  exit 1
fi

# Prevent accidental double pipeline (same CFG writing same scheduler).
LOCK_FILE="${LOCK_FILE:-$LOG_DIR/pipeline.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another run_w4_adequate_pipeline.sh holds $LOCK_FILE" >&2
  echo "If stale, check: pgrep -af run_w4_adequate_pipeline" >&2
  exit 1
fi

wait_for_pid() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    log "WAIT_PID=${pid} already exited"
    return 0
  fi
  log "Waiting for PID ${pid} (existing SmolVLA primary)…"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  local ec=0
  wait "${pid}" 2>/dev/null || ec=$?
  # wait only works for children; if not our child, check summary instead
  log "PID ${pid} gone (wait_ec=${ec})"
}

if [[ "$SUMMARY_ONLY" == "1" ]]; then
  log "SUMMARY_ONLY=1: skipping candidates and both oracle runners"
else
  # --- optional: wait for in-flight SmolVLA started outside this script ---
  if [[ -n "${WAIT_PID:-}" ]]; then
    wait_for_pid "${WAIT_PID}"
  fi

  # --- 0) candidates (consume one frozen key artifact) ---
  if [[ ! -f "${KEYS_JSON}" ]]; then
    log "Sampling and freezing ADEQUATE keys → ${KEYS_JSON}"
    python scripts/sample_adequate_keys.py \
      --config "${CFG}" --output "${KEYS_JSON}"
  fi
  expected_cand=$(python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(len(p if isinstance(p,list) else p["state_keys"]))' \
    "${KEYS_JSON}")
  shopt -s nullglob
  candidate_files=("${CAND_DIR}"/sp1_*.npz)
  n_cand="${#candidate_files[@]}"
  shopt -u nullglob
  if [[ "${n_cand}" -lt "${expected_cand}" ]]; then
    log "Generating K=8 candidates from frozen keys → ${CAND_DIR}"
    python scripts/generate_pool_candidates.py \
      --config "${CFG}" --state-keys-json "${KEYS_JSON}" \
      --output-dir "${CAND_DIR}" \
      2>&1 | tee -a "$LOG_DIR/candidates.log"
  else
    log "Skip candidates (${n_cand}/${expected_cand} artifacts present)"
  fi

  # --- 1) SmolVLA primary ---
  if [[ "${SKIP_SMOLVLA:-0}" == "1" ]]; then
    log "SKIP_SMOLVLA=1"
  elif [[ -f "${SMOL_OUT}/summary.json" \
    && "${FORCE_SMOLVLA:-0}" != "1" \
    && "${FRESH_RUN:-0}" != "1" ]]; then
    log "Skip completed SmolVLA suite (found ${SMOL_OUT}/summary.json)"
  else
    log "Starting/resuming SmolVLA primary → ${SMOL_OUT}"
    run_behavior=(--resume)
    if [[ "${FRESH_RUN:-0}" == "1" ]]; then
      run_behavior=(--fresh-run)
    fi
    python -u scripts/rollout_pool_candidates.py --config "${CFG}" \
      --state-keys-json "${KEYS_JSON}" --candidates-dir "${CAND_DIR}" \
      --output-dir "${SMOL_OUT}" "${run_behavior[@]}" \
      2>&1 | tee -a "$LOG_DIR/smolvla.log"
  fi

  if [[ ! -f "${SMOL_OUT}/summary.json" ]]; then
    log "ERROR: SmolVLA summary missing: ${SMOL_OUT}/summary.json"
    exit 1
  fi
  log "SmolVLA done."

  # --- 2) OFT four suites ---
  if [[ "${SKIP_OFT:-0}" == "1" ]]; then
    log "SKIP_OFT=1"
  else
    log "Starting/resuming OFT verify suites tag=${TAG}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}" \
      STATE_KEYS_JSON="${KEYS_JSON}" \
      CANDIDATES_DIR="${CAND_DIR}" \
      ./scripts/run_oft_verify_suites.sh "${CFG}" "${TAG}" \
      2>&1 | tee -a "$LOG_DIR/oft.log"
  fi
fi

# Both normal and summary-only modes require complete source summaries.
if [[ ! -f "${SMOL_OUT}/summary.json" ]]; then
  log "ERROR: SmolVLA summary missing: ${SMOL_OUT}/summary.json"
  exit 1
fi
for short in spatial object goal 10; do
  sum="runs/${OUTPUT_PREFIX}_${short}_${TAG}/summary.json"
  if [[ ! -f "${sum}" ]]; then
    log "ERROR: missing OFT summary ${sum}"
    exit 1
  fi
done
log "OFT done."

# --- 3) dual-oracle + causal ---
log "Summarizing dual-oracle yield"
python scripts/summarize_dual_oracle_yield.py \
  --smolvla-summary "${SMOL_OUT}/summary.json" \
  --oft-summary "libero_spatial=runs/${OUTPUT_PREFIX}_spatial_${TAG}/summary.json" \
  --oft-summary "libero_object=runs/${OUTPUT_PREFIX}_object_${TAG}/summary.json" \
  --oft-summary "libero_goal=runs/${OUTPUT_PREFIX}_goal_${TAG}/summary.json" \
  --oft-summary "libero_10=runs/${OUTPUT_PREFIX}_10_${TAG}/summary.json" \
  --pool pool/ngc_step1_scale200 \
  --output runs/ngc_w4_adequate_dual_oracle_summary.json \
  --markdown progress/2026-07-26_ngc_w4_adequate_scale.md \
  --causal-out runs/ngc_w4_adequate_causal_yield.json \
  --causal-markdown runs/ngc_w4_adequate_causal_yield.md \
  --ngc-oracle both \
  2>&1 | tee -a "$LOG_DIR/summarize.log"

log "PIPELINE_DONE"
log "  dual-oracle: runs/ngc_w4_adequate_dual_oracle_summary.json"
log "  progress:    progress/2026-07-26_ngc_w4_adequate_scale.md"

#!/usr/bin/env bash
set -euo pipefail

repo_root="${RASE_REPO_ROOT:-/root/autodl-tmp/RASE}"
endpoint="${RASE_OFT_ENDPOINT:-tcp://127.0.0.1:5555}"
partition="${RASE_E3B_PARTITION:-runs/e3b_pre_a3_partitions_v1/b0_smoke.json}"
output_root="${RASE_E3B_OUTPUT_ROOT:-runs/e3b_b1_teacher_calibration_v1}"
repeat="${RASE_E3B_REPEAT:-a}"

cd "$repo_root"
source /root/miniconda3/etc/profile.d/conda.sh

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  conda activate smolvla
  for _ in $(seq 1 120); do
    if python - "$endpoint" <<'PY' >/dev/null 2>&1
import sys
from rase.oracle.client import OracleClient

client = OracleClient(sys.argv[1], timeout_ms=1000)
try:
    client.model_info()
finally:
    client.close()
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "OFT server did not become ready" >&2
  return 1
}

run_suite() {
  local suite="$1"
  local checkpoint="$2"
  local label="$3"
  local output="$output_root/$repeat/$label"

  if [[ -f "$output/summary.json" ]]; then
    echo "E3B_TEACHER skip suite=$suite repeat=$repeat summary=$output/summary.json"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "partial output exists; resume explicitly or remove after audit: $output" >&2
    return 2
  fi

  conda activate oft
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
  export RASE_OFT_CHECKPOINT="$repo_root/$checkpoint"
  export RASE_OFT_SUITE="$suite"
  python -m rase.oracle.server \
    --endpoint "$endpoint" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    >"/tmp/e3b_oft_${label}_${repeat}.log" 2>&1 &
  server_pid="$!"
  wait_for_server

  conda activate smolvla
  python scripts/rollout_oft_prefix_ablation.py \
    --config configs/e3b_pre_a3_train_v1.json \
    --state-keys-json "$partition" \
    --output-dir "$output" \
    --suite "$suite" \
    --endpoint "$endpoint" \
    --arms direct \
    --fresh-run

  cleanup
  server_pid=""
}

run_suite libero_spatial ckpts/oft_spatial spatial
run_suite libero_object ckpts/oft_object object
run_suite libero_goal ckpts/oft_goal goal
run_suite libero_10 ckpts/oft_10 long

echo "E3B_TEACHER complete repeat=$repeat output_root=$output_root"

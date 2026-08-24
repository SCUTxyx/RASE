#!/usr/bin/env bash
set -euo pipefail

repo_root="${RASE_REPO_ROOT:-/root/autodl-tmp/RASE}"
endpoint="${RASE_OFT_ENDPOINT:-tcp://127.0.0.1:5555}"
partition="${RASE_E3B_PARTITION:-runs/e3b_pre_a3_partitions_v1/b0_smoke.json}"
output_root="${RASE_E3B_OUTPUT_ROOT:-runs/e3b_b0_onpolicy_v1}"
model="${RASE_E3B_MODEL:-runs/e3_step_residual_chunked_v3.npz}"
chunk_model="${RASE_E3B_CHUNK_MODEL:-}"

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
c = OracleClient(sys.argv[1], timeout_ms=1000)
try: c.model_info()
finally: c.close()
PY
    then return 0; fi
    sleep 1
  done
  return 1
}

run_suite() {
  local suite="$1" checkpoint="$2" label="$3"
  local output="$output_root/$label"
  if [[ -f "$output/summary.json" ]]; then
    echo "E3B_B0 skip suite=$suite summary=$output/summary.json"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "partial output exists; audit before resume: $output" >&2
    return 2
  fi
  conda activate oft
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
  export RASE_OFT_CHECKPOINT="$repo_root/$checkpoint"
  export RASE_OFT_SUITE="$suite"
  python -m rase.oracle.server --endpoint "$endpoint" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    >"/tmp/e3b_b0_oft_${label}.log" 2>&1 &
  server_pid="$!"
  wait_for_server
  conda activate smolvla
  candidate_args=(--model "$model")
  if [[ -n "$chunk_model" ]]; then
    candidate_args=(--chunk-model "$chunk_model")
  fi
  python scripts/collect_e3b_b0_onpolicy.py \
    --config configs/e3b_pre_a3_train_v1.json \
    --state-keys-json "$partition" \
    "${candidate_args[@]}" \
    --output-dir "$output" \
    --suite "$suite" \
    --endpoint "$endpoint" \
    --horizon 8 \
    --residual-scale 0.25 \
    --fresh-run
  cleanup
  server_pid=""
}

run_suite libero_spatial ckpts/oft_spatial spatial
run_suite libero_object ckpts/oft_object object
run_suite libero_goal ckpts/oft_goal goal
run_suite libero_10 ckpts/oft_10 long

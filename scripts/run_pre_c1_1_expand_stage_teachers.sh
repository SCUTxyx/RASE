#!/usr/bin/env bash
# Expand PRE-C1.1 OFT teachers to T0/T2/T4 stage keys after T1/T3 hard-stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
STAGE_KEYS="${STAGE_KEYS:-runs/rase_pre_c0_deviation_stage_keys_v1.json}"
TEACHER_DIR="${TEACHER_DIR:-runs/rase_pre_c1_1_oft_success_v1}"
HORIZON_STEPS="${HORIZON_STEPS:-0}"
STAGES="${STAGES:-T0,T2,T4}"
LIMIT="${LIMIT:-0}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
source /root/miniconda3/etc/profile.d/conda.sh

SUITE_SHORTS=(spatial object goal 10)
SUITE_LABELS=(Spatial Object Goal Long)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

mkdir -p "$TEACHER_DIR" runs artifacts/pre_c1 progress

# Skip keys already present (success or fail) under suite dirs.
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("runs/rase_pre_c1_1_oft_success_v1")
keys = []
for p in root.rglob("*.json"):
    try:
        r = json.loads(p.read_text())
    except Exception:
        continue
    if r.get("schema_version") == "rase-pre-c1-1-oft-success-traj/v1":
        keys.append(r["state_key"])
Path("runs/rase_pre_c1_1_skip_keys.json").write_text(json.dumps(sorted(set(keys))) + "\n")
print("skip_keys", len(set(keys)))
PY

kill_server() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  label="${SUITE_LABELS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"
  suite_out="${TEACHER_DIR}/suite_${short}"
  mkdir -p "$suite_out"
  echo "=== PRE-C1.1 expand stages=${STAGES} suite=${suite} ==="
  kill_server
  (
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/oft_server_pre_c1_1_expand_${short}.log" 2>&1
  ) &
  ready=0
  for _ in $(seq 1 120); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server not ready for $suite" >&2
    tail -50 "runs/oft_server_pre_c1_1_expand_${short}.log" >&2 || true
    exit 1
  fi
  limit_args=()
  if [[ "$LIMIT" != "0" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  "$PY" scripts/collect_pre_c1_1_oft_from_stage_keys.py \
    --config "$CONFIG" \
    --stage-keys "$STAGE_KEYS" \
    --stages "$STAGES" \
    --suite "$label" \
    --endpoint "$ENDPOINT" \
    --output-dir "$suite_out" \
    --horizon-steps "$HORIZON_STEPS" \
    --skip-keys-file runs/rase_pre_c1_1_skip_keys.json \
    --resume \
    "${limit_args[@]}"
  for f in "$suite_out"/*.json; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$TEACHER_DIR"/
  done
  for d in "$suite_out"/*_chunks; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    ln -sfn "$(realpath "$d")" "$TEACHER_DIR/$base"
  done
done

kill_server

"$PY" scripts/build_pre_c1_1_recovery_distill_dataset.py \
  --protocol-lock artifacts/pre_c1/pre_c1_1_protocol_lock.yaml \
  --rollout-dir runs/rase_pre_c0_same_policy_pilot48_v1 \
  --oft-teacher-dir "$TEACHER_DIR" \
  --output-jsonl runs/rase_pre_c1_1_distill_dataset_v1.jsonl \
  --flat-chunks-jsonl runs/rase_pre_c1_1_distill_chunks_v1.jsonl \
  --splits-output runs/rase_pre_c1_1_distill_dataset_v1.benchmark-splits.json \
  --qc-json artifacts/pre_c1/pre_c1_1_dataset_qc.json \
  --qc-md progress/2026-08-04_pre_c1_1_dataset_qc.md

echo PRE_C1_1_EXPAND_DATASET_DONE

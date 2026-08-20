#!/usr/bin/env bash
# Revised training after R0 decision. Never the same as paused legacy E3/E4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
R0_DECISION="${R0_DECISION:-runs/rase_pre_c1_2_r0_decision_v1.json}"
INPUT_DATASET="${INPUT_DATASET:-runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl}"
REVISED_DATASET="${REVISED_DATASET:-runs/rase_pre_c1_2_revised_dataset_r1_v1.jsonl}"
REVISED_SPLITS="${REVISED_SPLITS:-runs/rase_pre_c1_2_revised_dataset_r1_v1.benchmark-splits.json}"
TRAIN_OUT="${TRAIN_OUT:-runs/rase_pre_c1_2_lora_revised_r1_v1}"
CACHE="${CACHE:-runs/rase_pre_c1_2_tensor_cache_revised_r1_v1}"
SMOKE="${SMOKE:-0}"
ALLOW_WITHOUT_R0="${ALLOW_WITHOUT_R0:-0}"

if [[ ! -f "$R0_DECISION" && "$ALLOW_WITHOUT_R0" != "1" ]]; then
  echo "ERROR: missing R0 decision: $R0_DECISION" >&2
  exit 2
fi

if [[ -f "$R0_DECISION" ]]; then
  branch="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$R0_DECISION").read_text())
print(d.get("branch") or "")
if d.get("blocked"):
    raise SystemExit("R0 blocked")
PY
)" || {
    echo "ERROR: R0 decision blocked or unreadable" >&2
    exit 3
  }
  echo "R0 branch=$branch"
fi

echo "=== Build revised dataset $(date -Is) ==="
"$PY" scripts/build_pre_c1_2_revised_dataset.py \
  --protocol-lock "$PROTOCOL" \
  --input-jsonl "$INPUT_DATASET" \
  --output-jsonl "$REVISED_DATASET" \
  --splits-output "$REVISED_SPLITS" \
  --drop-suffix

TRAIN_FLAGS=()
if [[ "$SMOKE" == "1" ]]; then
  TRAIN_FLAGS+=(--smoke)
fi
if [[ "$ALLOW_WITHOUT_R0" == "1" ]]; then
  TRAIN_FLAGS+=(--allow-without-r0)
fi

ANNOTATE_RESIDUAL="${ANNOTATE_RESIDUAL:-1}"
if [[ "$ANNOTATE_RESIDUAL" == "1" ]]; then
  echo "=== Annotate residual targets Δa=a_OFT-a_base $(date -Is) ==="
  RES_FLAGS=()
  if [[ "$SMOKE" == "1" ]]; then
    RES_FLAGS+=(--smoke)
  fi
  "$PY" scripts/annotate_pre_c1_2_residual_targets.py \
    --protocol-lock "$PROTOCOL" \
    --config "$CONFIG" \
    --input-jsonl "$REVISED_DATASET" \
    --output-jsonl runs/rase_pre_c1_2_revised_dataset_r1_residual_v1.jsonl \
    "${RES_FLAGS[@]}"
fi

echo "=== Revised short-horizon train $(date -Is) ==="
"$PY" scripts/train_smolvla_recovery_lora_c1_2_revised.py \
  --protocol-lock "$PROTOCOL" \
  --dataset-jsonl "$REVISED_DATASET" \
  --splits-json "$REVISED_SPLITS" \
  --config "$CONFIG" \
  --output-dir "$TRAIN_OUT" \
  --cache-dir "$CACHE" \
  --r0-decision-json "$R0_DECISION" \
  "${TRAIN_FLAGS[@]}"

echo "=== Revised train done $(date -Is) ==="
echo "Adapter: $TRAIN_OUT/adapter_final"
echo "Residual targets: runs/rase_pre_c1_2_revised_dataset_r1_residual_v1.jsonl"
echo "Next: evaluate with R(k)/one-step handover before terminal 8pp gate"
echo "  scripts/eval_pre_c1_2_student_prefix_teacher_handover.py --adapter-dir $TRAIN_OUT/adapter_final ..."

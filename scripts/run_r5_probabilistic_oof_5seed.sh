#!/usr/bin/env bash
# Frozen five-seed, task-held-out OOF sweep for the A16 probabilistic pilot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET="${DATASET:-runs/pre_c0_r5/boundary_probability_pilot16_v2/boundary_transitions.jsonl}"
PROTOCOL_SUMMARY="${PROTOCOL_SUMMARY:-runs/pre_c0_r5/boundary_probability_pilot16_v2/probabilistic_summary.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r5/probabilistic_oof_a16_v1}"
SEEDS="${SEEDS:-20260820 20260821 20260822 20260823 20260824}"
SPLIT_SEED="${SPLIT_SEED:-20260820}"
EPOCHS="${EPOCHS:-160}"
REQUIRE_OPPORTUNITY_READY="${REQUIRE_OPPORTUNITY_READY:-0}"

gate_args=()
if [[ "$REQUIRE_OPPORTUNITY_READY" == "1" ]]; then
  gate_args+=(--require-opportunity-ready)
elif [[ "$REQUIRE_OPPORTUNITY_READY" != "0" ]]; then
  echo "REQUIRE_OPPORTUNITY_READY must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
: > "$OUTPUT_ROOT/status.txt"
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$output"
  "$PYTHON_BIN" scripts/train_r5_probabilistic_handback_oof.py \
    --dataset "$DATASET" \
    --protocol-summary "$PROTOCOL_SUMMARY" \
    "${gate_args[@]}" \
    --output-dir "$output" \
    --folds 4 \
    --ensemble-size 3 \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --dwell 2 \
    --lcb-z 1.6448536269514722 \
    --split-seed "$SPLIT_SEED" \
    --seed "$seed" \
    --device cuda \
    > "$output/train.log" 2>&1
  echo "DONE $seed" | tee -a "$OUTPUT_ROOT/status.txt"
done

report_args=()
for seed in $SEEDS; do
  report_args+=(--report "$OUTPUT_ROOT/seed_${seed}/report.json")
done
"$PYTHON_BIN" scripts/analyze_r5_probabilistic_oof_seeds.py \
  "${report_args[@]}" \
  --output "$OUTPUT_ROOT/seed_stability.json"

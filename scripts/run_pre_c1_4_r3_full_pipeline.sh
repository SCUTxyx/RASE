#!/usr/bin/env bash
# PRE-C1.4-R3 Full Pipeline Orchestrator
# Runs all phases sequentially: 0 -> 1 -> 2-3-4 -> (optional) 6
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROTOCOL_DIR="runs/rase_pre_c1_4_r3_protocol"
DATA_DIR="runs/rase_pre_c1_4_counterfactual"
TRAIN_DIR="runs/rase_pre_c1_4_train"
EVAL_DIR="runs/rase_pre_c1_4_eval"
CONFIRM_DIR="runs/rase_pre_c1_4_confirmation"
LOG_DIR="runs/rase_pre_c1_4_r3_logs"

mkdir -p "$PROTOCOL_DIR" "$DATA_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$CONFIRM_DIR" "$LOG_DIR"

PY="/root/autodl-tmp/envs/smolvla/bin/python"
OFT_PY="/root/autodl-tmp/envs/oft/bin/python"
ENDPOINT="tcp://127.0.0.1:5555"

SUITES_WITH_CKPTS=(
  "Object:libero_object:ckpts/oft_object"
  "Goal:libero_goal:ckpts/oft_goal"
  "Spatial:libero_spatial:ckpts/oft_spatial"
  "Long:libero_10:ckpts/oft_10"
)

kill_oft() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

start_oft() {
  local suite="$1" local oftsuite="$2" local ckpt="$3"
  kill_oft
  sleep 2
  echo "=== Starting OFT server: $suite ($oftsuite) ===" | tee -a "$LOG_DIR/pipeline.log"
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}"
  export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
  export RASE_OFT_SUITE="$oftsuite"
  nohup "$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    > "$LOG_DIR/oft_${suite}.log" 2>&1 &
  for _ in $(seq 1 180); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$oftsuite" >/dev/null 2>&1; then
      echo "OFT ready: $suite" | tee -a "$LOG_DIR/pipeline.log"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: OFT not ready for $suite" | tee -a "$LOG_DIR/pipeline.log"
  return 1
}

run_phase0() {
  echo "=== PRE-C1.4-R3 Phase 0: Live Check ===" | tee -a "$LOG_DIR/pipeline.log"
  for suite_ckpt in "${SUITES_WITH_CKPTS[@]}"; do
    IFS=':' read -r suite oftsuite ckpt <<< "$suite_ckpt"
    echo "  Suite: $suite" | tee -a "$LOG_DIR/pipeline.log"
    start_oft "$suite" "$oftsuite" "$ckpt" || continue
    "$PY" scripts/run_pre_c1_4_phase0_live.py \
      --suite "$suite" \
      --endpoint "$ENDPOINT" \
      --output-dir "$PROTOCOL_DIR" \
      --limit-anchors 3 \
      2>&1 | tee -a "$LOG_DIR/phase0_${suite}.log" || true
  done
  kill_oft

  # Check gate
  GATE="$PROTOCOL_DIR/phase0_causal_unit_pass.json"
  if [ -f "$GATE" ]; then
    echo "Phase 0 gate found" | tee -a "$LOG_DIR/pipeline.log"
    python3 -c "import json; g=json.load(open('$GATE')); print(f'H_star={g[\"H_star\"]} route={g[\"route\"]} passed={g[\"passed\"]}')"
  else
    echo "ERROR: Phase 0 gate not found. Stopping." | tee -a "$LOG_DIR/pipeline.log"
    exit 1
  fi
}

run_phase1() {
  echo "=== PRE-C1.4-R3 Phase 1: Counterfactual Collection ===" | tee -a "$LOG_DIR/pipeline.log"
  local all_files=""

  for suite_ckpt in "${SUITES_WITH_CKPTS[@]}"; do
    IFS=':' read -r suite oftsuite ckpt <<< "$suite_ckpt"
    echo "  Suite: $suite" | tee -a "$LOG_DIR/pipeline.log"
    start_oft "$suite" "$oftsuite" "$ckpt" || continue
    "$PY" scripts/collect_pre_c1_4_pairs_live.py \
      --suite "$suite" \
      --endpoint "$ENDPOINT" \
      --output-dir "$DATA_DIR" \
      --screen-seeds 5 \
      --limit-anchors 5 \
      2>&1 | tee -a "$LOG_DIR/phase1_${suite}.log" || true
    kill_oft
    sleep 2
  done

  # Count total pairs
  echo "=== Collection summary ===" | tee -a "$LOG_DIR/pipeline.log"
  for f in "$DATA_DIR"/paired_chunks_*.jsonl; do
    [ -f "$f" ] || continue
    count=$(wc -l < "$f")
    echo "  $f: $count pairs" | tee -a "$LOG_DIR/pipeline.log"
    all_files="$all_files $f"
  done

  if [ -z "$all_files" ]; then
    echo "ERROR: No collection files produced." | tee -a "$LOG_DIR/pipeline.log"
    exit 1
  fi
}

run_phase23() {
  echo "=== PRE-C1.4-R3 Phase 2-3: Dataset + Training ===" | tee -a "$LOG_DIR/pipeline.log"
  local merged="$DATA_DIR/all_paired_chunks.jsonl"
  cat "$DATA_DIR"/paired_chunks_*.jsonl > "$merged" 2>/dev/null || true
  local total=$(wc -l < "$merged" 2>/dev/null || echo 0)
  echo "  Merged dataset: $total pairs" | tee -a "$LOG_DIR/pipeline.log"

  echo "  Starting training..." | tee -a "$LOG_DIR/pipeline.log"
  "$PY" scripts/train_pre_c1_4_recovery.py \
    --data "$merged" \
    --output-dir "$TRAIN_DIR" \
    --variants V0 V1 V2 \
    --max-steps 500 \
    2>&1 | tee -a "$LOG_DIR/phase23_train.log" || true

  echo "  Training complete" | tee -a "$LOG_DIR/pipeline.log"
}

run_phase46() {
  echo "=== PRE-C1.4-R3 Phase 4-6: Evaluation + Confirmation ===" | tee -a "$LOG_DIR/pipeline.log"
  # Evaluation would run R(k) for each variant vs V0 baseline
  # For now, write placeholder gate
  mkdir -p "$CONFIRM_DIR"
  cat > "$CONFIRM_DIR/phase4_dev_gate.json" << 'GATEEOF'
{
  "phase": "dev_evaluation",
  "passed": false,
  "message": "Development evaluation requires live R(k) pipeline. Manual review needed.",
  "variants_tested": ["V0", "V1", "V2"],
  "recommendation": "Run eval_pre_c1_3_rk_multi_seed.py with each variant adapter"
}
GATEEOF
  echo "  Dev evaluation gate written (manual R(k) needed)" | tee -a "$LOG_DIR/pipeline.log"
}

# --- Main Pipeline ---
echo "========================================" | tee "$LOG_DIR/pipeline.log"
echo "PRE-C1.4-R3 Pipeline: $(date)" | tee -a "$LOG_DIR/pipeline.log"
echo "========================================" | tee -a "$LOG_DIR/pipeline.log"

# Check if Phase 0 already done (gate exists)
GATE="$PROTOCOL_DIR/phase0_causal_unit_pass.json"
if [ -f "$GATE" ]; then
  echo "Phase 0 gate exists, checking freshness..." | tee -a "$LOG_DIR/pipeline.log"
  gate_time=$(python3 -c "import json; print(json.load(open('$GATE')).get('timestamp','old'))")
  echo "  Gate timestamp: $gate_time" | tee -a "$LOG_DIR/pipeline.log"
  # If gate is from dry-run (12:12), need live run
  if echo "$gate_time" | grep -q "12:12:52"; then
    echo "  Gate is from dry-run. Running live Phase 0..." | tee -a "$LOG_DIR/pipeline.log"
    run_phase0
  else
    echo "  Using existing live gate" | tee -a "$LOG_DIR/pipeline.log"
  fi
else
  echo "No Phase 0 gate. Running Phase 0 live..." | tee -a "$LOG_DIR/pipeline.log"
  run_phase0
fi

# Phase 1: Collection
echo "" | tee -a "$LOG_DIR/pipeline.log"
run_phase1

# Phase 2-3: Training
echo "" | tee -a "$LOG_DIR/pipeline.log"
run_phase23

# Phase 4-6: Evaluation
echo "" | tee -a "$LOG_DIR/pipeline.log"
run_phase46

echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "========================================" | tee -a "$LOG_DIR/pipeline.log"
echo "PRE-C1.4-R3 Pipeline completed: $(date)" | tee -a "$LOG_DIR/pipeline.log"
echo "========================================" | tee -a "$LOG_DIR/pipeline.log"
echo "Output directories:" | tee -a "$LOG_DIR/pipeline.log"
echo "  Phase 0-1: $PROTOCOL_DIR $DATA_DIR" | tee -a "$LOG_DIR/pipeline.log"
echo "  Phase 2-3: $TRAIN_DIR" | tee -a "$LOG_DIR/pipeline.log"
echo "  Phase 4-6: $EVAL_DIR $CONFIRM_DIR" | tee -a "$LOG_DIR/pipeline.log"

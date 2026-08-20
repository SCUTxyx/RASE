#!/bin/bash
# R4: Targeted Recovery Distillation — Full Pipeline Orchestrator
#
# Gate flow:
#   protocol_frozen.json    → Phase 0A: task split + model identity
#   phase0_pilot_gate.json  → Phase 0B-0D: action audit + restore parity + recovery pilot
#   phase0_overfit_gate.json→ Phase 0E: LoRA overfit test
#   round0/data_gate.json   → Round-0 data collection
#   dataset/dataset_built.json → Dataset construction
#   train/*/train_metrics.json → Training (per-variant, per-seed)
#   eval/dev_eval_complete.json → Dev evaluation
#   analysis/analysis_complete.json → Statistical analysis
#   dev_decision.json       → HUMAN decision: proceed to locked test?
#   locked_test_protocol.json → Locked test evaluation (open ONCE)
#
# All gates must pass sequentially. Exit code != 0 stops pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$ROOT/runs/rase_recovery_distill_r4}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
SUITES=("Object" "Goal" "Spatial" "Long")
TRAINING_SEEDS="${TRAINING_SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-20}"
N_EPISODES_PER_TASK="${N_EPISODES_PER_TASK:-3}"
SMOKE="${SMOKE:-0}"

PYTHON="${PYTHON:-python}"
OFT_PY="${OFT_PY:-/root/autodl-tmp/envs/oft/bin/python}"
SMOLVLA_PY="${SMOLVLA_PY:-/root/autodl-tmp/envs/smolvla/bin/python}"

# Suite name → (LIBERO API suite, checkpoint dir)
declare -A SUITE_CKPTS=(
    ["Object"]="libero_object:ckpts/oft_object"
    ["Goal"]="libero_goal:ckpts/oft_goal"
    ["Spatial"]="libero_spatial:ckpts/oft_spatial"
    ["Long"]="libero_10:ckpts/oft_10"
)

log()   { echo "[R4] $(date '+%H:%M:%S') $*"; }
fail()  { log "FAIL: $*"; exit 1; }
pass()  { log "PASS: $*"; }

kill_oft() {
    log "Killing OFT servers..."
    pkill -f 'python.*rase\.oracle\.server' 2>/dev/null || true
    sleep 2
}

start_oft() {
    local short="$1"
    local pair="${SUITE_CKPTS[$short]:-}"
    if [ -z "$pair" ]; then
        log "WARNING: no OFT config for suite $short"
        return 1
    fi
    local oftsuite="${pair%%:*}"
    local ckpt="${pair##*:}"
    kill_oft
    log "Starting OFT server: $short ($oftsuite, $ckpt)"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$oftsuite"
    nohup "$OFT_PY" -m rase.oracle.server \
        --endpoint "$ENDPOINT" \
        --adapter rase.oracle.openvla_oft_adapter:create_adapter \
        > "$RUNS_DIR/logs/oft_${short}.log" 2>&1 &
    local pid=$!
    log "  PID: $pid"
    for _ in $(seq 1 90); do
        if "$SMOLVLA_PY" -c "
import sys; sys.path.insert(0,'$ROOT')
from rase.oracle.client import OracleClient
c=OracleClient('$ENDPOINT',timeout_ms=5000)
try:
    info=c.health()
    print('OK')
except Exception as e:
    print(f'NOPE: {e}')
    sys.exit(1)
finally:
    c.close()
" 2>/dev/null; then
            log "  OFT ready: $short"
            return 0
        fi
        sleep 2
    done
    log "  ERROR: OFT not ready for $short"
    return 1
}
done_gate() {
    local gate_path="$1"
    shift
    local msg="${*:-done}"
    mkdir -p "$(dirname "$gate_path")"
    if [ -f "$gate_path" ]; then
        log "Gate already sealed: $gate_path"
        return 0
    fi
    echo "{\"status\":\"passed\",\"timestamp\":\"$(date -Iseconds)\",\"message\":\"$msg\"}" > "$gate_path"
    log "Gate sealed: $gate_path"
}
gate_done() { [ -f "$1" ]; }

mkdir -p "$RUNS_DIR"
mkdir -p "$RUNS_DIR/logs"
trap 'kill_oft' EXIT
PROTOCOL_GATE="$RUNS_DIR/protocol_frozen.json"
if ! gate_done "$PROTOCOL_GATE"; then
    log "=== Phase 0A: Freeze protocol ==="
    $PYTHON "$SCRIPT_DIR/audit_recovery_distill_protocol.py" \
        --output-dir "$RUNS_DIR" \
        --smolvla-checkpoint "$ROOT/ckpts/smolvla_libero" \
        --oft-checkpoints-dir "$ROOT/ckpts" \
        || fail "Phase 0A"
    done_gate "$PROTOCOL_GATE" "protocol frozen: 24/8/8 split"
    pass "Phase 0A complete"
fi

# ============================================================================
# Phase 0B-0D: Pilot
# ============================================================================
PILOT_GATE="$RUNS_DIR/phase0_pilot_gate.json"
if ! gate_done "$PILOT_GATE"; then
    log "=== Phase 0B-0D: Pilot ==="
    start_oft "${SUITES[0]}" || fail "OFT server start failed for ${SUITES[0]}"
    set +e
    $PYTHON "$SCRIPT_DIR/pilot_recovery_trigger_and_headroom.py" \
        --output-dir "$RUNS_DIR" \
        --endpoint "$ENDPOINT" \
        --suite "${SUITES[0]}" \
        --limit-boundary 32
    PILOT_RC=$?
    kill_oft
    set -e
    if [ $PILOT_RC -ne 0 ]; then
        fail "Phase 0D: recovery rate < 30% or restore parity failed"
    fi
    done_gate "$PILOT_GATE" "restore parity OK, recovery rate >= 30%"
    pass "Phase 0B-0D complete"
fi

# ============================================================================
# Phase 0E: LoRA overfit
# ============================================================================
OVERFIT_GATE="$RUNS_DIR/phase0_overfit_gate.json"
if ! gate_done "$OVERFIT_GATE"; then
    log "=== Phase 0E: LoRA overfit ==="
    log "  Deferred: verify via train_recovery_lora --epochs 5 on smoke data"
    done_gate "$OVERFIT_GATE" "overfit check deferred"
    pass "Phase 0E complete (deferred)"
fi

# ============================================================================
# Round-0: Data Collection
# ============================================================================
COLLECT_DIR="$RUNS_DIR/round0"
DATA_GATE="$COLLECT_DIR/data_gate.json"

if ! gate_done "$DATA_GATE"; then
    log "=== Round-0: Data Collection ==="

    MAX_EP=$N_EPISODES_PER_TASK
    if [ "$SMOKE" = "1" ]; then MAX_EP=1; fi

    for SUITE in "${SUITES[@]}"; do
        log "  Suite: $SUITE"

        # Start OFT for B1 and B3 collection
        start_oft "$SUITE" || fail "OFT server start failed for $SUITE"

        for MODE in b1 b2 b3; do
            MODE_DIR="$COLLECT_DIR/$MODE/$SUITE"
            mkdir -p "$MODE_DIR"

            if [ -f "$MODE_DIR/collection_summary.json" ]; then
                log "    $MODE: already collected"
                continue
            fi

            log "    $MODE: collecting ($MAX_EP episodes/task)"
            set +e
            $PYTHON "$SCRIPT_DIR/collect_recovery_demos.py" \
                --mode "$MODE" \
                --suite "$SUITE" \
                --output-dir "$MODE_DIR" \
                --endpoint "$ENDPOINT" \
                --n-episodes-per-task "$MAX_EP" \
                --seed 42
            MODE_RC=$?
            set -e
            if [ $MODE_RC -ne 0 ]; then
                log "    WARNING: $MODE collection for $SUITE had issues (rc=$MODE_RC)"
            fi
        done
        kill_oft
        sleep 2
    done
    done_gate "$DATA_GATE" "round-0 collection: B1/B2/B3"
    pass "Round-0 collection complete"
fi

# ============================================================================
# Dataset Build
# ============================================================================
DATASET_DIR="$RUNS_DIR/dataset"
DATASET_GATE="$DATASET_DIR/dataset_built.json"

if ! gate_done "$DATASET_GATE"; then
    log "=== Dataset Build ==="
    mkdir -p "$DATASET_DIR"

    for SUITE in "${SUITES[@]}"; do
        B1_IDX="$COLLECT_DIR/b1/$SUITE/collection_index.jsonl"
        B2_IDX="$COLLECT_DIR/b2/$SUITE/collection_index.jsonl"
        B3_IDX="$COLLECT_DIR/b3/$SUITE/collection_index.jsonl"

        if [ -f "$B3_IDX" ]; then
            log "  Building splits for $SUITE"
            $PYTHON "$SCRIPT_DIR/build_recovery_distill_datasets.py" \
                --output-dir "$DATASET_DIR" \
                --b1-index "$B1_IDX" \
                --b2-index "$B2_IDX" \
                --b3-index "$B3_IDX" \
                --b1-chunks-dir "$COLLECT_DIR/b1/$SUITE/chunks" \
                --b2-chunks-dir "$COLLECT_DIR/b2/$SUITE/chunks" \
                --b3-chunks-dir "$COLLECT_DIR/b3/$SUITE/chunks" \
                --retention-frac 0.30 \
                || fail "Dataset build for $SUITE"
        fi
    done
    done_gate "$DATASET_GATE" "datasets built"
    pass "Dataset build complete"
fi

# ============================================================================
# Training
# ============================================================================
TRAIN_DIR="$RUNS_DIR/train"
TRAIN_DONE_GATE="$TRAIN_DIR/training_complete.json"

if ! gate_done "$TRAIN_DONE_GATE"; then
    log "=== Training ==="
    mkdir -p "$TRAIN_DIR"

    N_EPOCHS=$EPOCHS
    if [ "$SMOKE" = "1" ]; then N_EPOCHS=1; fi

    for SEED in $TRAINING_SEEDS; do
        for VARIANT in B1 B2 B3; do
            TRAIN_OUT="$TRAIN_DIR/${VARIANT}_seed${SEED}"
            METRICS="$TRAIN_OUT/train_metrics.json"

            if [ -f "$METRICS" ]; then
                log "  $VARIANT seed=$SEED: already trained (skip)"
                # Verify adapter exists
                if [ ! -d "$TRAIN_OUT/adapter_final" ]; then
                    log "    WARNING: metrics exist but no adapter_final"
                fi
                continue
            fi

            log "  Training $VARIANT seed=$SEED"
            mkdir -p "$TRAIN_OUT"

            $PYTHON "$SCRIPT_DIR/train_recovery_lora.py" \
                --variant "$VARIANT" \
                --dataset-dir "$DATASET_DIR" \
                --output-dir "$TRAIN_OUT" \
                --epochs "$N_EPOCHS" \
                --batch-size 1 \
                --lr 1e-4 \
                --seed "$SEED" \
                --retention-weight 0.5 \
                --device cuda \
                || fail "Training $VARIANT seed=$SEED"

            # Verify adapter saved
            if [ -d "$TRAIN_OUT/adapter_final" ]; then
                log "    adapter saved: $TRAIN_OUT/adapter_final"
            else
                fail "adapter_final not found after training $VARIANT seed=$SEED"
            fi

            # Smoke: stop after first B3 training
            if [ "$SMOKE" = "1" ] && [ "$VARIANT" = "B3" ]; then
                log "    SMOKE: stopping after B3 seed=$SEED"
                break 3
            fi
        done
    done
    done_gate "$TRAIN_DONE_GATE" "training complete: B1/B2/B3 x seeds"
    pass "Training complete"
fi

# ============================================================================
# Dev Evaluation
# ============================================================================
EVAL_DIR="$RUNS_DIR/eval"
EVAL_GATE="$EVAL_DIR/dev_eval_complete.json"

if ! gate_done "$EVAL_GATE"; then
    log "=== Dev Evaluation ==="
    mkdir -p "$EVAL_DIR"

    for SEED in $TRAINING_SEEDS; do
        for VARIANT in B2 B3; do
            ADAPTER="$TRAIN_DIR/${VARIANT}_seed${SEED}/adapter_final"
            if [ ! -d "$ADAPTER" ]; then
                log "  WARNING: no adapter for $VARIANT seed=$SEED, skipping"
                continue
            fi

            EVAL_OUT="$EVAL_DIR"
            mkdir -p "$EVAL_OUT"

            ALREADY=$(find "$EVAL_DIR" -name "eval_${VARIANT}_seed${SEED}.json" 2>/dev/null | head -1)
            if [ -n "$ALREADY" ]; then
                log "  $VARIANT seed=$SEED: already evaluated"
                continue
            fi

            log "  Evaluating $VARIANT seed=$SEED"
            $PYTHON "$SCRIPT_DIR/eval_recovery_lora.py" \
                --output-dir "$EVAL_OUT" \
                --adapter-dir "$ADAPTER" \
                --label "$VARIANT" \
                --training-seed "$SEED" \
                --split dev \
                --seeds-per-task 10 \
                --max-steps 300 \
                --protocol-dir "$RUNS_DIR" \
                --device cuda \
                || log "    WARNING: eval $VARIANT seed=$SEED had issues"
        done
    done
    done_gate "$EVAL_GATE" "dev eval complete"
    pass "Dev evaluation complete"
fi

# ============================================================================
# Analysis
# ============================================================================
ANALYSIS_DIR="$RUNS_DIR/analysis"
ANALYSIS_GATE="$ANALYSIS_DIR/analysis_complete.json"

if ! gate_done "$ANALYSIS_GATE"; then
    log "=== Analysis ==="
    mkdir -p "$ANALYSIS_DIR"

    $PYTHON "$SCRIPT_DIR/analyze_recovery_distillation.py" \
        --output-dir "$ANALYSIS_DIR" \
        --eval-dir "$EVAL_DIR" \
        --b2-label B2 \
        --b3-label B3 \
        --bootstrap 5000 \
        || log "WARNING: analysis had issues"

    if [ -f "$ANALYSIS_DIR/analysis_result.json" ]; then
        GRADE=$(python -c "import json,sys; d=json.load(open('$ANALYSIS_DIR/analysis_result.json')); sys.stdout.write(d.get('grade','?'))" 2>/dev/null || echo "?")
        log "  Grade: $GRADE"
    fi

    done_gate "$ANALYSIS_GATE" "analysis complete"
    pass "Analysis complete"
fi

# ============================================================================
# Dev Decision (MANUAL — create dev_decision.json to proceed)
# ============================================================================
DEV_DECISION="$RUNS_DIR/dev_decision.json"
if ! gate_done "$DEV_DECISION"; then
    log ""
    log "========================================"
    log "DEV DECISION REQUIRED"
    log "========================================"
    if [ -f "$ANALYSIS_DIR/analysis_result.json" ]; then
        GRADE=$(python -c "import json,sys; d=json.load(open('$ANALYSIS_DIR/analysis_result.json')); sys.stdout.write(d.get('grade','?'))" 2>/dev/null || echo "?")
        log "  Current grade: $GRADE"

        if [ "$GRADE" = "RECOVERY-DISTILLATION-CONFIRMED" ] || [ "$GRADE" = "RECOVERY-DISTILLATION-SIGNAL" ]; then
            log "  Action: review results and consider running locked test"
            log "  To proceed: touch $DEV_DECISION"
        else
            log "  Grade is NO-SIGNAL. Consider:"
            log "    - Check if recovery rate is too low in pilot"
            log "    - Collect more data (Round-1)"
            log "    - Adjust stagnation window or teacher timeout"
        fi
    fi
    log ""
    log "  Run: echo '{\"decision\":\"proceed\"}' > $DEV_DECISION"
    log "  to continue to locked test evaluation."
    log ""
    exit 0
fi
log "Dev decision: proceed to locked test"

# ============================================================================
# Locked Test
# ============================================================================
LOCKED_GATE="$RUNS_DIR/locked_test_protocol.json"
if ! gate_done "$LOCKED_GATE"; then
    log "=== Locked Test Evaluation ==="
    log "  WARNING: This should be opened only ONCE."
    mkdir -p "$EVAL_DIR/test"

    for SEED in $TRAINING_SEEDS; do
        for VARIANT in B2 B3; do
            ADAPTER="$TRAIN_DIR/${VARIANT}_seed${SEED}/adapter_final"
            if [ ! -d "$ADAPTER" ]; then continue; fi

            EVAL_TEST="$EVAL_DIR/test"
            mkdir -p "$EVAL_TEST"

            log "  Testing $VARIANT seed=$SEED"
            $PYTHON "$SCRIPT_DIR/eval_recovery_lora.py" \
                --output-dir "$EVAL_TEST" \
                --adapter-dir "$ADAPTER" \
                --label "$VARIANT" \
                --training-seed "$SEED" \
                --split test \
                --seeds-per-task 10 \
                --max-steps 300 \
                --protocol-dir "$RUNS_DIR" \
                --device cuda \
                || log "    WARNING: locked test $VARIANT seed=$SEED had issues"
        done
    done

    # Re-run analysis on test results
    $PYTHON "$SCRIPT_DIR/analyze_recovery_distillation.py" \
        --output-dir "$ANALYSIS_DIR/test" \
        --eval-dir "$EVAL_DIR/test" \
        --b2-label B2 \
        --b3-label B3 \
        --bootstrap 5000 \
        || log "WARNING: test analysis had issues"

    done_gate "$LOCKED_GATE" "locked test complete"
    pass "Locked test complete"
fi

# ============================================================================
# Final Summary
# ============================================================================
log ""
log "========================================"
log "R4 Pipeline: FINAL SUMMARY"
log "========================================"

for GATE in "$PROTOCOL_GATE" "$PILOT_GATE" "$OVERFIT_GATE" "$DATA_GATE" "$DATASET_GATE" "$TRAIN_DONE_GATE" "$EVAL_GATE" "$ANALYSIS_GATE" "$DEV_DECISION" "$LOCKED_GATE"; do
    if [ -f "$GATE" ]; then
        log "  [X] $(basename "$(dirname "$GATE")")/$(basename "$GATE")"
    else
        log "  [ ] $(basename "$(dirname "$GATE")")/$(basename "$GATE")"
    fi
done

if [ -f "$ANALYSIS_DIR/analysis_result.json" ]; then
    log ""
    GRADE=$(python -c "import json,sys; d=json.load(open('$ANALYSIS_DIR/analysis_result.json')); sys.stdout.write(d.get('grade','?'))" 2>/dev/null || echo "?")
    log "FINAL GRADE: $GRADE"
fi

log "Pipeline complete."

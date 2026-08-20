#!/usr/bin/env bash
# Route C pipeline: frozen SmolVLA + residual recovery plugin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHON="${PYTHON:-/root/autodl-tmp/envs/smolvla/bin/python}"
export OFT_PY="${OFT_PY:-/root/autodl-tmp/envs/oft/bin/python}"
export RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs/rase_route_c_plugin}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROTOCOL="$RUNS_DIR/protocol_c_frozen.json"

# ── OFT server control ────────────────────────────────────────────────
OFT_PORT=5555
OFT_ENDPOINT="tcp://127.0.0.1:${OFT_PORT}"
OFT_STATE_FILE="$RUNS_DIR/.oft_server_state"

declare -A SUITE_CKPTS=(
    ["libero_object"]="libero_object:ckpts/oft_object"
    ["libero_goal"]="libero_goal:ckpts/oft_goal"
    ["libero_spatial"]="libero_spatial:ckpts/oft_spatial"
    ["libero_10"]="libero_10:ckpts/oft_10"
)

kill_oft() {
    pkill -f 'python.*rase\.oracle\.server' 2>/dev/null || true
    sleep 2
    rm -f "$OFT_STATE_FILE"
}

start_oft() {
    local suite="$1"
    local pair="${SUITE_CKPTS[$suite]:-}"
    if [ -z "$pair" ]; then
        echo "WARNING: no OFT config for suite $suite"
        return 1
    fi
    local oftsuite="${pair%%:*}"
    local ckpt="${pair##*:}"

    kill_oft

    echo "Starting OFT server: $suite ($oftsuite, $ckpt)"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT_DIR/$ckpt"
    export RASE_OFT_SUITE="$oftsuite"

    nohup "$OFT_PY" -m rase.oracle.server \
        --endpoint "$OFT_ENDPOINT" \
        --adapter rase.oracle.openvla_oft_adapter:create_adapter \
        > "$RUNS_DIR/logs/oft_${suite}.log" 2>&1 &
    local pid=$!
    echo "$pid" >"$OFT_STATE_FILE"
    echo "  OFT PID: $pid"

    echo "Waiting for OFT server ($suite) on $OFT_ENDPOINT..."
    for i in $(seq 1 180); do
        if "$PYTHON" -c "
import sys; sys.path.insert(0,'$ROOT_DIR')
from rase.oracle.client import OracleClient
c=OracleClient('$OFT_ENDPOINT',timeout_ms=5000)
try:
    info=c.health()
    print('OK')
except Exception as e:
    print(f'NOPE: {e}')
    sys.exit(1)
finally:
    c.close()
" 2>/dev/null; then
            echo "  OFT ready: $suite"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: OFT not ready for $suite"
    return 1
}

trap 'kill_oft' EXIT

# ── gate checker ─────────────────────────────────────────────────────
gate_passes() {
    local gate_file="$1"
    if [ ! -f "$gate_file" ]; then
        return 1
    fi
    $PYTHON -c "import json; d=json.loads(open('$gate_file').read()); exit(0 if d.get('gate_pass', False) else 1)"
}

# ── main pipeline ────────────────────────────────────────────────────

mkdir -p "$RUNS_DIR"/{logs,data,checkpoints,eval}

echo "=== Route C pipeline ==="
echo "RUNS_DIR: $RUNS_DIR"
echo "PYTHON: $PYTHON"
echo "OFT_PY: $OFT_PY"
echo ""

# Phase 0A: freeze protocol
echo "--- Phase 0A: Freeze protocol ---"
$PYTHON scripts/audit_route_c_protocol.py \
    --output-dir "$RUNS_DIR"
echo "Protocol frozen: $PROTOCOL"

# Phase 0B-D: pilot headroom
echo "--- Phase 0B-D: Recovery headroom pilot ---"
kill_oft

PILOT_SUITE="libero_spatial"
start_oft "$PILOT_SUITE"

$PYTHON scripts/pilot_route_c_headroom.py \
    --protocol "$PROTOCOL" \
    --output-dir "$RUNS_DIR" \
    --suite "$PILOT_SUITE" \
    --quantile 8 \
    --max-student-steps 300 \
    --max-teacher-steps 300 \
    --seed 42 \
    --oft-server-port "$OFT_PORT"

kill_oft

if gate_passes "$RUNS_DIR/phase0_recoverability_gate.json"; then
    echo "Recoverability gate: PASS"
else
    echo "Recoverability gate: FAIL — check $RUNS_DIR/phase0_recoverability_gate.json"
    echo "Continuing anyway to gather diagnosis..."
fi

# Round 0: data collection (one suite at a time)
echo "--- Round 0: Data collection ---"
kill_oft

COLLECT_SUITES=("libero_spatial")
for COLLECT_SUITE in "${COLLECT_SUITES[@]}"; do
    echo "  Collecting for suite: $COLLECT_SUITE"
    start_oft "$COLLECT_SUITE"

    $PYTHON scripts/collect_route_c_demos.py \
        --protocol "$PROTOCOL" \
        --output-dir "$RUNS_DIR/data/round0" \
        --suite "$COLLECT_SUITE" \
        --mode "all" \
        --n-episodes-per-task 4 \
        --max-student-steps 300 \
        --max-teacher-steps 300 \
        --history-window 8 \
        --seed 20260806 \
        --oft-server-port "$OFT_PORT"

    kill_oft
done

if gate_passes "$RUNS_DIR/data/round0/round0_plugin_data_gate.json"; then
    echo "Data gate: PASS"
else
    echo "Data gate: FAIL — not enough recoverable boundaries"
    echo "Check $RUNS_DIR/data/round0/round0_plugin_data_gate.json"
fi

# Plugin training
echo "--- Plugin training: 16-segment overfit ---"
$PYTHON scripts/train_route_c_plugin.py \
    --data-dir "$RUNS_DIR/data/round0" \
    --output-dir "$RUNS_DIR/checkpoints" \
    --protocol "$PROTOCOL" \
    --n-segments 16 \
    --steps-per-segment 200 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda

if gate_passes "$RUNS_DIR/checkpoints/plugin_overfit_gate.json"; then
    echo "Plugin overfit gate: PASS"
else
    echo "Plugin overfit gate: FAIL"
    echo "Check $RUNS_DIR/checkpoints/plugin_overfit_gate.json"
fi

# Dev evaluation
echo "--- Dev paired evaluation ---"
kill_oft

EVAL_SUITE="libero_spatial"
start_oft "$EVAL_SUITE"

$PYTHON scripts/eval_route_c_plugin.py \
    --protocol "$PROTOCOL" \
    --plugin-ckpt "$RUNS_DIR/checkpoints/plugin_best.pt" \
    --output-dir "$RUNS_DIR/eval" \
    --suite "$EVAL_SUITE" \
    --modes B0 B1 B2 B3 \
    --n-episodes 3 \
    --max-student-steps 300 \
    --max-teacher-steps 300 \
    --seed 20260806 \
    --oft-server-port "$OFT_PORT"

kill_oft

if gate_passes "$RUNS_DIR/eval/dev_decision.json"; then
    echo "Dev gate: PASS"
else
    echo "Dev gate: FAIL"
    echo "Check $RUNS_DIR/eval/dev_decision.json"
fi

echo "=== Route C pipeline complete ==="
echo "Results: $RUNS_DIR"
echo "  Protocol: $PROTOCOL"
echo "  Data: $RUNS_DIR/data/round0"
echo "  Plugin: $RUNS_DIR/checkpoints/plugin_best.pt"
echo "  Eval: $RUNS_DIR/eval/paired_eval_results.json"

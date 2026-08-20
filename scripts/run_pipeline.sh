#!/bin/bash
# =========================================================================
# Route C Plugin — Full Pipeline
# Step 1: Collect R0 data (4 suites × train/dev)
# Step 2: Train (F2, hidden=128, 64 segments)
# Step 3: Headroom replay
# Step 4: Dev evaluation
#
# Usage:  bash scripts/run_route_c_pipeline.sh  [--skip-collect]
# =========================================================================
set -euo pipefail

cd /root/autodl-tmp/RASE

SKIP_COLLECT=false
if [[ "${1:-}" == "--skip-collect" ]]; then
    SKIP_COLLECT=true
fi

# ── Color helpers ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
say()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

run_py() {
    # Run a python command, tee output to log AND stdout in real-time
    local logfile=$1; shift
    echo "=== $(date) ===" >> "$logfile"
    echo "CMD: $*" >> "$logfile"
    # PYTHONUNBUFFERED=1 forces python to flush every line (conda run buffers otherwise)
    PYTHONUNBUFFERED=1 conda run --no-capture-output -p /root/autodl-tmp/envs/smolvla python -u "$@" >> "$logfile" 2>&1
    local rc=$?
    echo "EXIT_CODE: $rc" >> "$logfile"
    return $rc
}

# =========================================================================
# STEP 1 — Data Collection
# =========================================================================
SUITES=("libero_spatial" "libero_10" "libero_object" "libero_goal")
OFTCKPT=("ckpts/oft_spatial" "ckpts/oft_10" "ckpts/oft_object" "ckpts/oft_goal")
DATA_DIR="runs/route_c_r0_scaled"
LOG_DIR="runs/pipeline_logs"

mkdir -p "$DATA_DIR" "$LOG_DIR"

collect_suite() {
    local suite=$1  split=$2
    run_py "$LOG_DIR/collect_${suite}_${split}.log" \
      scripts/collect_route_c_demos.py \
        --protocol runs/route_c_protocol/protocol_c_frozen.json \
        --output-dir "$DATA_DIR" \
        --mode R0 --suite "$suite" --split "$split" \
        --n-episodes-per-task 4
}

if $SKIP_COLLECT; then
    warn "Skipping data collection (--skip-collect)"
else
    say "========== STEP 1/4: Data Collection =========="
    for i in "${!SUITES[@]}"; do
        suite="${SUITES[$i]}"; ckpt="${OFTCKPT[$i]}"

        say "Starting OFT server: ${suite} (${ckpt})"
        fuser -k 5555/tcp 2>/dev/null || true
        sleep 2
        RASE_OFT_CHECKPOINT="${ckpt}" RASE_OFT_SUITE="${suite}" \
          conda run -p /root/autodl-tmp/envs/oft \
            python -m rase.oracle.server \
            --adapter "rase.oracle.openvla_oft_adapter:create_adapter" \
            &>/root/autodl-tmp/RASE/runs/oft_server_${suite}.log &

        # Wait for server
        for j in $(seq 1 60); do
            if python3 -c "
import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.REQ);
s.connect('tcp://localhost:5555'); s.setsockopt(zmq.RCVTIMEO, 2000)
try: s.send_json({'cmd':'ping'}); s.recv_json(); print('OK')
except: pass
s.close()
" 2>/dev/null | grep -q OK; then
                ok "OFT ${suite} ready (${j}s)"
                break
            fi
            sleep 2
        done

        collect_suite "$suite" train
        n=$(ls "$DATA_DIR"/R0/*.json 2>/dev/null | wc -l)
        ok "collected ${suite}/train → total ${n} episodes"

        collect_suite "$suite" dev
        n=$(ls "$DATA_DIR"/R0/*.json 2>/dev/null | wc -l)
        ok "collected ${suite}/dev  → total ${n} episodes"

        fuser -k 5555/tcp 2>/dev/null || true; sleep 2
    done
fi

TOTAL=$(ls "$DATA_DIR"/R0/*.json 2>/dev/null | wc -l || echo 0)
say "Data collection complete: ${TOTAL} total episodes"

# =========================================================================
# STEP 2 — Training
# =========================================================================
OUT_TRAIN="runs/route_c_final"
say "========== STEP 2/4: Training (F2, hidden=128) =========="
run_py "$LOG_DIR/train_final.log" \
  scripts/train_route_c_plugin.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_TRAIN" \
    --protocol runs/route_c_protocol/protocol_c_frozen.json \
    --mode train --feature-level F2 --n-segments 64

if [ -f "$OUT_TRAIN/plugin_best.pt" ]; then
    ok "Training done → $OUT_TRAIN/plugin_best.pt"
else
    fail "Training failed — no checkpoint produced"; exit 1
fi

# =========================================================================
# STEP 3 — Headroom Replay
# =========================================================================
OUT_HEADROOM="runs/route_c_headroom_final"
say "========== STEP 3/4: Headroom Replay =========="
run_py "$LOG_DIR/headroom_final.log" \
  scripts/headroom_route_c.py \
    --data-dir "$DATA_DIR" \
    --checkpoint "$OUT_TRAIN/plugin_best.pt" \
    --output-dir "$OUT_HEADROOM"

if [ -f "$OUT_HEADROOM/headroom_replay.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT_HEADROOM/headroom_replay.json'))
o = d['overall']
print('  Student L2: {:.4f}'.format(o['mean_student_l2']))
print('  Plugin  L2: {:.4f}'.format(o['mean_plugin_l2']))
print('  Improvement: {:.1f}% median'.format(o['median_improvement_pct']))
print('  GATE:', 'PASS' if d['headroom_pass'] else 'FAIL')
"
else
    warn "Headroom results missing — continuing"
fi

# =========================================================================
# STEP 4 — Dev Evaluation
# =========================================================================
OUT_EVAL="runs/route_c_eval_final"
say "========== STEP 4/4: Dev Evaluation =========="

# Need OFT server for evaluation
fuser -k 5555/tcp 2>/dev/null || true; sleep 2
RASE_OFT_CHECKPOINT="ckpts/oft_spatial" RASE_OFT_SUITE="libero_spatial" \
  conda run -p /root/autodl-tmp/envs/oft \
    python -m rase.oracle.server \
    --adapter "rase.oracle.openvla_oft_adapter:create_adapter" \
    &>/root/autodl-tmp/RASE/runs/oft_server_eval.log &

for j in $(seq 1 60); do
    if python3 -c "
import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.REQ);
s.connect('tcp://localhost:5555'); s.setsockopt(zmq.RCVTIMEO, 2000)
try: s.send_json({'cmd':'ping'}); s.recv_json(); print('OK')
except: pass
s.close()
" 2>/dev/null | grep -q OK; then
        ok "OFT ready for eval (${j}s)"
        break
    fi
    sleep 2
done

run_py "$LOG_DIR/eval_final.log" \
  scripts/eval_route_c_plugin.py \
    --protocol runs/route_c_protocol/protocol_c_frozen.json \
    --plugin-ckpt "$OUT_TRAIN/plugin_best.pt" \
    --output-dir "$OUT_EVAL" \
    --modes B0 B3 --suite libero_spatial --n-episodes 5

# =========================================================================
# Summary
# =========================================================================
echo
echo "=============================================="
echo "  PIPELINE COMPLETE"
echo "=============================================="
echo "  Data:      ${TOTAL} episodes"
echo "  Model:     ${OUT_TRAIN}/plugin_best.pt"
echo "  Headroom:  ${OUT_HEADROOM}/headroom_replay.json"
echo "  Eval:      ${OUT_EVAL}/"
echo
ls -lh "$OUT_TRAIN/plugin_best.pt" "$OUT_HEADROOM/headroom_replay.json" 2>/dev/null
echo
say "All done."

#!/usr/bin/env bash
# PRE-C0-R0 Step 1.3: Always-On sanity check
# F0 always-on on both tasks, F2 always-on on both tasks

set -euo pipefail
cd "$(dirname "$0")/.."

PROTOCOL="runs/route_c_final/protocol_frozen.json"
F0_CKPT="runs/route_c_controls/F0/plugin_best.pt"
F2_CKPT="runs/route_c_final/plugin_best.pt"
F0_VECTOR="runs/pre_c0_r0/f0_constant_vector.json"
OUTDIR="runs/pre_c0_r0"
MAX_STEPS=300
SEED=20260807

echo "=== F0 Always-On (constant delta from f0_vector.json) ==="
for TASK in libero_spatial_000002 libero_spatial_000004; do
    echo "--- Task: $TASK (feature_level=F0) ---"
    python scripts/eval_route_c_paired.py \
        --protocol "$PROTOCOL" \
        --plugin-ckpt "$F0_CKPT" \
        --output-dir "$OUTDIR" \
        --arm b3 \
        --suite libero_spatial \
        --task-id "$TASK" \
        --n-episodes 20 \
        --max-student-steps $MAX_STEPS \
        --max-teacher-steps $MAX_STEPS \
        --seed $SEED \
        --feature-level F0 \
        --always-on \
        --constant-delta "$F0_VECTOR" \
        --delta-scale 1.0 \
        --trace-jsonl "${OUTDIR}/always_on_f0_${TASK}.jsonl" \
        2>&1 | tee "${OUTDIR}/always_on_f0_${TASK}.log"
    sleep 5
done

echo ""
echo "=== F2 Always-On (SmolVLA latent features) ==="
for TASK in libero_spatial_000002 libero_spatial_000004; do
    echo "--- Task: $TASK (feature_level=F2) ---"
    python scripts/eval_route_c_paired.py \
        --protocol "$PROTOCOL" \
        --plugin-ckpt "$F2_CKPT" \
        --output-dir "$OUTDIR" \
        --arm b3 \
        --suite libero_spatial \
        --task-id "$TASK" \
        --n-episodes 20 \
        --max-student-steps $MAX_STEPS \
        --max-teacher-steps $MAX_STEPS \
        --seed $SEED \
        --feature-level F2 \
        --always-on \
        --trace-jsonl "${OUTDIR}/always_on_f2_${TASK}.jsonl" \
        2>&1 | tee "${OUTDIR}/always_on_f2_${TASK}.log"
    sleep 5
done

echo ""
echo "=== DONE ==="
echo "Results in: $OUTDIR/always_on_*.jsonl"

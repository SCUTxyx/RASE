#!/usr/bin/env bash
# PRE-C0-R2 corrected pipeline: train-split labels -> grouped-CV gate -> paired dev eval.
set -uo pipefail

ROOT=/root/autodl-tmp/RASE
cd "$ROOT"

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
OUT=runs/pre_c0_r2
LOG="$OUT/pipeline.log"
PROTOCOL=runs/route_c_final/protocol_frozen.json
F0_PLUGIN=runs/route_c_controls/F0/plugin_best.pt
F0_VECTOR=runs/pre_c0_r0/f0_constant_vector.json
MANIFEST=runs/route_c_final/s2_manifest_b3.json
OFT_PORT=5555
OFT_PID=""

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

cleanup_oft() {
    if [ -n "${OFT_PID:-}" ]; then
        kill "$OFT_PID" 2>/dev/null || true
        wait "$OFT_PID" 2>/dev/null || true
        OFT_PID=""
    fi
    pkill -f 'python.*rase\.oracle\.server.*5555' 2>/dev/null || true
}
trap cleanup_oft EXIT

echo "=== PRE-C0-R2 corrected pipeline $(date -Is) ==="
echo "code=$(git rev-parse HEAD)"
echo "protocol=$PROTOCOL"

echo "--- Static preflight ---"
$PY -m py_compile scripts/collect_activation_labels.py scripts/train_activation_gate.py \
    scripts/eval_route_c_paired.py scripts/summarize_pre_c0_r2.py || exit 10
$PY -c '
import json
def keys(path):
  xs=[json.loads(x) for x in open(path) if x.strip()]
  return {(x["task_id"],int(x["init_state_id"]),int(x["seed"])) for x in xs}
b0=keys("runs/route_c_final/paired_results_b0.jsonl")
bounded=keys("runs/route_c_final/ablation/paired_results_b3.jsonl")
manifest={(x["task_id"],int(x["init_state_id"]),int(x["seed"])) for x in json.load(open("runs/route_c_final/s2_manifest_b3.json"))}
assert len(b0)==len(bounded)==len(manifest)==40, (len(b0),len(bounded),len(manifest))
assert b0==bounded==manifest
print("paired manifest preflight PASS: 40 keys")
' || exit 11

echo "--- Start spatial OFT server ---"
cleanup_oft
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$ROOT:${PYTHONPATH:-}"
export RASE_OFT_CHECKPOINT="$ROOT/ckpts/oft_spatial"
export RASE_OFT_SUITE=libero_spatial
"$OFT_PY" -m rase.oracle.server \
    --endpoint "tcp://127.0.0.1:${OFT_PORT}" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    > "$OUT/oft_spatial.log" 2>&1 &
OFT_PID=$!
echo "OFT pid=$OFT_PID"

ready=0
for _ in $(seq 1 180); do
    if "$PY" -c "from rase.oracle.client import OracleClient; c=OracleClient('tcp://127.0.0.1:${OFT_PORT}',timeout_ms=3000); print(c.health()); c.close()" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$OFT_PID" 2>/dev/null; then
        echo "OFT server exited"; tail -100 "$OUT/oft_spatial.log"; exit 12
    fi
    sleep 2
done
if [ "$ready" -ne 1 ]; then echo "OFT health timeout"; exit 13; fi
echo "OFT health PASS"

echo "--- Step 1: task-disjoint activation labels ---"
"$PY" scripts/collect_activation_labels.py \
    --protocol "$PROTOCOL" \
    --f0-vector "$F0_VECTOR" \
    --output-dir "$OUT" \
    --suite libero_spatial \
    --split train \
    --task-limit 4 \
    --n-episodes-per-task 8 \
    --snapshot-limit 60 \
    --max-snapshots-per-episode 3 \
    --seed 20260808 \
    --dev-high 0.15 --dev-low 0.05 --dev-recover 0.10 \
    --oft-port "$OFT_PORT" || exit 20

cleanup_oft
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "--- Step 2: grouped-CV gate training ---"
"$PY" scripts/train_activation_gate.py \
    --labels-path "$OUT/activation_labels.jsonl" \
    --output-dir "$OUT" \
    --device cuda \
    --epochs 200 --lr 1e-3 --hidden-dim 16 \
    --seed 20260808 --patience 35
TRAIN_RC=$?
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "TRAIN-GATE-FAIL rc=$TRAIN_RC; stopping before expensive dev eval"
    echo "TRAIN-GATE-FAIL" > "$OUT/decision.txt"
    exit 0
fi

echo "--- Step 3a: learned gate + envelope, 40 paired dev episodes ---"
mkdir -p "$OUT/eval_gate" "$OUT/eval_envelope_only"
"$PY" scripts/eval_route_c_paired.py \
    --protocol "$PROTOCOL" --plugin-ckpt "$F0_PLUGIN" \
    --constant-delta "$F0_VECTOR" --feature-level F0 \
    --gate-ckpt "$OUT/gate_checkpoint.pt" --lean-features \
    --output-dir "$OUT/eval_gate" --episode-manifest "$MANIFEST" \
    --arm b3 --suite libero_spatial --max-student-steps 300 \
    --trace-jsonl "$OUT/eval_gate/traces.jsonl" || exit 30

echo "--- Step 3b: same envelope with always-allow threshold, 40 paired dev episodes ---"
"$PY" scripts/eval_route_c_paired.py \
    --protocol "$PROTOCOL" --plugin-ckpt "$F0_PLUGIN" \
    --constant-delta "$F0_VECTOR" --feature-level F0 \
    --gate-ckpt "$OUT/gate_checkpoint.pt" --gate-threshold 0.0 --lean-features \
    --output-dir "$OUT/eval_envelope_only" --episode-manifest "$MANIFEST" \
    --arm b3 --suite libero_spatial --max-student-steps 300 \
    --trace-jsonl "$OUT/eval_envelope_only/traces.jsonl" || exit 31

cp "$OUT/eval_gate/paired_results_b3.jsonl" "$OUT/paired_results_gate.jsonl"
cp "$OUT/eval_envelope_only/paired_results_b3.jsonl" "$OUT/paired_results_envelope_only.jsonl"

echo "--- Step 4: paired summary ---"
"$PY" scripts/summarize_pre_c0_r2.py \
    --output-dir "$OUT" \
    --b0 runs/route_c_final/paired_results_b0.jsonl \
    --bounded runs/route_c_final/ablation/paired_results_b3.jsonl \
    --gate "$OUT/paired_results_gate.jsonl" \
    --envelope-only "$OUT/paired_results_envelope_only.jsonl" \
    --labels "$OUT/activation_labels.jsonl" \
    --training-report "$OUT/gate_training_report.json"

echo "=== PRE-C0-R2 complete $(date -Is) ==="

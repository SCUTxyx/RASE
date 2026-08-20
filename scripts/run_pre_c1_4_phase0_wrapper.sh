#!/usr/bin/env bash
# PRE-C1.4-R3 Phase 0 Live
# Starts OFT server for Object suite, runs restore parity + causal-unit pilot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

SUITE="${1:-Object}"
OFT_SUITE="libero_object"
OFT_CKPT="ckpts/oft_object"
ENDPOINT="tcp://127.0.0.1:5555"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
OFT_PY="${OFT_PY:-/root/autodl-tmp/envs/oft/bin/python}"
LOG_DIR="runs/rase_pre_c1_4_r3_protocol"

mkdir -p "$LOG_DIR"

echo "=== Starting OFT server for $SUITE ($OFT_SUITE) ==="
pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
sleep 2

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}"
export RASE_OFT_CHECKPOINT="$ROOT/$OFT_CKPT"
export RASE_OFT_SUITE="$OFT_SUITE"

nohup "$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter \
  > "$LOG_DIR/oft_server_phase0_${SUITE}.log" 2>&1 &

for _ in $(seq 1 180); do
  if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$OFT_SUITE" >/dev/null 2>&1; then
    echo "OFT server ready ($SUITE)"
    break
  fi
  sleep 2
done

echo "=== Running Phase 0 Live: $SUITE ==="
"$PY" scripts/run_pre_c1_4_phase0_live.py \
  --suite "$SUITE" \
  --endpoint "$ENDPOINT" \
  --output-dir "$LOG_DIR" \
  --limit-anchors 3

RC=$?
pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
echo "Phase 0 exit code: $RC"
if [ $RC -eq 0 ]; then
    echo "=== Phase 0 PASSED. H_star frozen, ready for Phase 1 ==="
else
    echo "=== Phase 0 needs attention (check gate files) ==="
fi
exit $RC

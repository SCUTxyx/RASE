#!/bin/bash
# Phase 6: Scaled R0 data collection across all 4 suites.
# Each suite needs its own OFT checkpoint — restart server between suites.
set -euo pipefail

cd /root/autodl-tmp/RASE
mkdir -p runs/route_c_r0_scaled

SUITES=("libero_spatial" "libero_10" "libero_object" "libero_goal")
OFTCKPT=("ckpts/oft_spatial" "ckpts/oft_10" "ckpts/oft_object" "ckpts/oft_goal")

collect_suite() {
    local suite=$1
    local split=$2
    echo ">>> ${suite}/${split} <<<"
    conda run -p /root/autodl-tmp/envs/smolvla \
      python scripts/collect_route_c_demos.py \
        --protocol runs/route_c_protocol/protocol_c_frozen.json \
        --output-dir runs/route_c_r0_scaled \
        --mode R0 --suite "$suite" --split "$split" \
        --n-episodes-per-task 4
}

for i in "${!SUITES[@]}"; do
    suite="${SUITES[$i]}"
    ckpt="${OFTCKPT[$i]}"

    # Kill old OFT server and free GPU
    fuser -k 5555/tcp 2>/dev/null || true
    sleep 2

    # Start OFT server for this suite
    echo "=== Starting OFT server: ${suite} (${ckpt}) ==="
    RASE_OFT_CHECKPOINT="${ckpt}" RASE_OFT_SUITE="${suite}" \
      conda run -p /root/autodl-tmp/envs/oft \
        python -m rase.oracle.server \
        --adapter "rase.oracle.openvla_oft_adapter:create_adapter" \
        &>/root/autodl-tmp/RASE/runs/oft_server_${suite}.log &

    # Wait for server to be ready (poll)
    for j in $(seq 1 60); do
        if python3 -c "
import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.REQ);
s.connect('tcp://localhost:5555'); s.setsockopt(zmq.RCVTIMEO, 2000)
try:
    s.send_json({'cmd':'ping'}); s.recv_json(); print('OK')
except: pass
s.close()
" 2>/dev/null | grep -q OK; then
            echo "  OFT server ready (waited ${j}s)"
            break
        fi
        sleep 2
    done

    # Collect train + dev
    collect_suite "$suite" train
    collect_suite "$suite" dev

    # Cleanup
    fuser -k 5555/tcp 2>/dev/null || true
    sleep 2
done

echo "=== ALL DONE ==="
python3 -c "
from pathlib import Path
p = Path('runs/route_c_r0_scaled/R0')
n = len(list(p.glob('*.json'))) if p.exists() else 0
print(f'Total episodes: {n}')
"

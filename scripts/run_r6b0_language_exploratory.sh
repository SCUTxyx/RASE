#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
DATA=runs/pre_c0_r6/r6b0_takeover_language_v2.npz
REPORT=runs/pre_c0_r6/r6b0_takeover_language_v2.report.json
ROOT=runs/pre_c0_r6/r6b0_language_exploratory_v1
mkdir -p "$ROOT"

run_group() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  for seed in 10 11 12 13 14; do
    "$PY" scripts/train_r6b0_takeover_oof.py \
      --dataset "$DATA" --dataset-report "$REPORT" --use-language \
      --output "$out/seed_${seed}.json" --seed "$seed" --epochs 100 "$@"
  done
}

run_group shared_id_calibrated --mode shared_id_calibrated > "$ROOT/shared_id_calibrated.log" 2>&1 &
run_group per_vla_pi0fast --mode per_vla --target-policy pi0fast_libero > "$ROOT/per_vla_pi0fast.log" 2>&1 &
run_group per_vla_pi05 --mode per_vla --target-policy pi05_libero > "$ROOT/per_vla_pi05.log" 2>&1 &
wait
echo complete > "$ROOT/COMPLETE"

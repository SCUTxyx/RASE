#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
DATA=runs/pre_c0_r6/r6b0_takeover_v1.npz
REPORT=runs/pre_c0_r6/r6b0_takeover_v1.report.json
ROOT=runs/pre_c0_r6/r6b0_comparisons_v1
mkdir -p "$ROOT"

run_group() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  for seed in 10 11 12 13 14; do
    if [[ -s "$out/seed_${seed}.json" ]]; then
      continue
    fi
    "$PY" scripts/train_r6b0_takeover_oof.py \
      --dataset "$DATA" --dataset-report "$REPORT" \
      --output "$out/seed_${seed}.json" --seed "$seed" --epochs 100 "$@"
  done
}

run_group shared_universal --mode shared_universal > "$ROOT/shared_universal.log" 2>&1 &
run_group per_vla_pi0fast --mode per_vla --target-policy pi0fast_libero > "$ROOT/per_vla_pi0fast.log" 2>&1 &
run_group per_vla_pi05 --mode per_vla --target-policy pi05_libero > "$ROOT/per_vla_pi05.log" 2>&1 &
wait

run_group zero_pi0fast_to_pi05 --mode zero_shot --source-policy pi0fast_libero --target-policy pi05_libero > "$ROOT/zero_pi0fast_to_pi05.log" 2>&1 &
run_group zero_pi05_to_pi0fast --mode zero_shot --source-policy pi05_libero --target-policy pi0fast_libero > "$ROOT/zero_pi05_to_pi0fast.log" 2>&1 &
run_group loo_pi0fast --mode loo --target-policy pi0fast_libero > "$ROOT/loo_pi0fast.log" 2>&1 &
wait

run_group loo_pi05 --mode loo --target-policy pi05_libero > "$ROOT/loo_pi05.log" 2>&1
echo complete > "$ROOT/COMPLETE"

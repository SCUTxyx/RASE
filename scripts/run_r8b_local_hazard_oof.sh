#!/usr/bin/env bash
# Run the five pre-registered canonical R8-B shared+VLA-ID OOF probes.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
DATASET=runs/pre_c0_r6/r6c1b_replica_aggregated_v2/r6c_candidate_arm_dataset.npz
REPORT=runs/pre_c0_r6/r6c1b_replica_aggregated_v2/r6c_candidate_arm_dataset.npz.report.json
A0=runs/pre_c0_r8/r8a_recoverability_hazard_v1.json
A1=runs/pre_c0_r8/r8a1_rep3_pilot_audit_v1.json
PROTOCOL=configs/r8b_local_recoverability_hazard_probe_v1.json
OUT=runs/pre_c0_r8/r8b_local_hazard_oof_v1
mkdir -p "$OUT"

"$PY" - "$A1" "$PROTOCOL" <<'PY'
import json, sys
a1, protocol = (json.load(open(path)) for path in sys.argv[1:])
if a1.get("status") != "PASS":
    raise SystemExit("R8-B remains locked: R8-A1 did not PASS")
if protocol.get("status") != "frozen_before_r8a1_outcome":
    raise SystemExit("R8-B protocol is not frozen")
PY

for seed in 701 702 703 704 705; do
  output="$OUT/seed_${seed}.json"
  if [[ -f "$output" ]]; then
    echo "R8B skip seed $seed: report exists"
    continue
  fi
  "$PY" scripts/train_r8b_local_hazard_probe.py \
    --dataset "$DATASET" --dataset-report "$REPORT" \
    --r8a0-audit "$A0" --r8a1-audit "$A1" --protocol "$PROTOCOL" \
    --output "$output" --seed "$seed" --members 3 --folds 5 \
    --epochs 180 --bootstrap-samples 1000 --policy-conditioning id
done

"$PY" scripts/audit_r8b_hazard_stability.py \
  --input-root "$OUT" --protocol "$PROTOCOL" --output "$OUT/stability.json"

#!/usr/bin/env bash
# Manually launched only after R7-A label support has passed.  This script is
# intentionally not called by run_r7a_after_reset.sh.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT="${R7_SOURCE_ROOT:-runs/pre_c0_r7/r7a_pi0fast_source_labels_v1}"
DATASET="${R7_DATASET:-$ROOT/r7a_source_risk_dataset.npz}"
REPORT="$DATASET.report.json"
AUDIT="${R7_LABEL_AUDIT:-$ROOT/label_support.json}"
REPEAT_AUDIT="${R7_EXACT_REPEAT_AUDIT:-$ROOT/exact_repeat_audit.json}"
OUT="${R7_OOF_ROOT:-runs/pre_c0_r7/r7a_source_risk_oof_v1}"
SEEDS=(2026081207 2026081208 2026081209 2026081210 2026081211)

"$PY" - "$AUDIT" "$REPEAT_AUDIT" "$DATASET" "$REPORT" <<'PY'
import json, pathlib, sys
audit, repeat, dataset, report = map(pathlib.Path, sys.argv[1:])
if not (audit.is_file() and repeat.is_file() and dataset.is_file() and report.is_file()):
    raise SystemExit("R7A_RISK STOP: frozen label/repeat audit/dataset/report is incomplete")
row = json.loads(audit.read_text())
if row.get("status") != "PASS" or row.get("states") not in (191, 192) or row.get("tasks") != 48:
    raise SystemExit("R7A_RISK STOP: label-support gate did not pass")
if json.loads(repeat.read_text()).get("status") != "PASS":
    raise SystemExit("R7A_RISK STOP: exact-repeat stability gate did not pass")
print("R7A_RISK label and exact-repeat gates PASS")
PY

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  target="$OUT/seed_${seed}/report.json"
  if [[ -f "$target" ]]; then
    echo "R7A_RISK skip seed=$seed: report exists"
    continue
  fi
  mkdir -p "$(dirname "$target")"
  "$PY" scripts/train_r7a_source_risk_probe.py \
    --dataset "$DATASET" --dataset-report "$REPORT" --label-audit "$AUDIT" \
    --exact-repeat-audit "$REPEAT_AUDIT" \
    --output "$target" --seed "$seed" --fold-seed 2026081207 \
    --folds 5 --members 3 --epochs 180 --bootstrap-samples 1000 --device cuda
done

"$PY" scripts/audit_r7a_source_risk_stability.py \
  --input-root "$OUT" --output "$OUT/stability.json"

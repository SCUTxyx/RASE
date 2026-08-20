#!/usr/bin/env bash
# Build the aligned qualified-policy dataset and run the fixed shared-model ladder.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
DATASET=runs/pre_c0_r7/r7c_multivla_source_dataset_v1.npz
REPORT="$DATASET.report.json"
OUT=runs/pre_c0_r7/r7c_multivla_source_oof_v1
SEEDS=(2026081207 2026081208 2026081209 2026081210 2026081211)
MODES=(pooled shared_id shared_desc shared_calib)

declare -a build_args=()
declare -a stability_args=()
qualified=0

add_if_qualified() {
  local source_root="$1" oof_root="$2"
  local stability="$oof_root/stability.json"
  if [[ ! -f "$stability" ]]; then
    return 0
  fi
  if "$PY" - "$stability" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
raise SystemExit(0 if row.get("status") == "PASS"
                 and row.get("decision") == "FULL_PASS" else 1)
PY
  then
    build_args+=(--dataset "$source_root/r7a_source_risk_dataset.npz")
    build_args+=(--dataset-report "$source_root/r7a_source_risk_dataset.npz.report.json")
    stability_args+=(--per-vla-stability "$stability")
    qualified=$((qualified + 1))
  fi
}

add_if_qualified runs/pre_c0_r7/r7a_pi0fast_source_labels_v1 \
  runs/pre_c0_r7/r7a_source_risk_oof_v1
add_if_qualified runs/pre_c0_r7/r7b_pi05_libero_source_labels_v1 \
  runs/pre_c0_r7/r7b_pi05_libero_source_risk_oof_v1
add_if_qualified runs/pre_c0_r7/r7b_smolvla_libero_source_labels_v1 \
  runs/pre_c0_r7/r7b_smolvla_libero_source_risk_oof_v1

if [[ "$qualified" -lt 2 ]]; then
  echo "R7C STOP: only $qualified source VLAs FULL_PASS; require at least two" >&2
  exit 20
fi

"$PY" scripts/build_r7c_multivla_source_dataset.py \
  "${build_args[@]}" --output "$DATASET"

for seed in "${SEEDS[@]}"; do
  mkdir -p "$OUT/seed_${seed}"
  for mode in "${MODES[@]}"; do
    target="$OUT/seed_${seed}/${mode}.json"
    [[ -f "$target" ]] && continue
    "$PY" scripts/train_r7c_multivla_source_risk.py \
      --dataset "$DATASET" --dataset-report "$REPORT" --output "$target" \
      --mode "$mode" --seed "$seed" --folds 5 --members 3 --epochs 180 \
      --bootstrap-samples 1000 --device cuda
  done
done

for mode in "${MODES[@]}"; do
  "$PY" scripts/audit_r7c_multivla_stability.py \
    --input-root "$OUT" --mode "$mode" "${stability_args[@]}" \
    --output "$OUT/stability_${mode}.json"
done

# Zero-shot/LOVO is a challenge metric.  Run it only after the canonical
# shared+descriptor+calibration model passes the strict shared gate.
if "$PY" - "$OUT/stability_shared_calib.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)
PY
then
  policies=( $("$PY" - "$REPORT" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1]))["policies"])
PY
) )
  for seed in "${SEEDS[@]}"; do
    for policy in "${policies[@]}"; do
      target="$OUT/seed_${seed}/heldout_${policy}.json"
      [[ -f "$target" ]] && continue
      "$PY" scripts/train_r7c_lovo_adaptation.py \
        --dataset "$DATASET" --dataset-report "$REPORT" \
        --heldout-policy "$policy" --output "$target" --seed "$seed" \
        --folds 5 --members 3 --epochs 180 --bootstrap-samples 1000 --device cuda
    done
  done
  for policy in "${policies[@]}"; do
    "$PY" scripts/audit_r7c_lovo_stability.py \
      --input-root "$OUT" --heldout-policy "$policy" \
      --output "$OUT/stability_heldout_${policy}.json"
  done
else
  echo "R7C LOVO locked: shared_calib gate did not PASS"
fi

"$PY" scripts/audit_r7d_selector_readiness.py \
  "${stability_args[@]}" \
  --shared-stability "$OUT/stability_shared_calib.json" \
  --output runs/pre_c0_r7/r7d_selector_readiness.json

echo "R7C_MULTIVLA_SOURCE_OOF COMPLETE"

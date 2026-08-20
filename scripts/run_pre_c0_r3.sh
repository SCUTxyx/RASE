#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RASE
PY_AUDIT="${PY_AUDIT:-$(command -v python3)}"
OUT="${OUT:-runs/pre_c0_r3}"
PRE_A3="${PRE_A3:-runs/rase_pre_a3_recovery_duration120_train_v1}"
FEATURES="${FEATURES:-$OUT/boundary_features.jsonl}"
mkdir -p "$OUT"

"$PY_AUDIT" scripts/audit_pre_a3_operator_opportunity.py \
  --input-root "$PRE_A3" \
  --output "$OUT/opportunity_audit.json" \
  --matrix-output "$OUT/operator_matrix.jsonl" \
  --min-complete-states 60 --min-oracle-gap 0.05 \
  --min-winning-operators 2 --min-tasks-per-winning-operator 2 \
  --max-best-fixed-harm 0.05 |& tee "$OUT/opportunity_audit.log" || audit_rc=$?

audit_rc="${audit_rc:-0}"
if [[ "$audit_rc" -eq 2 ]]; then
  echo "PRE-C0-R3 STOP: operator opportunity gate failed." | tee "$OUT/decision.txt"
  exit 0
elif [[ "$audit_rc" -ne 0 ]]; then
  echo "PRE-C0-R3 ERROR: opportunity audit crashed (rc=$audit_rc)." | tee "$OUT/decision.txt"
  exit "$audit_rc"
fi

if [[ ! -s "$FEATURES" ]]; then
  echo "PRE-C0-R3 OPPORTUNITY PASS; boundary feature collection is required next." \
    | tee "$OUT/decision.txt"
  exit 0
fi

# Locate a Python environment with PyTorch only when learned-model stages are
# actually reached.  PRE-A3 opportunity audit is intentionally torch-free.
PY_TORCH=""
for candidate in "${PY:-}" "$(command -v python 2>/dev/null || true)" \
                 /root/miniconda3/envs/*/bin/python; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" -c 'import torch' >/dev/null 2>&1; then
    PY_TORCH="$candidate"
    break
  fi
done
if [[ -z "$PY_TORCH" ]]; then
  echo "PRE-C0-R3 ERROR: no Python environment with torch was found." | tee "$OUT/decision.txt"
  exit 3
fi

"$PY_TORCH" scripts/build_counterfactual_latent_dataset.py \
  --opportunity-audit "$OUT/opportunity_audit.json" \
  --features "$FEATURES" --output "$OUT/selector_dataset.jsonl" \
  |& tee "$OUT/build_dataset.log"

"$PY_TORCH" scripts/train_counterfactual_latent_world_model.py \
  --dataset "$OUT/selector_dataset.jsonl" \
  --output-dir "$OUT/world_model" --ensemble-size 5 --hidden-dim 256 \
  --epochs 300 --dynamics-weight 1.0 --selector-margin 0.10 --device cuda \
  |& tee "$OUT/train_world_model.log" || train_rc=$?

train_rc="${train_rc:-0}"
if [[ "$train_rc" -eq 2 ]]; then
  echo "PRE-C0-R3 STOP: task-held-out world-model/selector gate failed." | tee "$OUT/decision.txt"
  exit 0
fi
if [[ "$train_rc" -ne 0 ]]; then
  echo "PRE-C0-R3 ERROR: training crashed (rc=$train_rc)." | tee "$OUT/decision.txt"
  exit "$train_rc"
fi
echo "PRE-C0-R3 READY FOR CLOSED-LOOP DEV." | tee "$OUT/decision.txt"

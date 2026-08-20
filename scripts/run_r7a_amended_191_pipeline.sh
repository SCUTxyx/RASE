#!/usr/bin/env bash
# Build the exclusion-bound 191-state cohort and run OOF only after amended 16/16 PASS.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT=runs/pre_c0_r7/r7a_pi0fast_source_labels_v1
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json
EXCLUSION="$ROOT/reproducibility_exclusions_v1.json"
LABEL="$ROOT/label_support_amended_191.json"
MANIFEST="$ROOT/exact_repeat_manifest_amended_191.json"
REPEAT_AUDIT="$ROOT/exact_repeat_audit_amended_191.json"
DATASET="$ROOT/r7a_source_risk_dataset_191.npz"
OOF=runs/pre_c0_r7/r7a_source_risk_oof_191_v1

"$PY" scripts/audit_r7a_source_labels.py --initial-keys "$KEYS" --input-root "$ROOT" \
  --output "$LABEL" --policy-id pi0fast_libero --exclusion-manifest "$EXCLUSION"

R7_LABEL_AUDIT="$LABEL" R7_EXCLUSION_MANIFEST="$EXCLUSION" \
R7_EXACT_REPEAT_MANIFEST="$MANIFEST" R7_EXACT_REPEAT_AUDIT="$REPEAT_AUDIT" \
bash scripts/run_r7a_exact_repeat.sh

"$PY" - "$REPEAT_AUDIT" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
if row.get("status") != "PASS" or row.get("audited_records") != 16:
    raise SystemExit("R7A amended exact-repeat is not strict 16/16 PASS; OOF stays locked")
PY

R7_LABEL_AUDIT="$LABEL" R7_EXACT_REPEAT_AUDIT="$REPEAT_AUDIT" \
R7_EXCLUSION_MANIFEST="$EXCLUSION" R7_DATASET="$DATASET" \
bash scripts/run_r7a_build_source_dataset.sh

R7_LABEL_AUDIT="$LABEL" R7_EXACT_REPEAT_AUDIT="$REPEAT_AUDIT" \
R7_DATASET="$DATASET" R7_OOF_ROOT="$OOF" bash scripts/run_r7a_source_risk_oof.sh

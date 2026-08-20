#!/usr/bin/env bash
# Gate-controlled R7 continuation launched independently of the already-running
# reset/source driver.  It is safe to start early: it only waits until the
# formal label audit exists, stops on FAIL, and never runs OFT/selector/WM.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT=runs/pre_c0_r7/r7a_pi0fast_source_labels_v1
AUDIT="$ROOT/label_support.json"
DATASET="$ROOT/r7a_source_risk_dataset.npz"
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json

echo "R7A_POSTLABEL waiting_for=$AUDIT"
while [[ ! -f "$AUDIT" ]]; do
  sleep 15
done

"$PY" - "$AUDIT" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
if row.get("status") != "PASS":
    raise SystemExit("R7A_POSTLABEL STOP: label-support gate is not PASS")
if row.get("states") != 192 or row.get("tasks") != 48:
    raise SystemExit("R7A_POSTLABEL STOP: audit is not the complete 192/48 cohort")
print("R7A_POSTLABEL label-support PASS")
PY

scripts/run_r7a_exact_repeat.sh

# The wrapper serializes the original and post-label drivers and binds the
# dataset to both source-label and exact-repeat audit hashes.
scripts/run_r7a_build_source_dataset.sh

scripts/run_r7a_source_risk_oof.sh
echo "R7A_POSTLABEL COMPLETE_CANONICAL_SOURCE_RISK"

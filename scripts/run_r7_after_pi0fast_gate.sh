#!/usr/bin/env bash
# Wait for the canonical Pi0Fast verdict, then conditionally execute R7-B/C.
set -euo pipefail
cd /root/autodl-tmp/RASE

STABILITY=runs/pre_c0_r7/r7a_source_risk_oof_v1/stability.json
echo "R7_AFTER_PI0FAST waiting_for=$STABILITY"
while [[ ! -f "$STABILITY" ]]; do sleep 30; done

if ! /root/autodl-tmp/envs/smolvla/bin/python - "$STABILITY" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
raise SystemExit(0 if row.get("status") == "PASS"
                 and row.get("decision") == "FULL_PASS" else 1)
PY
then
  echo "R7_AFTER_PI0FAST STOP: canonical source-risk did not FULL_PASS"
  exit 0
fi

scripts/run_r7b_multivla_source_pipeline.sh
scripts/run_r7c_multivla_source_oof.sh
echo "R7_AFTER_PI0FAST COMPLETE"

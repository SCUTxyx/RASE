#!/usr/bin/env bash
# Gate-controlled source-only cohorts for Pi0.5 and SmolVLA.
# Requires the canonical Pi0Fast R7-A five-seed source-risk gate to FULL_PASS.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
PI0FAST_STABILITY=runs/pre_c0_r7/r7a_source_risk_oof_v1/stability.json

"$PY" - "$PI0FAST_STABILITY" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("R7B STOP: Pi0Fast stability report is missing")
row = json.loads(path.read_text())
if row.get("status") != "PASS" or row.get("decision") != "FULL_PASS":
    raise SystemExit("R7B STOP: Pi0Fast source-risk did not FULL_PASS")
print("R7B Pi0Fast source-risk FULL_PASS")
PY

run_policy() {
  local policy="$1" path="$2" tokenizer="$3" action_tokenizer="$4"
  local root="runs/pre_c0_r7/r7b_${policy}_source_labels_v1"
  local oof="runs/pre_c0_r7/r7b_${policy}_source_risk_oof_v1"
  echo "R7B_SOURCE policy=$policy root=$root"

  if ! R7_POLICY_ID="$policy" R7_POLICY_PATH="$path" \
    R7_TOKENIZER_PATH="$tokenizer" R7_ACTION_TOKENIZER_PATH="$action_tokenizer" \
    R7_SOURCE_ROOT="$root" R7_MIN_FAILURES=40 R7_MIN_SUCCESSES=40 \
    R7_MIN_FAILURE_TASKS=12 R7_MIN_MIXED_TASKS=8 R7_MIN_SUITE_PER_CLASS=4 \
      scripts/run_r7a_source_labels.sh; then
    echo "R7B_SOURCE policy=$policy collection_or_contract=ERROR; continue other policy"
    return 0
  fi

  if ! "$PY" - "$root/label_support.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
raise SystemExit(0 if row.get("status") == "PASS" else 1)
PY
  then
    echo "R7B_SOURCE policy=$policy label_support=FAIL; skip repeat/dataset/OOF"
    return 0
  fi

  if ! R7_POLICY_ID="$policy" R7_POLICY_PATH="$path" \
    R7_TOKENIZER_PATH="$tokenizer" R7_ACTION_TOKENIZER_PATH="$action_tokenizer" \
    R7_SOURCE_ROOT="$root" scripts/run_r7a_exact_repeat.sh; then
    echo "R7B_SOURCE policy=$policy exact_repeat=FAIL; continue other policy"
    return 0
  fi

  if ! R7_POLICY_ID="$policy" R7_SOURCE_ROOT="$root" \
    scripts/run_r7a_build_source_dataset.sh; then
    echo "R7B_SOURCE policy=$policy dataset=FAIL; continue other policy"
    return 0
  fi

  if ! R7_SOURCE_ROOT="$root" R7_OOF_ROOT="$oof" \
    scripts/run_r7a_source_risk_oof.sh; then
    echo "R7B_SOURCE policy=$policy OOF=FAIL; continue other policy"
    return 0
  fi
}

# Pi0.5 shares the PaliGemma tokenizer and has no separate action tokenizer.
run_policy pi05_libero ckpts/pi05_libero ckpts/paligemma_tokenizer_35e4f46 ""

# SmolVLA's LeRobot checkpoint packages its own pre/postprocessors.
run_policy smolvla_libero ckpts/smolvla_libero "" ""

echo "R7B_MULTIVLA_SOURCE COMPLETE"

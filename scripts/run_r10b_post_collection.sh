#!/usr/bin/env bash
# Wait for the frozen R10-B collection, then execute only the pre-model hard
# gates in order.  A failed gate stops the chain and is preserved as a result.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
MANIFEST=${R10B_MANIFEST:-runs/pre_c0_r10/r10b_case_control_manifest_v1.json}
COLLECT=${R10B_COLLECT:-runs/pre_c0_r10/r10b_case_control_collect_v1}
REPRO=${R10B_REPRO:-runs/pre_c0_r10/r10b_case_control_repro_audit_v1.json}
DATASET=${R10B_DATASET:-runs/pre_c0_r10/r10b_case_control_dataset_v1.npz}
INFO=${R10B_INFO:-runs/pre_c0_r10/r10c_case_control_information_v1.json}

while [[ ! -f "$COLLECT/COMPLETE" ]]; do
  if ! tmux has-session -t r10b_full 2>/dev/null; then
    echo "R10B POST ERROR: collection session exited without COMPLETE" >&2
    exit 41
  fi
  sleep 30
done

echo "R10B POST: collection complete; running reproducibility gate"
if ! "$PY" scripts/audit_r10b_case_control_repro.py \
  --manifest "$MANIFEST" --collect-root "$COLLECT" --output "$REPRO"; then
  echo "R10B POST: reproducibility gate FAIL; stopping"
  printf 'repro_fail\n' > runs/pre_c0_r10/R10B_STOP
  exit 42
fi

echo "R10B POST: reproducibility gate PASS; building dataset"
"$PY" scripts/build_r10b_case_control_dataset.py \
  --manifest "$MANIFEST" --collect-root "$COLLECT" --repro-audit "$REPRO" \
  --output "$DATASET"

echo "R10B POST: running pre-model information gate"
if ! "$PY" scripts/audit_r10c_case_control_information.py \
  --dataset "$DATASET" --dataset-report "${DATASET%.npz}.npz.report.json" \
  --repro-audit "$REPRO" --output "$INFO"; then
  echo "R10B POST: information gate FAIL; no model will start"
  printf 'information_fail\n' > runs/pre_c0_r10/R10B_STOP
  exit 43
fi

printf 'information_pass\n' > runs/pre_c0_r10/R10B_READY_FOR_R10D
echo "R10B POST: information gate PASS; R10D is unlocked but not started"

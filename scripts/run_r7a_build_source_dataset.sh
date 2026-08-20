#!/usr/bin/env bash
# Serialize and hash-bind the canonical R7-A source-risk dataset build.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT="${R7_SOURCE_ROOT:-runs/pre_c0_r7/r7a_pi0fast_source_labels_v1}"
POLICY_ID="${R7_POLICY_ID:-pi0fast_libero}"
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json
AUDIT="${R7_LABEL_AUDIT:-$ROOT/label_support.json}"
REPEAT="${R7_EXACT_REPEAT_AUDIT:-$ROOT/exact_repeat_audit.json}"
EXCLUSION="${R7_EXCLUSION_MANIFEST:-}"
DATASET="${R7_DATASET:-$ROOT/r7a_source_risk_dataset.npz}"
REPORT="$DATASET.report.json"

exec 9>"$ROOT/.dataset_build.lock"
flock 9
if [[ -f "$DATASET" && -f "$REPORT" ]] && "$PY" - "$DATASET" "$REPORT" "$AUDIT" "$REPEAT" <<'PY'
import hashlib, json, sys
dataset, report, audit, repeat = sys.argv[1:]
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
row = json.load(open(report))
ok = (row.get("dataset_sha256") == sha(dataset)
      and row.get("label_audit_sha256") == sha(audit)
      and row.get("exact_repeat_audit_sha256") == sha(repeat)
      and row.get("rows") in (191, 192) and row.get("tasks") == 48)
raise SystemExit(0 if ok else 1)
PY
then
  echo "R7A_DATASET already_frozen_and_hash_bound"
  exit 0
fi

build_args=()
[[ -n "$EXCLUSION" ]] && build_args+=(--exclusion-manifest "$EXCLUSION")
"$PY" scripts/build_r7a_source_risk_dataset.py \
  --initial-keys "$KEYS" --label-audit "$AUDIT" \
  --exact-repeat-audit "$REPEAT" --input-root "$ROOT" --output "$DATASET" \
  --policy-id "$POLICY_ID" "${build_args[@]}"

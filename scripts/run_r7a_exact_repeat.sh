#!/usr/bin/env bash
# Manual, gate-controlled source-only exact-repeat audit for R7-A.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
ROOT="${R7_SOURCE_ROOT:-runs/pre_c0_r7/r7a_pi0fast_source_labels_v1}"
KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json
POLICY_ID="${R7_POLICY_ID:-pi0fast_libero}"
POLICY_PATH="${R7_POLICY_PATH:-ckpts/pi0fast_libero}"
TOKENIZER_PATH="${R7_TOKENIZER_PATH-ckpts/paligemma_tokenizer_35e4f46}"
ACTION_TOKENIZER_PATH="${R7_ACTION_TOKENIZER_PATH-ckpts/pi0fast_action_tokenizer_79ae83e}"
LABEL_AUDIT="${R7_LABEL_AUDIT:-$ROOT/label_support.json}"
EXCLUSION="${R7_EXCLUSION_MANIFEST:-}"
MANIFEST="${R7_EXACT_REPEAT_MANIFEST:-$ROOT/exact_repeat_manifest.json}"
REPEAT="$ROOT/exact_repeat"
AUDIT="${R7_EXACT_REPEAT_AUDIT:-$ROOT/exact_repeat_audit.json}"
LOCK="$ROOT/.exact_repeat.lock"

# Both the original reset driver and the independently launched post-label
# driver may reach this stage.  Serialize the expensive GPU reruns and make the
# stage content-addressed so a second caller becomes a no-op.
mkdir -p "$ROOT"
exec 9>"$LOCK"
flock 9
"$PY" - "$LABEL_AUDIT" "$POLICY_ID" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
declared = str(row.get("policy_id") or "pi0fast_libero")
if row.get("status") != "PASS" or declared != sys.argv[2]:
    raise SystemExit(
        f"R7 exact-repeat preflight failed: status={row.get('status')} "
        f"audit_policy={declared} requested_policy={sys.argv[2]}"
    )
PY
if [[ -f "$AUDIT" ]] && "$PY" - "$LABEL_AUDIT" "$AUDIT" <<'PY'
import hashlib, json, sys
label, repeat = map(str, sys.argv[1:])
value = json.load(open(repeat))
expected = hashlib.sha256(open(label, "rb").read()).hexdigest()
raise SystemExit(0 if value.get("status") == "PASS"
                 and value.get("audited_records") == 16
                 and value.get("label_audit_sha256") == expected else 1)
PY
then
  echo "R7A_EXACT_REPEAT already_passed_and_hash_bound"
  exit 0
fi

freeze_args=()
[[ -n "$EXCLUSION" ]] && freeze_args+=(--exclusion-manifest "$EXCLUSION")
"$PY" scripts/freeze_r7a_exact_repeat_manifest.py \
  --label-audit "$LABEL_AUDIT" --input-root "$ROOT" --output "$MANIFEST" \
  "${freeze_args[@]}"
if [[ -n "$EXCLUSION" ]]; then
  "$PY" - "$MANIFEST" "$EXCLUSION" <<'PY'
import json, sys
manifest, exclusion = (json.load(open(path)) for path in sys.argv[1:])
expected = exclusion.get("proposed_exact_repeat_replacement", {}).get("state_key")
selected = {row["state_key"] for row in manifest.get("records", [])}
excluded = set(exclusion.get("excluded_state_keys", []))
if not expected or expected not in selected or selected & excluded:
    raise SystemExit("amended exact-repeat manifest violates frozen replacement/exclusion")
PY
fi

for label in Spatial Object Goal Long; do
  suite=libero_${label,,}
  [[ "$label" == Long ]] && suite=libero_10
  keys=( $("$PY" - "$MANIFEST" "$REPEAT" "$label" <<'PY'
import json, pathlib, sys
for row in json.load(open(sys.argv[1]))["records"]:
    if row["suite"] != sys.argv[3]:
        continue
    target = pathlib.Path(sys.argv[2]) / f"suite_{sys.argv[3].lower()}" / "seed_0" / f"{row['state_key']}__seed0__rep1.json"
    if not target.is_file():
        print(row["state_key"])
PY
) )
  if [[ ${#keys[@]} -eq 0 ]]; then
    echo "R7A_EXACT_REPEAT suite=$label already_complete"
    continue
  fi
  args=()
  for key in "${keys[@]}"; do
    args+=(--state-key "$key")
  done
  policy_args=(--policy-path "$POLICY_PATH")
  [[ -n "$TOKENIZER_PATH" ]] && policy_args+=(--tokenizer-path "$TOKENIZER_PATH")
  [[ -n "$ACTION_TOKENIZER_PATH" ]] && policy_args+=(--action-tokenizer-path "$ACTION_TOKENIZER_PATH")
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$KEYS" --policy-id "$POLICY_ID" \
    "${policy_args[@]}" \
    --suite "$suite" --seed-index 0 --rollout-index 1 --no-oracle \
    --output-dir "$REPEAT/suite_${label,,}/seed_0" --boundary 0 --bookkeeping-mode full \
    "${args[@]}"
done

"$PY" scripts/audit_r7a_exact_repeat.py \
  --manifest "$MANIFEST" --repeat-root "$REPEAT" --output "$AUDIT"

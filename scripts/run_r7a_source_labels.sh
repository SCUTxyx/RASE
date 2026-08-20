#!/usr/bin/env bash
# R7-A2: collect true Pi0Fast source outcomes and deployable t0 features.
# No OFT branch, selector training, validation, test, or WM feature is unlocked.
set -euo pipefail
cd /root/autodl-tmp/RASE

VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
INITIAL_KEYS=runs/pre_c0_r7/r7a_reset_keys_v1.json
POLICY_ID="${R7_POLICY_ID:-pi0fast_libero}"
POLICY_PATH="${R7_POLICY_PATH:-ckpts/pi0fast_libero}"
TOKENIZER_PATH="${R7_TOKENIZER_PATH-ckpts/paligemma_tokenizer_35e4f46}"
ACTION_TOKENIZER_PATH="${R7_ACTION_TOKENIZER_PATH-ckpts/pi0fast_action_tokenizer_79ae83e}"
OUT="${R7_SOURCE_ROOT:-runs/pre_c0_r7/r7a_pi0fast_source_labels_v1}"
MIN_FAILURES="${R7_MIN_FAILURES:-48}"
MIN_SUCCESSES="${R7_MIN_SUCCESSES:-32}"
MIN_FAILURE_TASKS="${R7_MIN_FAILURE_TASKS:-16}"
MIN_MIXED_TASKS="${R7_MIN_MIXED_TASKS:-8}"
MIN_SUITE_PER_CLASS="${R7_MIN_SUITE_PER_CLASS:-4}"
mkdir -p "$OUT"

if [[ ! -f "$INITIAL_KEYS" ]]; then
  echo "R7A_SOURCE STOP: missing audited reset-key manifest $INITIAL_KEYS" >&2
  exit 20
fi

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
    *) echo "unknown suite label: $1" >&2; exit 2 ;;
  esac
}

screen_suite() {
  local label="$1" suite output key
  suite="$(actual_suite "$label")"
  output="$OUT/suite_${label,,}/seed_0"
  mkdir -p "$output"
  local -a args=()
  while read -r key; do
    [[ -z "$key" ]] && continue
    if [[ ! -f "$output/${key}__seed0.json" ]]; then
      args+=(--state-key "$key")
    fi
  done < <("$VLA_PY" - "$INITIAL_KEYS" "$label" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
for row in p["records"]:
    if row["suite"] == sys.argv[2]:
        print(row["state_key"])
PY
)
  local n_states=$(( ${#args[@]} / 2 ))
  if [[ "$n_states" -eq 0 ]]; then
    echo "R7A_SOURCE skip $label: complete"
    return 0
  fi
  echo "R7A_SOURCE suite=$label missing=$n_states"
  local -a policy_args=(--policy-path "$POLICY_PATH")
  [[ -n "$TOKENIZER_PATH" ]] && policy_args+=(--tokenizer-path "$TOKENIZER_PATH")
  [[ -n "$ACTION_TOKENIZER_PATH" ]] && policy_args+=(--action-tokenizer-path "$ACTION_TOKENIZER_PATH")
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL_KEYS" \
    --policy-id "$POLICY_ID" \
    "${policy_args[@]}" \
    --suite "$suite" --seed-index 0 --no-oracle \
    --output-dir "$output" --boundary 0 --bookkeeping-mode full \
    "${args[@]}"
}

for label in Spatial Object Goal Long; do
  screen_suite "$label"
done

"$VLA_PY" scripts/audit_r7a_source_labels.py \
  --initial-keys "$INITIAL_KEYS" --input-root "$OUT" \
  --output "$OUT/label_support.json" --policy-id "$POLICY_ID" \
  --min-failures "$MIN_FAILURES" --min-successes "$MIN_SUCCESSES" \
  --min-failure-tasks "$MIN_FAILURE_TASKS" --min-mixed-tasks "$MIN_MIXED_TASKS" \
  --min-suite-per-class "$MIN_SUITE_PER_CLASS"
echo complete > "$OUT/COMPLETE"

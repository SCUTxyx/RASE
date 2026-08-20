#!/usr/bin/env bash
# R6-C.1B-2: source-only screening rollouts (no OFT counterfactual).
# For each source VLA, roll out the new seeds over the frozen R6-C.1B candidate
# states (natural eval + train enrichment) using the cheapest possible mode
# (no snapshots, no oracle).  The screening outcome decides which states get
# OFT labels and enter train enrichment.
set -euo pipefail
cd /root/autodl-tmp/RASE

VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
INITIAL_KEYS=runs/pre_c0_r6/r6c1b_initial_keys_v1.json
OUT=runs/pre_c0_r6/r6c1b_screen_v1
mkdir -p "$OUT"

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
    *) echo "unknown suite label: $1" >&2; exit 2 ;;
  esac
}
suite_keys() {
  "$VLA_PY" - "$INITIAL_KEYS" "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for row in payload["records"]:
    if row["suite"] == sys.argv[2] and row["role"] == sys.argv[3]:
        print(row["state_key"])
PY
}
screen() {
  local label="$1" policy="$2" seed="$3" role="$4"
  local suite
  suite="$(actual_suite "$label")"
  local -a args=()
  while read -r key; do
    [[ -n "$key" ]] && args+=(--state-key "$key")
  done < <(suite_keys "$label" "$role")
  local n_states=$(( ${#args[@]} / 2 ))
  if [[ "$n_states" -eq 0 ]]; then
    echo "R6C1B_SCREEN skip $policy/$label/$role: no states" >&2
    return 0
  fi
  local output="$OUT/suite_${label,,}/$policy/$role/seed_$seed"
  # Idempotent resume: skip the batch if every state already produced a
  # source-only metadata file for this (policy, seed, role).
  local all_done=1 missing=0 key
  for key in "${args[@]}"; do
    [[ "$key" == --state-key ]] && continue
    if [[ ! -f "$output/${key}__seed${seed}.json" ]]; then all_done=0; missing=$((missing + 1)); fi
  done
  if [[ "$all_done" -eq 1 ]]; then
    echo "R6C1B_SCREEN skip $policy/$label/$role: all $n_states states already screened" >&2
    return 0
  fi
  echo "R6C1B_SCREEN resume $policy/$label/$role: $missing/$n_states states missing" >&2
  local -a policy_args
  if [[ "$policy" == pi0fast_libero ]]; then
    policy_args=(--policy-path ckpts/pi0fast_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  else
    policy_args=(--policy-path ckpts/pi05_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  fi
  echo "R6C1B_SCREEN suite=$label policy=$policy seed=$seed role=$role states=$n_states"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL_KEYS" \
    --policy-id "$policy" "${policy_args[@]}" --suite "$suite" --seed-index "$seed" \
    --no-oracle --output-dir "$output" \
    --boundary 0 --boundary 8 --boundary 16 \
    "${args[@]}" \
    --bookkeeping-mode none
}

for label in Spatial Object Goal Long; do
  # Pi0.5: screening seeds 2 and 3 on eval + enrichment candidates
  for seed in 2 3; do
    screen "$label" pi05_libero "$seed" natural_development_eval
    screen "$label" pi05_libero "$seed" train_enrichment
  done
  # Pi0Fast: screening seed 1 on eval + enrichment candidates
  screen "$label" pi0fast_libero 1 natural_development_eval
  screen "$label" pi0fast_libero 1 train_enrichment
done
echo complete > "$OUT/COMPLETE"

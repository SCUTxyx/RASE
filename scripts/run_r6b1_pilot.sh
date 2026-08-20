#!/usr/bin/env bash
# R6-B1 frozen, task-distinct cross-suite dynamic-boundary pilot.
set -euo pipefail
cd /root/autodl-tmp/RASE

OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
MANIFEST=runs/pre_c0_r6/r6b1_pilot_manifest_v1.json
# A failed v1 attempt wrote only a server log, so preserve it and use a fresh,
# immutable attempt directory for the diagnostic rerun.
OUT=runs/pre_c0_r6/r6b1_pilot_v4
ENDPOINT=tcp://127.0.0.1:5555
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
checkpoint() {
  case "$1" in
    Spatial) echo ckpts/oft_spatial ;;
    Object) echo ckpts/oft_object ;;
    Goal) echo ckpts/oft_goal ;;
    Long) echo ckpts/oft_10 ;;
  esac
}
policy_keys() {
  "$VLA_PY" - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for row in payload["records"]:
    if row["policy_id"] == sys.argv[2] and row["suite"] == sys.argv[3]:
        print(row["state_key"])
PY
}
collect() {
  local label="$1" policy="$2" seed="$3"
  local suite
  suite="$(actual_suite "$label")"
  local -a args=()
  while read -r key; do
    [[ -n "$key" ]] && args+=(--state-key "$key")
  done < <(policy_keys "$policy" "$label")
  # Each state contributes the pair: --state-key <key>.
  if [[ "${#args[@]}" -ne 4 ]]; then
    echo "R6B1_PILOT ERROR: expected two manifest states for $policy/$label, got $((${#args[@]} / 2))" >&2
    exit 2
  fi
  local output="$OUT/suite_${label,,}/$policy/seed_$seed"
  local -a policy_args
  if [[ "$policy" == pi0fast_libero ]]; then
    policy_args=(--policy-path ckpts/pi0fast_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  else
    policy_args=(--policy-path ckpts/pi05_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  fi
  echo "R6B1_PILOT collecting suite=$label policy=$policy seed=$seed states=$((${#args[@]} / 2))"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys runs/rase_ui_phase1a_replacement48_initial_keys_v2.json \
    --policy-id "$policy" "${policy_args[@]}" --suite "$suite" --seed-index "$seed" \
    --endpoint "$ENDPOINT" --output-dir "$output" \
    --boundary 0 --boundary 16 --boundary 32 "${args[@]}" \
    --bookkeeping-mode full
}

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/suite_${label,,}/oft_server.log"
  mkdir -p "$(dirname "$server_log")"
  echo "R6B1_PILOT starting OFT suite=$label checkpoint=$ckpt"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 60); do
    if "$VLA_PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    echo "R6B1_PILOT ERROR: OFT server did not become ready for suite=$label; tail follows" >&2
    tail -100 "$server_log" >&2 || true
    exit 1
  fi
  echo "R6B1_PILOT OFT ready suite=$label pid=$server_pid"
  collect "$label" pi0fast_libero 0
  collect "$label" pi05_libero 0
  collect "$label" pi05_libero 1
  cleanup
  trap - EXIT
  echo "R6B1_PILOT completed suite=$label"
done

"$VLA_PY" scripts/audit_r6b1_pilot.py --manifest "$MANIFEST" --input-root "$OUT" --output "$OUT/audit.json"
# Hard gate: every source trajectory must reproduce its frozen R6-A reference
# (rollout seed, final success, env steps) with finite features.
"$VLA_PY" scripts/audit_r6b1_source_parity.py \
  --atlas runs/pre_c0_r6/policy_pair_atlas_v1.json \
  --input-root "$OUT" --output "$OUT/parity_audit.json"
echo complete > "$OUT/COMPLETE"

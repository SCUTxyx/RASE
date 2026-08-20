#!/usr/bin/env bash
# R9-B temporal development pilot.  This runner is deliberately separate from
# R6/R7/R8 outputs and is idempotent per (suite, policy, state, replica).
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
MANIFEST=${R9B_MANIFEST:-runs/pre_c0_r9/r9b_temporal_manifest_v1.json}
OUT=${R9B_OUT:-runs/pre_c0_r9/r9b_temporal_collect_v1}
INITIAL="$OUT/r9b_initial_keys.json"
RECORD_LIMIT=${R9B_RECORD_LIMIT:-0}
REPLICA_LIMIT=${R9B_REPLICA_LIMIT:-3}
SUITE_LIMIT=${R9B_SUITE_LIMIT:-all}
mkdir -p "$OUT"

"$PY" - "$MANIFEST" "$INITIAL" <<'PY'
import json, sys
from pathlib import Path
manifest_path, output_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "frozen":
    raise SystemExit("R9-B manifest is not frozen")
payload = {
    "schema_version": "rase-r9b-compatible-initial-keys/v1",
    "status": "frozen",
    "pool": manifest["pool"],
    "state_keys": manifest["state_keys"],
    "state_keys_sha256": manifest["state_keys_sha256"],
    "records": manifest["records"],
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
    *) echo "unknown suite $1" >&2; exit 2 ;;
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
policy_index() {
  case "$1" in
    pi05_libero) echo 4 ;;
    pi0fast_libero) echo 2 ;;
    *) echo "unknown policy $1" >&2; exit 2 ;;
  esac
}

make_key_file() {
  local suite="$1" path="$2"
  "$PY" - "$MANIFEST" "$suite" "$path" "$RECORD_LIMIT" <<'PY'
import json, sys
from pathlib import Path
manifest, suite, output, limit = (json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]))
keys=[row["state_key"] for row in manifest["records"] if row["suite"] == suite]
if limit > 0: keys=keys[:limit]
output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(keys)+"\n")
if not keys: raise SystemExit(f"no R9-B records for {suite}")
PY
}

collect_batch() {
  local suite_label="$1" policy="$2" replica="$3" key_file="$4"
  local suite output seed
  suite="$(actual_suite "$suite_label")"
  seed="$(policy_index "$policy")"
  output="$OUT/suite_${suite_label,,}/$policy/rep$replica"
  mkdir -p "$output"
  local -a state_args=()
  local missing=0
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    state_args+=(--state-key "$key")
    local stem="${key}__seed${seed}"
    [[ "$replica" -eq 0 ]] || stem+="__rep${replica}"
    [[ -f "$output/${stem}.json" ]] || missing=$((missing + 1))
  done < "$key_file"
  if [[ "$missing" -eq 0 ]]; then
    echo "R9B skip $suite_label/$policy/rep$replica"
    return 0
  fi
  local -a policy_args=()
  if [[ "$policy" == pi05_libero ]]; then
    policy_args=(--policy-path ckpts/pi05_libero
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  else
    policy_args=(--policy-path ckpts/pi0fast_libero
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46
      --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  fi
  echo "R9B collect $suite_label/$policy/rep$replica missing=$missing"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL" --policy-id "$policy" "${policy_args[@]}" \
    --suite "$suite" --seed-index "$seed" --rollout-index "$replica" \
    --endpoint tcp://127.0.0.1:5555 --output-dir "$output" \
    --boundary 0 --boundary 4 --boundary 8 --boundary 12 --boundary 16 \
    --temporal-history 4 "${state_args[@]}" --bookkeeping-mode full
}

if [[ "$SUITE_LIMIT" == all ]]; then
  suites=(Spatial Object Goal Long)
else
  suites=("$SUITE_LIMIT")
fi
for label in "${suites[@]}"; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  suite_dir="$OUT/suite_${label,,}"
  mkdir -p "$suite_dir"
  server_log="$suite_dir/oft_server.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 60); do
    if "$VLA_PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -100 "$server_log" >&2 || true
    echo "R9B ERROR: OFT server not ready for $label" >&2
    exit 31
  fi
  for policy in pi05_libero pi0fast_libero; do
    key_file="$OUT/${label,,}_keys.txt"
    make_key_file "$label" "$key_file"
    for replica in $(seq 0 $((REPLICA_LIMIT - 1))); do
      collect_batch "$label" "$policy" "$replica" "$key_file"
    done
  done
  cleanup; trap - EXIT
done
printf 'collection_complete\n' > "$OUT/COMPLETE"

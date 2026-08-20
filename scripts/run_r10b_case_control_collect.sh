#!/usr/bin/env bash
# Collect causal temporal inputs and K=3 counterfactual labels for the frozen
# R10-B t8->t16 case-control representation pilot.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
MANIFEST=${R10B_MANIFEST:-runs/pre_c0_r10/r10b_case_control_manifest_v1.json}
OUT=${R10B_OUT:-runs/pre_c0_r10/r10b_case_control_collect_v1}
REPLICA_LIMIT=${R10B_REPLICA_LIMIT:-3}
GROUP_LIMIT=${R10B_GROUP_LIMIT:-0}
SUITE_LIMIT=${R10B_SUITE_LIMIT:-all}
RECORD_OFT_TRACE_HASH=${R10B_RECORD_OFT_TRACE_HASH:-0}
RECORD_OFT_CHUNK_TRACE=${R10B_RECORD_OFT_CHUNK_TRACE:-0}
INITIAL="$OUT/r10b_initial_keys.json"
mkdir -p "$OUT/batches"

"$PY" - "$MANIFEST" "$INITIAL" "$OUT/batches" "$GROUP_LIMIT" <<'PY'
import json, sys
from pathlib import Path
manifest_path, initial_path, batch_root = map(Path, sys.argv[1:4])
limit = int(sys.argv[4])
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") not in {"frozen", "frozen_diagnostic"}:
    raise SystemExit("R10-B manifest is not frozen")
records = manifest["records"][:limit] if limit > 0 else manifest["records"]
by_state = {}
for row in records:
    by_state[row["state_key"]] = row["initial_record"]
initial = {
    "schema_version": "rase-r10b-compatible-initial-keys/v1", "status": "frozen",
    "pool": manifest["pool"], "records": list(by_state.values()),
    "state_keys": sorted(by_state),
}
initial_path.write_text(json.dumps(initial, indent=2, sort_keys=True) + "\n")
groups = {}
for row in records:
    key = (row["suite"], row["policy_id"], int(row["seed_index"]))
    groups.setdefault(key, []).append(row["state_key"])
batch_root.mkdir(parents=True, exist_ok=True)
index = []
for (suite, policy, seed), keys in sorted(groups.items()):
    path = batch_root / f"{suite.lower()}__{policy}__seed{seed}.txt"
    path.write_text("\n".join(sorted(set(keys))) + "\n")
    index.append({"suite": suite, "policy_id": policy, "seed_index": seed,
                  "key_file": str(path), "groups": len(set(keys))})
(batch_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
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

collect_batch() {
  local label="$1" policy="$2" seed="$3" replica="$4" key_file="$5"
  local suite output
  suite="$(actual_suite "$label")"
  output="$OUT/suite_${label,,}/$policy/seed_${seed}/rep${replica}"
  mkdir -p "$output"
  local -a state_args=()
  local missing=0
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    state_args+=(--state-key "$key")
    stem="${key}__seed${seed}"
    [[ "$replica" -eq 0 ]] || stem+="__rep${replica}"
    [[ -f "$output/${stem}.json" ]] || missing=$((missing + 1))
  done < "$key_file"
  if [[ "$missing" -eq 0 ]]; then
    echo "R10B skip $label/$policy/seed$seed/rep$replica"
    return 0
  fi
  local -a policy_args=()
  local -a diagnostic_args=()
  if [[ "$RECORD_OFT_TRACE_HASH" == 1 ]]; then
    diagnostic_args=(--record-oft-trace-hash)
  fi
  if [[ "$RECORD_OFT_CHUNK_TRACE" == 1 ]]; then
    diagnostic_args+=(--record-oft-chunk-trace)
  fi
  if [[ "$policy" == pi05_libero ]]; then
    policy_args=(--policy-path ckpts/pi05_libero
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  else
    policy_args=(--policy-path ckpts/pi0fast_libero
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46
      --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  fi
  echo "R10B collect $label/$policy/seed$seed/rep$replica missing=$missing"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL" --policy-id "$policy" "${policy_args[@]}" \
    --suite "$suite" --seed-index "$seed" --rollout-index "$replica" \
    --endpoint tcp://127.0.0.1:5555 --output-dir "$output" \
    --boundary 0 --boundary 4 --boundary 8 --boundary 12 --boundary 16 \
    --temporal-history 8 "${state_args[@]}" --bookkeeping-mode full \
    "${diagnostic_args[@]}"
}

if [[ "$SUITE_LIMIT" == all ]]; then suites=(Spatial Object Goal Long); else suites=("$SUITE_LIMIT"); fi
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
  if [[ "$ready" != 1 ]]; then tail -100 "$server_log" >&2 || true; exit 31; fi
  "$PY" - "$OUT/batches/index.json" "$label" <<'PY' > "$OUT/batches/${label,,}_index.tsv"
import json,sys
for row in json.load(open(sys.argv[1])):
    if row["suite"] == sys.argv[2]:
        print(row["policy_id"], row["seed_index"], row["key_file"], sep="\t")
PY
  while IFS=$'\t' read -r policy seed key_file; do
    [[ -n "$policy" ]] || continue
    for replica in $(seq 0 $((REPLICA_LIMIT - 1))); do
      collect_batch "$label" "$policy" "$seed" "$replica" "$key_file"
    done
  done < "$OUT/batches/${label,,}_index.tsv"
  cleanup; trap - EXIT
done
printf 'collection_complete\n' > "$OUT/COMPLETE"

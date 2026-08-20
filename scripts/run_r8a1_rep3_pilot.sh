#!/usr/bin/env bash
# R8-A1: collect one pre-registered third replica for 32 balanced stable K=2 groups.
# This is a label-stability experiment, not a model or prevalence evaluation.
set -euo pipefail
cd /root/autodl-tmp/RASE

OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
DATASET=runs/pre_c0_r6/r6c1b_replica_aggregated_v2/r6c_candidate_arm_dataset.npz
DATASET_REPORT=runs/pre_c0_r6/r6c1b_replica_aggregated_v2/r6c_candidate_arm_dataset.npz.report.json
HAZARD_AUDIT=runs/pre_c0_r8/r8a_recoverability_hazard_v1.json
INITIAL_KEYS=runs/pre_c0_r6/r6c1b_initial_keys_v1.json
RAW_ROOT=runs/pre_c0_r6/r6c1b_collect_v1
MANIFEST=runs/pre_c0_r8/r8a1_rep3_pilot_manifest_v1.json
OUT=runs/pre_c0_r8/r8a1_rep3_pilot_v1
AUDIT=runs/pre_c0_r8/r8a1_rep3_pilot_audit_v1.json
BATCH_ROOT="$OUT/batches"
mkdir -p "$OUT" "$BATCH_ROOT"

available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || "$available_kb" -lt 5242880 ]]; then
  echo "R8A1 STOP: at least 5 GiB free space is required" >&2
  exit 30
fi

if [[ ! -f "$MANIFEST" ]]; then
  "$VLA_PY" scripts/freeze_r8a1_rep3_pilot_manifest.py \
    --dataset "$DATASET" --dataset-report "$DATASET_REPORT" \
    --hazard-audit "$HAZARD_AUDIT" --initial-keys "$INITIAL_KEYS" \
    --raw-root "$RAW_ROOT" --output "$MANIFEST"
fi

# Bind the run to a frozen manifest and materialize deterministic batch files.
"$VLA_PY" - "$MANIFEST" "$BATCH_ROOT" <<'PY'
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path

manifest_path, root = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
if manifest.get("status") != "frozen" or len(manifest.get("records", [])) != 32:
    raise SystemExit("R8-A1 manifest is not frozen at 32 records")
batches = defaultdict(list)
for row in manifest["records"]:
    role = "natural_development_eval" if row["cohort_role"] == "natural" else "train_enrichment"
    batches[(row["suite"], row["policy_id"], int(row["seed_index"]), role)].append(row["state_key"])
for (suite, policy, seed, role), keys in sorted(batches.items()):
    path = root / suite.lower() / policy / role / f"seed_{seed}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(keys)) + "\n")
print(json.dumps({"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "records": len(manifest["records"]), "batches": len(batches)}, sort_keys=True))
PY

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

collect_batch() {
  local label="$1" policy="$2" seed="$3" role="$4" key_file="$5"
  local suite output missing=0
  suite="$(actual_suite "$label")"
  output="$OUT/suite_${label,,}/$policy/$role/seed_$seed/rep2"
  mkdir -p "$output"
  local -a state_args=()
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    state_args+=(--state-key "$key")
    [[ -f "$output/${key}__seed${seed}__rep2.json" ]] || missing=$((missing + 1))
  done < "$key_file"
  if [[ "$missing" -eq 0 ]]; then
    echo "R8A1 skip $label/$policy/$role/seed$seed: complete"
    return 0
  fi
  local -a policy_args
  if [[ "$policy" == pi0fast_libero ]]; then
    policy_args=(--policy-path ckpts/pi0fast_libero \
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
      --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  elif [[ "$policy" == pi05_libero ]]; then
    policy_args=(--policy-path ckpts/pi05_libero \
      --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  else
    echo "unsupported policy in frozen manifest: $policy" >&2
    exit 2
  fi
  echo "R8A1 collect $label/$policy/$role/seed$seed: $missing missing"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL_KEYS" --policy-id "$policy" "${policy_args[@]}" \
    --suite "$suite" --seed-index "$seed" --rollout-index 2 \
    --endpoint tcp://127.0.0.1:5555 --output-dir "$output" \
    --boundary 0 --boundary 8 --boundary 16 "${state_args[@]}" \
    --bookkeeping-mode full
}

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/suite_${label,,}/oft_server.log"
  mkdir -p "$(dirname "$server_log")"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 60); do
    if "$VLA_PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 \
      --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -100 "$server_log" >&2 || true
    echo "R8A1 ERROR: OFT server failed for $label" >&2
    exit 31
  fi
  while IFS= read -r key_file; do
    rel="${key_file#"$BATCH_ROOT/${label,,}/"}"
    policy="${rel%%/*}"
    rest="${rel#*/}"
    role="${rest%%/*}"
    seed_file="${rest#*/}"
    seed="${seed_file#seed_}"
    seed="${seed%.txt}"
    collect_batch "$label" "$policy" "$seed" "$role" "$key_file"
  done < <(find "$BATCH_ROOT/${label,,}" -type f -name 'seed_*.txt' | sort)
  cleanup
  trap - EXIT
done

set +e
"$VLA_PY" scripts/audit_r8a1_rep3_pilot.py \
  --manifest "$MANIFEST" --repeat-root "$OUT" --output "$AUDIT"
audit_code=$?
set -e
if [[ "$audit_code" -eq 0 ]]; then
  printf 'complete\n' > "$OUT/COMPLETE"
elif [[ "$audit_code" -eq 3 ]]; then
  printf 'requires_k5_expansion\n' > "$OUT/REQUIRES_K5_EXPANSION"
else
  echo "R8A1 audit failed with exit code $audit_code" >&2
fi
exit "$audit_code"

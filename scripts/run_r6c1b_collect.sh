#!/usr/bin/env bash
# R6-C.1B-4: targeted OFT-labelled collection for the early-window selector.
# Collects t={0,8,16} persistent-OFT labels on:
#   - natural_development_eval states (from the frozen R6-C.1B manifest) for the
#     new seeds (Pi0.5 seed 2-3, Pi0Fast seed 1);
#   - train_enrichment states that the source-only screening kept (hard cases:
#     source failure / difficult success) for the same new seeds.
# Reproducibility: every new triple is collected twice (--rollout-index 0 and 1)
# so audit_r6c1b_repro.py can classify it (reproducible / step-diff / flip).
set -euo pipefail
cd /root/autodl-tmp/RASE

OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
INITIAL_KEYS=runs/pre_c0_r6/r6c1b_initial_keys_v1.json
SCREEN_ROOT=runs/pre_c0_r6/r6c1b_screen_v1
SELECTION=runs/pre_c0_r6/r6c1b_oft_selection_v2.json
SCREEN_AUDIT="$SCREEN_ROOT/screening_go_no_go.json"
APPROVAL="$SCREEN_ROOT/APPROVE_OFT_LABEL_COLLECTION"
OUT=runs/pre_c0_r6/r6c1b_collect_v1
mkdir -p "$OUT"

if [[ ! -f "$APPROVAL" ]]; then
  echo "R6C1B_COLLECT STOP: missing explicit approval marker $APPROVAL" >&2
  exit 20
fi
"$VLA_PY" - "$SELECTION" "$INITIAL_KEYS" "$SCREEN_AUDIT" <<'PY'
import hashlib, json, sys
from pathlib import Path
selection_path, initial_path, audit_path = map(Path, sys.argv[1:])
selection = json.load(open(selection_path))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if selection.get("status") != "frozen":
    raise SystemExit("selection manifest is not frozen")
if selection.get("initial_keys_sha256") != digest(initial_path):
    raise SystemExit("selection/initial-keys hash mismatch")
if selection.get("screen_audit_sha256") != digest(audit_path):
    raise SystemExit("selection/screen-audit hash mismatch")
print(json.dumps({
    "selection": str(selection_path),
    "selection_sha256": digest(selection_path),
    "initial_keys_sha256": digest(initial_path),
    "screen_audit_sha256": digest(audit_path),
}, sort_keys=True))
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
# States to collect with OFT labels.
#   natural_development_eval: all manifest states (natural distribution);
#   train_enrichment: screening-kept hard cases only (source failure / difficult
#   success), i.e. states whose source-only screening outcome is a failure or a
#   boundary where the source was still running at the last recorded boundary.
# Usage: suite_keys <suite_label> <role> <policy_id> [--hard-only]
suite_keys() {
  "$VLA_PY" - "$INITIAL_KEYS" "$SELECTION" "$1" "$2" "$3" "$4" <<'PY'
import json, sys, glob
from pathlib import Path
initial = json.load(open(sys.argv[1]))
selection = json.load(open(sys.argv[2]))
suite_label = sys.argv[3]
role = sys.argv[4]
policy_id = sys.argv[5]
hard_only = len(sys.argv) > 6 and sys.argv[6] == "--hard-only"
# Map (suite, role) -> candidate state_keys from the frozen manifest.
records = [row for row in initial["records"]
           if row["suite"] == suite_label and row["role"] == role]
candidates = {row["state_key"] for row in records}
if not hard_only:
    for row in records:
        print(row["state_key"])
    raise SystemExit(0)
if selection.get("status") != "frozen":
    raise SystemExit("OFT selection manifest is not frozen")
selected = set(selection["policies"][policy_id]["selected_enrichment_state_keys"])
for key in sorted(candidates & selected):
    print(key)
PY
}
collect() {
  local label="$1" policy="$2" seed="$3" role="$4" rep="$5" key_file="${6:-}"
  local suite
  suite="$(actual_suite "$label")"
  local hard=""
  if [[ "$role" == train_enrichment ]]; then hard="--hard-only"; fi
  local -a args=()
  if [[ -n "$key_file" ]]; then
    while read -r key; do
      [[ -n "$key" ]] && args+=(--state-key "$key")
    done < "$key_file"
  else
    while read -r key; do
      [[ -n "$key" ]] && args+=(--state-key "$key")
    done < <(suite_keys "$label" "$role" "$policy" "$hard")
  fi
  local n_states=$(( ${#args[@]} / 2 ))
  if [[ "$n_states" -eq 0 ]]; then
    echo "R6C1B_COLLECT skip $policy/$label/$role/rep$rep: no states" >&2
    return 0
  fi
  local output="$OUT/suite_${label,,}/$policy/$role/seed_$seed/rep$rep"
  # Idempotent resume: skip a batch if every requested state already produced a
  # metadata file for this (policy, seed, rep) with OFT labels.
  local all_done=1 missing=0
  for key in "${args[@]}"; do
    [[ "$key" == --state-key ]] && continue
    local meta="$output/${key}__seed${seed}.json"
    if [[ "$rep" != 0 ]]; then meta="$output/${key}__seed${seed}__rep${rep}.json"; fi
    if [[ ! -f "$meta" ]]; then all_done=0; missing=$((missing + 1)); fi
  done
  if [[ "$all_done" -eq 1 ]]; then
    echo "R6C1B_COLLECT skip $policy/$label/$role/rep$rep: all $n_states states already collected" >&2
    return 0
  fi
  echo "R6C1B_COLLECT resume $policy/$label/$role/rep$rep: $missing/$n_states states missing" >&2
  local -a policy_args
  if [[ "$policy" == pi0fast_libero ]]; then
    policy_args=(--policy-path ckpts/pi0fast_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  else
    policy_args=(--policy-path ckpts/pi05_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  fi
  echo "R6C1B_COLLECT suite=$label policy=$policy seed=$seed role=$role rep=$rep states=$n_states"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys "$INITIAL_KEYS" \
    --policy-id "$policy" "${policy_args[@]}" --suite "$suite" --seed-index "$seed" \
    --rollout-index "$rep" \
    --endpoint tcp://127.0.0.1:5555 --output-dir "$output" \
    --boundary 0 --boundary 8 --boundary 16 \
    "${args[@]}" \
    --bookkeeping-mode full
}

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/suite_${label,,}/oft_server.log"
  mkdir -p "$(dirname "$server_log")"
  echo "R6C1B_COLLECT starting OFT suite=$label checkpoint=$ckpt"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 60); do
    if "$VLA_PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite "$suite" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    echo "R6C1B_COLLECT ERROR: OFT server did not become ready for suite=$label" >&2
    tail -100 "$server_log" >&2 || true
    exit 1
  fi
  echo "R6C1B_COLLECT OFT ready suite=$label pid=$server_pid"
  # Natural eval cohort: new seeds on the new eval states + existing states.
  for seed in 2 3; do
    collect "$label" pi05_libero "$seed" natural_development_eval 0
    collect "$label" pi05_libero "$seed" natural_development_eval 1
  done
  collect "$label" pi0fast_libero 1 natural_development_eval 0
  collect "$label" pi0fast_libero 1 natural_development_eval 1
  # Train enrichment cohort (screening-kept states).
  for seed in 2 3; do
    collect "$label" pi05_libero "$seed" train_enrichment 0
    collect "$label" pi05_libero "$seed" train_enrichment 1
  done
  collect "$label" pi0fast_libero 1 train_enrichment 0
  collect "$label" pi0fast_libero 1 train_enrichment 1
  cleanup
  trap - EXIT
  echo "R6C1B_COLLECT completed suite=$label"
done

# Reproducibility audit + extended exclusion manifest.
"$VLA_PY" scripts/audit_r6c1b_repro.py \
  --input-root "$OUT" \
  --atlas-root runs/pre_c0_r6/policy_pair_atlas_v1 \
  --base-exclusions runs/pre_c0_r6/r6b1_b12_exclusions_v1.json \
  --exclusions-output runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json

# A disagreement between the two exact-seed replicas triggers one targeted
# third rollout.  Do not collect rep2 for stable triples.
if [[ "$(python3 -c 'import json; print(json.load(open("runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json"))["status"])')" == "incomplete_needs_third" ]]; then
  REP2_KEYS="$(mktemp -d "$OUT/rep2_keys.XXXXXX")"
  "$VLA_PY" - runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json "$REP2_KEYS" <<'PY'
import json, sys
from pathlib import Path
audit = json.load(open(sys.argv[1]))
root = Path(sys.argv[2])
for item in audit.get("needs_third", []):
    policy, seed, state = item["key"]
    parts = Path(item["replicas"][0]["path"]).parts
    suite = next(part.removeprefix("suite_") for part in parts if part.startswith("suite_"))
    role = "train_enrichment" if "train_enrichment" in parts else "natural_development_eval"
    path = root / suite / policy / role / f"seed_{seed}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(state + "\n")
PY
  for label in Spatial Object Goal Long; do
    label_lower="${label,,}"
    [[ -d "$REP2_KEYS/$label_lower" ]] || continue
    suite="$(actual_suite "$label")"
    ckpt="$(checkpoint "$label")"
    server_log="$OUT/suite_${label_lower}/oft_server_rep2.log"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
    RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
    "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
    server_pid=$!
    cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
    trap cleanup EXIT
    ready=0
    for _ in $(seq 1 60); do
      if "$VLA_PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite "$suite" >/dev/null 2>&1; then ready=1; break; fi
      sleep 5
    done
    [[ "$ready" == 1 ]] || { echo "rep2 OFT server failed for $label" >&2; exit 1; }
    while IFS= read -r key_file; do
      rel="${key_file#"$REP2_KEYS/$label_lower/"}"
      policy="${rel%%/*}"; rest="${rel#*/}"
      role="${rest%%/*}"; seed_file="${rest#*/}"
      seed="${seed_file#seed_}"; seed="${seed%.txt}"
      collect "$label" "$policy" "$seed" "$role" 2 "$key_file"
    done < <(find "$REP2_KEYS/$label_lower" -type f -name 'seed_*.txt' | sort)
    cleanup
    trap - EXIT
  done
  "$VLA_PY" scripts/audit_r6c1b_repro.py \
    --input-root "$OUT" \
    --atlas-root runs/pre_c0_r6/policy_pair_atlas_v1 \
    --base-exclusions runs/pre_c0_r6/r6b1_b12_exclusions_v1.json \
    --exclusions-output runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json
fi
if [[ "$(python3 -c 'import json; print(json.load(open("runs/pre_c0_r6/r6c1b_repro_exclusions_v1.json"))["status"])')" != "frozen" ]]; then
  echo "R6C1B_COLLECT ERROR: reproducibility audit is not frozen" >&2
  exit 21
fi
echo complete > "$OUT/COMPLETE"

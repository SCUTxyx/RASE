#!/usr/bin/env bash
# PRE-C1.2 pipeline (R0 pivot):
#   E0 → E1 → freeze H → DAgger R1 → global QC → R0 diagnostics → branch decision
# Legacy E3/E4 full OFT-action BC is paused and requires explicit unlock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
KEYS="${KEYS:-artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
LOG_DIR="${LOG_DIR:-runs/rase_pre_c1_2_pipeline_logs}"
SUCC_OUT="${SUCC_OUT:-runs/rase_pre_c1_2_successor_v1.json}"
HORIZON_OUT="${HORIZON_OUT:-runs/rase_pre_c1_2_horizon_sweep_v1.json}"
DAGGER_OUT="${DAGGER_OUT:-runs/rase_pre_c1_2_dagger_r1_v1}"
ORIGINAL="${ORIGINAL:-runs/rase_pre_c1_1_distill_dataset_v1.jsonl}"
# Round1 practical seeds (protocol min 5); override with SEEDS_PER_ANCHOR=
SEEDS_PER_ANCHOR="${SEEDS_PER_ANCHOR:-5}"
MAX_STUDENT_STEPS="${MAX_STUDENT_STEPS:-80}"
# Default: stop after R1 dataset + global QC + optional R0. Never auto-run legacy E3/E4.
RUN_R0_AFTER_DAGGER="${RUN_R0_AFTER_DAGGER:-1}"
ALLOW_LEGACY_E3_E4="${ALLOW_LEGACY_E3_E4:-0}"

mkdir -p "$LOG_DIR" runs artifacts/pre_c1 progress
exec > >(tee -a "$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "=== PRE-C1.2 PIPELINE START $(date -Is) ==="
echo "GPU=$(nvidia-smi -L | head -1)"

kill_oft() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

start_oft() {
  local short="$1"
  local suite="$2"
  local ckpt="$3"
  kill_oft
  echo "=== Start OFT server suite=${suite} ckpt=${ckpt} ==="
  (
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "$LOG_DIR/oft_server_${short}.log" 2>&1
  ) &
  local ready=0
  for _ in $(seq 1 150); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server not ready for $suite" >&2
    tail -80 "$LOG_DIR/oft_server_${short}.log" >&2 || true
    exit 1
  fi
  echo "OFT ready suite=$suite"
}

SUITE_SHORTS=(spatial object goal 10)
SUITE_LABELS=(Spatial Object Goal Long)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

########################################
# E0: successor test (suite-serial OFT)
########################################
echo "=== E0 successor $(date -Is) ==="
SKIP_E0="${SKIP_E0:-0}"
need_e0=1
if [[ "$SKIP_E0" == "1" && -f "$SUCC_OUT" ]]; then
  need_e0="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$SUCC_OUT").read_text())
keys=set(json.loads(Path("$KEYS").read_text())["state_keys"])
have={str(r["state_key"]) for r in d.get("results") or []}
print(0 if keys.issubset(have) and not d.get("block_training") else 1)
PY
)"
fi
if [[ "$need_e0" == "0" ]]; then
  echo "SKIP E0: complete successor results already present"
else
  if [[ ! -f "$SUCC_OUT" ]]; then
    echo '{"schema_version":"rase-pre-c1-2-successor/v1","results":[]}' > "$SUCC_OUT"
  fi
  for idx in "${!SUITE_SHORTS[@]}"; do
    short="${SUITE_SHORTS[$idx]}"
    label="${SUITE_LABELS[$idx]}"
    suite="${SUITE_NAMES[$idx]}"
    ckpt="${CKPTS[$idx]}"
    n_keys="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$KEYS").read_text())
print(len(d.get("by_suite",{}).get("$label",[])))
PY
)"
    if [[ "$n_keys" == "0" ]]; then
      echo "SKIP successor suite=$label (0 locked keys)"
      continue
    fi
    # Skip suite if all its locked keys already present.
    missing="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$SUCC_OUT").read_text())
have={str(r["state_key"]) for r in d.get("results") or []}
need=set(json.loads(Path("$KEYS").read_text()).get("by_suite",{}).get("$label",[]))
print(len(need-have))
PY
)"
    if [[ "$missing" == "0" ]]; then
      echo "SKIP successor suite=$label (already complete)"
      continue
    fi
    start_oft "$short" "$suite" "$ckpt"
    "$PY" scripts/eval_pre_c1_2_successor_test.py \
      --protocol-lock "$PROTOCOL" \
      --config "$CONFIG" \
      --adapter-dir "$ADAPTER" \
      --failure-rollout-dir "$FAILURES" \
      --state-keys-json "$KEYS" \
      --suite "$label" \
      --endpoint "$ENDPOINT" \
      --output "$LOG_DIR/successor_${short}.json" \
      --merge-into "$SUCC_OUT"
  done
  kill_oft
fi
"$PY" - <<'PY'
import json, sys
from pathlib import Path
from rase.adapt.pre_c1_2 import interface_mismatch_decision
p=Path("runs/rase_pre_c1_2_successor_v1.json")
d=json.loads(p.read_text())
# Re-score with current decision rule (robust floor).
mismatches=[]
for r in d.get("results") or []:
    mae=(r.get("actions") or {}).get("env_action_mae")
    if mae is None:
        mae=(r.get("interface_decision") or {}).get("env_action_mae") or 0.0
    dec=interface_mismatch_decision(
        env_action_mae=float(mae),
        cross_successor_error=float((r.get("cross") or {}).get("aggregate_l2") or 0.0),
        sim_floor_error=float((r.get("sim_floor") or {}).get("aggregate_l2") or 0.0),
        student_repeat_error=float((r.get("student_repeat") or {}).get("aggregate_l2") or 0.0),
    )
    r["interface_decision"]=dec
    if dec["interface_mismatch"]:
        mismatches.append(r["state_key"])
d["n_states"]=len(d.get("results") or [])
d["n_interface_mismatch"]=len(mismatches)
d["block_training"]=bool(mismatches)
d["decision"]="fix_interface" if mismatches else "proceed"
p.write_text(json.dumps(d, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print("E0_SUMMARY", json.dumps({k:d[k] for k in d if k!="results"}, sort_keys=True))
if d.get("block_training"):
    print("E0_BLOCK: interface mismatch — stop pipeline", file=sys.stderr)
    sys.exit(3)
print("E0_PROCEED")
PY

########################################
# E1: same-H horizon sweep + freeze
########################################
echo "=== E1 horizon sweep $(date -Is) ==="
if [[ "${SKIP_E1:-0}" == "1" && -f "$HORIZON_OUT" ]] && grep -q '"sweep_valid": true' "$HORIZON_OUT"; then
  echo "SKIP E1: horizon sweep already valid; ensuring horizon frozen"
  "$PY" scripts/freeze_pre_c1_2_horizon.py --protocol-lock "$PROTOCOL" --sweep-json "$HORIZON_OUT"
else
  "$PY" scripts/eval_pre_c1_2_horizon_sweep.py \
    --protocol-lock "$PROTOCOL" \
    --config "$CONFIG" \
    --adapter-dir "$ADAPTER" \
    --failure-rollout-dir "$FAILURES" \
    --state-keys-json "$KEYS" \
    --output "$HORIZON_OUT" \
    --freeze-protocol
fi

########################################
# Phase 2: DAgger Round 1 (suite-serial)
########################################
echo "=== Phase2 DAgger Round1 $(date -Is) ==="
mkdir -p "$DAGGER_OUT"
for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  label="${SUITE_LABELS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"
  n_keys="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$KEYS").read_text())
print(len(d.get("by_suite",{}).get("$label",[])))
PY
)"
  if [[ "$n_keys" == "0" ]]; then
    echo "SKIP dagger suite=$label"
    continue
  fi
  start_oft "$short" "$suite" "$ckpt"
  "$PY" scripts/collect_pre_c1_2_student_state_oft_relabel.py \
    --protocol-lock "$PROTOCOL" \
    --config "$CONFIG" \
    --adapter-dir "$ADAPTER" \
    --failure-rollout-dir "$FAILURES" \
    --state-keys-json "$KEYS" \
    --suite "$label" \
    --endpoint "$ENDPOINT" \
    --round-id 1 \
    --output-dir "$DAGGER_OUT" \
    --seeds-per-anchor "$SEEDS_PER_ANCHOR" \
    --max-student-steps "$MAX_STUDENT_STEPS" \
    --resume
done
kill_oft

"$PY" scripts/build_pre_c1_2_dagger_dataset.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$DAGGER_OUT" \
  --original-dataset-jsonl "$ORIGINAL" \
  --output-jsonl runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  --splits-output runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  --qc-json artifacts/pre_c1/pre_c1_2_dataset_qc_r1.json

"$PY" scripts/analyze_pre_c1_2_dagger_global_qc.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$DAGGER_OUT" \
  --state-keys-json "$KEYS" \
  --dataset-jsonl runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  --splits-json runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  --output artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json \
  --progress-md progress/2026-08-05_pre_c1_2_dagger_r1_global_qc.md

########################################
# Hard stop before legacy E3/E4 (R0 pivot)
########################################
echo "=== HARD STOP before legacy E3/E4 $(date -Is) ==="
cat > runs/rase_pre_c1_2_pipeline_hard_stop.json <<EOF
{
  "schema_version": "rase-pre-c1-2-pipeline-hard-stop/v1",
  "stopped_after": "dagger_r1_dataset_and_global_qc",
  "legacy_e3_e4_paused": true,
  "next": "scripts/run_pre_c1_2_r0.sh then branch decision",
  "allow_legacy_e3_e4": ${ALLOW_LEGACY_E3_E4}
}
EOF

if [[ "$RUN_R0_AFTER_DAGGER" == "1" ]]; then
  echo "=== R0 diagnostics $(date -Is) ==="
  PROTOCOL="$PROTOCOL" CONFIG="$CONFIG" ADAPTER="$ADAPTER" \
  KEYS="$KEYS" FAILURES="$FAILURES" ENDPOINT="$ENDPOINT" \
  bash scripts/run_pre_c1_2_r0.sh
else
  echo "SKIP R0: RUN_R0_AFTER_DAGGER=0; run scripts/run_pre_c1_2_r0.sh manually"
fi

if [[ "$ALLOW_LEGACY_E3_E4" == "1" ]]; then
  echo "=== LEGACY E3/E4 unlocked explicitly $(date -Is) ==="
  STAGE=e3 \
  DATASET=runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  SPLITS=runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  PROTOCOL="$PROTOCOL" CONFIG="$CONFIG" \
  ALLOW_LEGACY_E3_E4=1 \
  bash scripts/run_pre_c1_2_train_eval.sh

  STAGE=e4 \
  DATASET=runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  SPLITS=runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  PROTOCOL="$PROTOCOL" CONFIG="$CONFIG" \
  ALLOW_LEGACY_E3_E4=1 \
  bash scripts/run_pre_c1_2_train_eval.sh

  echo "=== Capacity ladder configs (legacy; only if E4 fails) $(date -Is) ==="
  ALLOW_LEGACY_E3_E4=1 "$PY" scripts/run_pre_c1_2_capacity_ladder.py --allow-legacy
else
  echo "LEGACY_E3_E4_SKIPPED: paused pending R0 branch decision"
  # Even if an older in-memory parent reaches here later, train_eval.sh also blocks.
fi

echo "=== PRE-C1.2 PIPELINE STOPPED AFTER R1/R0 GATE $(date -Is) ==="
echo "Successor: $SUCC_OUT"
echo "Horizon:   $HORIZON_OUT"
echo "DAgger:    $DAGGER_OUT"
echo "Global QC: artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json"
echo "R0 decision: runs/rase_pre_c1_2_r0_decision_v1.json"
echo "Legacy E3/E4: paused (ALLOW_LEGACY_E3_E4=${ALLOW_LEGACY_E3_E4})"

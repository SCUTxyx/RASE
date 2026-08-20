#!/usr/bin/env bash
# Continue after in-flight DAgger R1 (old parent may die on legacy E3 block).
# Builds/refreshes dataset + global QC + R0. Does not run legacy E3/E4.
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
KEYS="${KEYS:-artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json}"
DAGGER_OUT="${DAGGER_OUT:-runs/rase_pre_c1_2_dagger_r1_v1}"
ORIGINAL="${ORIGINAL:-runs/rase_pre_c1_1_distill_dataset_v1.jsonl}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
WAIT_FOR_COMPLETE="${WAIT_FOR_COMPLETE:-1}"
RUN_REVISED_TRAIN="${RUN_REVISED_TRAIN:-1}"
SMOKE="${SMOKE:-0}"

mkdir -p runs artifacts/pre_c1 progress

wait_dagger_complete() {
  echo "Waiting for DAgger R1 coverage of all locked anchors..."
  while true; do
    ready="$("$PY" - <<'PY'
import json
from pathlib import Path
keys=set(json.loads(Path("artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json").read_text())["state_keys"])
root=Path("runs/rase_pre_c1_2_dagger_r1_v1")
have=set()
seed_counts={}
for p in root.glob("*.json"):
    if p.name in {"dagger_qc.json"} or p.name.endswith("_qc.json"):
        continue
    d=json.loads(p.read_text())
    if d.get("schema_version")!="rase-pre-c1-2-dagger-run/v1":
        continue
    a=str(d.get("anchor_id"))
    have.add(a)
    seed_counts[a]=seed_counts.get(a,0)+1
missing=sorted(keys-have)
short=[a for a in keys if seed_counts.get(a,0)<5]
print("ready" if (not missing and not short) else f"wait missing={len(missing)} short={len(short)} covered={len(have)}/{len(keys)}")
# also wait until collect process gone if still short
PY
)"
    echo "$(date -Is) dagger_status=$ready"
    if [[ "$ready" == "ready" ]]; then
      # ensure collector not still writing
      if pgrep -f 'collect_pre_c1_2_student_state_oft_relabel.py' >/dev/null; then
        echo "anchors complete but collector still running; waiting..."
        sleep 30
        continue
      fi
      break
    fi
    sleep 60
  done
}

if [[ "$WAIT_FOR_COMPLETE" == "1" ]]; then
  wait_dagger_complete
fi

echo "=== Build / refresh R1 dataset $(date -Is) ==="
"$PY" scripts/build_pre_c1_2_dagger_dataset.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$DAGGER_OUT" \
  --original-dataset-jsonl "$ORIGINAL" \
  --output-jsonl runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  --splits-output runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  --qc-json artifacts/pre_c1/pre_c1_2_dataset_qc_r1.json

echo "=== Global QC $(date -Is) ==="
"$PY" scripts/analyze_pre_c1_2_dagger_global_qc.py \
  --protocol-lock "$PROTOCOL" \
  --dagger-dir "$DAGGER_OUT" \
  --state-keys-json "$KEYS" \
  --dataset-jsonl runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl \
  --splits-json runs/rase_pre_c1_2_distill_dataset_r1_v1.benchmark-splits.json \
  --output artifacts/pre_c1/pre_c1_2_dagger_global_qc_r1.json \
  --progress-md progress/2026-08-05_pre_c1_2_dagger_r1_global_qc.md

cat > runs/rase_pre_c1_2_pipeline_hard_stop.json <<EOF
{
  "schema_version": "rase-pre-c1-2-pipeline-hard-stop/v1",
  "stopped_after": "dagger_r1_dataset_and_global_qc",
  "legacy_e3_e4_paused": true,
  "next": "scripts/run_pre_c1_2_r0.sh then branch decision"
}
EOF

echo "=== R0 $(date -Is) ==="
SMOKE="$SMOKE" \
PROTOCOL="$PROTOCOL" CONFIG="$CONFIG" ADAPTER="$ADAPTER" \
KEYS="$KEYS" FAILURES="$FAILURES" ENDPOINT="$ENDPOINT" \
bash scripts/run_pre_c1_2_r0.sh

if [[ "$RUN_REVISED_TRAIN" == "1" ]]; then
  echo "=== Revised train (R0-gated) $(date -Is) ==="
  SMOKE="$SMOKE" bash scripts/run_pre_c1_2_revised_train.sh
fi

echo "=== AFTER_DAGGER_R0_DONE $(date -Is) ==="

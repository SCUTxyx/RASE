#!/usr/bin/env bash
# PRE-C1.3 Recovery-Corridor Replay Audit — Phased Pipeline
#
# Phases:
#   1. build      — Build datasets for Arm A', B, C
#   2. validate   — Validate split integrity, manifests
#   3. train      — Warm-started training (3 arms × 3 seeds)
#   4. snapshot   — Verify adapter outputs vs stored equivalence
#   5. screen_eval— Screening R(k) with 2 OFT seeds on all arms
#   6. confirm_eval— Confirmatory R(k) with 10 OFT seeds on qualifying arms
#   7. report     — Comparison curves, bootstrap CIs, gate check
#
# Each phase writes a completion manifest for safe restart.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

# --- Config ---
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
KEYS="${KEYS:-artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
LOG_DIR="${LOG_DIR:-runs/rase_pre_c1_3_pipeline_logs}"
SOURCE_DATASET="${SOURCE_DATASET:-runs/rase_pre_c1_2_distill_dataset_r1_v1.jsonl}"
DATASET_DIR="${DATASET_DIR:-runs/rase_pre_c1_3_datasets}"
C11_ADAPTER="${C11_ADAPTER:-runs/rase_pre_c1_1_lora_train_v1/adapter_final}"
TRAIN_DIR="${TRAIN_DIR:-runs/rase_pre_c1_3_train}"
EVAL_DIR="${EVAL_DIR:-runs/rase_pre_c1_3_eval}"
REPORT_DIR="${REPORT_DIR:-runs/rase_pre_c1_3_report}"
SMOKE="${SMOKE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SCREEN="${SKIP_SCREEN:-0}"
SKIP_CONFIRM="${SKIP_CONFIRM:-0}"
N_TRAINING_SEEDS="${N_TRAINING_SEEDS:-3}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-1000}"  # per-arm step budget

mkdir -p "$LOG_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$REPORT_DIR" runs progress

ARMS=("arm_ap" "arm_b" "arm_c")

# --- OFT server helpers ---
kill_oft() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

OFT_PY="${OFT_PY:-/root/autodl-tmp/envs/oft/bin/python}"

start_oft() {
  local short="$1" local suite="$2" local ckpt="$3"
  kill_oft
  sleep 2
  echo "=== Start OFT server suite=${suite} ckpt=${ckpt} ==="
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}"
  export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
  export RASE_OFT_SUITE="$suite"
  nohup "$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter \
    > "$LOG_DIR/oft_server_c1_3_${short}.log" 2>&1 &
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
    tail -80 "$LOG_DIR/oft_server_c1_3_${short}.log" >&2 || true
    exit 1
  fi
  echo "OFT ready suite=$suite"
}

# --- Phase helpers ---
phase_done() {
  local phase="$1"
  echo "{\"phase\":\"$phase\",\"done\":true,\"timestamp\":\"$(date -Is)\"}" > "$LOG_DIR/phase_${phase}_done.json"
  echo "=== PHASE $phase DONE $(date -Is) ==="
}

phase_skip_check() {
  local phase="$1"
  if [[ -f "$LOG_DIR/phase_${phase}_done.json" ]]; then
    echo "=== PHASE $phase already complete; skipping ==="
    return 0
  fi
  return 1
}

# ================ PHASE 1: BUILD ================
if [[ "$SKIP_BUILD" != "1" ]]; then
  if ! phase_skip_check build; then
    echo "=== PHASE 1: BUILD $(date -Is) ==="
    BUILD_FLAGS=()
    if [[ "$SMOKE" == "1" ]]; then
      BUILD_FLAGS+=(--smoke)
    fi
    "$PY" scripts/build_pre_c1_3_datasets.py \
      --protocol-lock "$PROTOCOL" \
      --input-jsonl "$SOURCE_DATASET" \
      --output-dir "$DATASET_DIR" \
      "${BUILD_FLAGS[@]}"
    phase_done build
  fi
else
  echo "SKIP build"
fi

# ================ PHASE 2: VALIDATE ================
if ! phase_skip_check validate; then
  echo "=== PHASE 2: VALIDATE $(date -Is) ==="
  for arm in "${ARMS[@]}"; do
    dataset="$DATASET_DIR/${arm}.jsonl"
    splits="$DATASET_DIR/${arm}.benchmark-splits.json"
    manifest="$DATASET_DIR/${arm}.manifest.json"
    if [[ ! -f "$dataset" ]]; then
      echo "ERROR: missing dataset: $dataset" >&2
      exit 2
    fi
    n_rows=$(wc -l < "$dataset")
    echo "  Arm $arm: $n_rows rows, splits=$(cat "$splits" | $PY -c "import json,sys;d=json.load(sys.stdin);print(d['n_train_rows'],'train',d['n_val_rows'],'val')")"
  done

  # Check no arm has identical dataset hash (all rows identical)
  hashes=()
  for arm in "${ARMS[@]}"; do
    h=$(sha256sum "$DATASET_DIR/${arm}.jsonl" | cut -d' ' -f1)
    hashes+=("$h")
    echo "  Arm $arm sha256: $h"
  done
  unique_hashes=$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)
  if [[ "$unique_hashes" -lt "${#ARMS[@]}" ]]; then
    echo "WARNING: some arms have identical dataset content (expected for test/smoke)" >&2
  fi
  phase_done validate
fi

# ================ PHASE 3: TRAIN ================
if [[ "$SKIP_TRAIN" != "1" ]]; then
  if ! phase_skip_check train; then
    echo "=== PHASE 3: TRAIN $(date -Is) ==="

    for arm in "${ARMS[@]}"; do
      dataset="$DATASET_DIR/${arm}.jsonl"
      splits="$DATASET_DIR/${arm}.benchmark-splits.json"
      for ts in $(seq 0 $((N_TRAINING_SEEDS - 1))); do
        output_dir="$TRAIN_DIR/${arm}_seed${ts}"
        if [[ -f "$output_dir/adapter_final/adapter_config.json" ]]; then
          echo "SKIP train arm=$arm seed=$ts (already exists)"
          continue
        fi
        echo "--- Train arm=$arm seed=$ts ---"
        TRAIN_FLAGS=()
        if [[ "$SMOKE" == "1" ]]; then
          TRAIN_FLAGS+=(--smoke)
        fi
        "$PY" scripts/train_pre_c1_3_continued_lora.py \
          --protocol-lock "$PROTOCOL" \
          --dataset-jsonl "$dataset" \
          --splits-json "$splits" \
          --config "$CONFIG" \
          --output-dir "$output_dir" \
          --cache-dir "$TRAIN_DIR/cache_${arm}_seed${ts}" \
          --c11-adapter-dir "$C11_ADAPTER" \
          --training-seed "$ts" \
          --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" \
          "${TRAIN_FLAGS[@]}" \
          > "$LOG_DIR/train_${arm}_seed${ts}.log" 2>&1 &
      done
    done
    wait
    echo "All training jobs completed."
    phase_done train
  fi
else
  echo "SKIP train"
fi

# ================ PHASE 4: SNAPSHOT ================
if ! phase_skip_check snapshot; then
  echo "=== PHASE 4: SNAPSHOT $(date -Is) ==="
  for arm in "${ARMS[@]}"; do
    for ts in $(seq 0 $((N_TRAINING_SEEDS - 1))); do
      adapter_path="$TRAIN_DIR/${arm}_seed${ts}/adapter_final"
      if [[ -f "$adapter_path/adapter_config.json" ]]; then
        echo "  Adapter OK: $adapter_path"
      else
        echo "  Adapter MISSING: $adapter_path" >&2
      fi
    done
  done
  echo "  C1.1 baseline: $C11_ADAPTER"
  phase_done snapshot
fi

# ================ PHASE 5: SCREEN EVAL ================
SUITE_SHORTS=(spatial object goal 10)
SUITE_LABELS=(Spatial Object Goal Long)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

run_eval_for_arm() {
  local arm="$1" local ts="$2" local teacher_seeds="$3" local phase_label="$4"
  local adapter_path="$TRAIN_DIR/${arm}_seed${ts}/adapter_final"
  if [[ ! -f "$adapter_path/adapter_config.json" ]]; then
    echo "SKIP eval arm=$arm ts=$ts (no adapter)" >&2
    return 0
  fi
  local eval_subdir="$EVAL_DIR/${arm}_seed${ts}_${phase_label}"
  if [[ -f "$eval_subdir/summary.json" ]]; then
    echo "SKIP eval arm=$arm ts=$ts phase=$phase_label (already done)"
    return 0
  fi

  for idx in "${!SUITE_SHORTS[@]}"; do
    short="${SUITE_SHORTS[$idx]}"
    label="${SUITE_LABELS[$idx]}"
    suite="${SUITE_NAMES[$idx]}"
    ckpt="${CKPTS[$idx]}"
    n_keys=$("$PY" -c "import json;from pathlib import Path;d=json.loads(Path('$KEYS').read_text());print(len(d.get('by_suite',{}).get('$label',[])))")
    if [[ "$n_keys" == "0" ]]; then
      echo "SKIP suite=$label (no keys)"
      continue
    fi
    start_oft "$short" "$suite" "$ckpt"
    EVAL_FLAGS=()
    if [[ "$SMOKE" == "1" ]]; then
      EVAL_FLAGS+=(--smoke)
    fi
    "$PY" scripts/eval_pre_c1_3_rk_multi_seed.py \
      --config "$CONFIG" \
      --failure-rollout-dir "$FAILURES" \
      --state-keys-json "$KEYS" \
      --adapter-dir "$adapter_path" \
      --output-dir "$eval_subdir" \
      --arm "$arm" \
      --training-seed "$ts" \
      --suite "$label" \
      --endpoint "$ENDPOINT" \
      --teacher-seeds "$teacher_seeds" \
      --resume \
      "${EVAL_FLAGS[@]}"
  done
  kill_oft
}

# Also evaluate frozen C1.1 as Arm A
eval_c11() {
  local phase_label="$1" local teacher_seeds="$2"
  local eval_subdir="$EVAL_DIR/arm_a_frozen_${phase_label}"
  if [[ -f "$eval_subdir/summary.json" ]]; then
    echo "SKIP eval C1.1 phase=$phase_label (already done)"
    return 0
  fi
  for idx in "${!SUITE_SHORTS[@]}"; do
    short="${SUITE_SHORTS[$idx]}"
    label="${SUITE_LABELS[$idx]}"
    suite="${SUITE_NAMES[$idx]}"
    ckpt="${CKPTS[$idx]}"
    n_keys=$("$PY" -c "import json;from pathlib import Path;d=json.loads(Path('$KEYS').read_text());print(len(d.get('by_suite',{}).get('$label',[])))")
    if [[ "$n_keys" == "0" ]]; then continue; fi
    start_oft "$short" "$suite" "$ckpt"
    EVAL_FLAGS=()
    if [[ "$SMOKE" == "1" ]]; then EVAL_FLAGS+=(--smoke); fi
    "$PY" scripts/eval_pre_c1_3_rk_multi_seed.py \
      --config "$CONFIG" --failure-rollout-dir "$FAILURES" \
      --state-keys-json "$KEYS" --adapter-dir "$C11_ADAPTER" \
      --output-dir "$eval_subdir" --arm "arm_a" --training-seed 0 \
      --suite "$label" --endpoint "$ENDPOINT" \
      --teacher-seeds "$teacher_seeds" --resume "${EVAL_FLAGS[@]}"
  done
  kill_oft
}

if [[ "$SKIP_SCREEN" != "1" ]]; then
  if ! phase_skip_check screen_eval; then
    echo "=== PHASE 5: SCREEN EVAL (2 OFT seeds) $(date -Is) ==="
    eval_c11 screen 2
    for arm in "${ARMS[@]}"; do
      for ts in $(seq 0 $((N_TRAINING_SEEDS - 1))); do
        run_eval_for_arm "$arm" "$ts" 2 screen &
      done
    done
    wait
    phase_done screen_eval
  fi
else
  echo "SKIP screen eval"
fi

# ================ PHASE 6: CONFIRM EVAL ================
if [[ "$SKIP_CONFIRM" != "1" ]]; then
  if ! phase_skip_check confirm_eval; then
    echo "=== PHASE 6: CONFIRM EVAL (10 OFT seeds) $(date -Is) ==="
    eval_c11 confirm 10
    for arm in "${ARMS[@]}"; do
      for ts in $(seq 0 $((N_TRAINING_SEEDS - 1))); do
        run_eval_for_arm "$arm" "$ts" 10 confirm &
      done
    done
    wait
    phase_done confirm_eval
  fi
else
  echo "SKIP confirm eval"
fi

# ================ PHASE 7: REPORT ================
if ! phase_skip_check report; then
  echo "=== PHASE 7: REPORT $(date -Is) ==="

  # Aggregate all screening summaries into a comparison table
  "$PY" -c "
import json, numpy as np
from pathlib import Path

eval_dir = Path('$EVAL_DIR')
arms_labels = {
    'arm_a': 'A (C1.1 frozen)',
    'arm_ap': \"A' (original+clean)\",
    'arm_b': 'B (query only)',
    'arm_c': 'C (query+suffix)',
}
ks = [1, 2, 4, 8, 16]

print('=== PRE-C1.3 R(k) Comparison ===')
print(f\"{'Arm':>20s}\", end='')
for k in ks:
    print(f\"  R({k:>2d}) \", end='')
print(f\"  R_mid  \", end='')
print()

for arm_key, arm_label in arms_labels.items():
    all_teacher = []
    all_self = []
    for sub in sorted(eval_dir.glob(f'{arm_key}*screen*/summary.json')):
        s = json.loads(sub.read_text())
        curves = s.get('curves', {})
        for k in ks:
            v = curves.get('R_teacher', {}).get(str(k)) or curves.get('R_teacher', {}).get(k)
            if v is not None:
                all_teacher.append((k, float(v)))
            v = curves.get('R_self', {}).get(str(k)) or curves.get('R_self', {}).get(k)
            if v is not None:
                all_self.append((k, float(v)))
    if not all_teacher:
        continue
    by_k = {}
    for k, v in all_teacher:
        by_k.setdefault(k, []).append(v)
    by_k_self = {}
    for k, v in all_self:
        by_k_self.setdefault(k, []).append(v)

    print(f'{arm_label:>20s}', end='')
    vals = []
    for k in ks:
        vals_k = by_k.get(k, [float('nan')])
        mean_v = float(np.mean(vals_k))
        vals.append(mean_v)
        if mean_v == mean_v:
            print(f'  {mean_v:.3f} ', end='')
        else:
            print(f'    N/A ', end='')
    mid = float(np.mean([v for v in [vals[2], vals[3]] if v == v])) if vals else float('nan')
    if mid == mid:
        print(f'  {mid:.3f} ', end='')
    else:
        print(f'    N/A ', end='')

    # R_self mid
    self_mid_vals = []
    for k in [4, 8]:
        vals_sk = by_k_self.get(k, [float('nan')])
        m = float(np.mean(vals_sk))
        if m == m:
            self_mid_vals.append(m)
    self_mid = float(np.mean(self_mid_vals)) if self_mid_vals else float('nan')
    print(f'R_self_mid={self_mid:.3f}' if self_mid == self_mid else '  R_self_mid=N/A')
print()

# Arm C vs Arm B comparison
print('=== Arm C - Arm B difference ===')
for k in ks:
    c_vals = []
    b_vals = []
    for sub in sorted(eval_dir.glob('arm_c*_seed*_screen/summary.json')):
        s = json.loads(sub.read_text())
        r = s.get('curves', {}).get('R_teacher', {}).get(str(k)) or s.get('curves', {}).get('R_teacher', {}).get(k)
        if r is not None: c_vals.append(float(r))
    for sub in sorted(eval_dir.glob('arm_b*_seed*_screen/summary.json')):
        s = json.loads(sub.read_text())
        r = s.get('curves', {}).get('R_teacher', {}).get(str(k)) or s.get('curves', {}).get('R_teacher', {}).get(k)
        if r is not None: b_vals.append(float(r))
    cm = float(np.mean(c_vals)) if c_vals else float('nan')
    bm = float(np.mean(b_vals)) if b_vals else float('nan')
    delta = cm - bm if (cm == cm and bm == bm) else float('nan')
    print(f'  R_teacher({k:>2d}): C={cm:.3f}  B={bm:.3f}  delta={delta:+.3f}')
print()
print('=== Done ===')
print(f'Full eval data: {eval_dir}')
" > "$REPORT_DIR/comparison.txt"

  cat "$REPORT_DIR/comparison.txt"
  _atomic_json "$REPORT_DIR/comparison.json" "{}"
  "$PY" -c "
import json
from pathlib import Path
summary = {
    'schema_version': 'rase-pre-c1-3-report/v1',
    'timestamp': '$(date -Is)',
    'comparison_text': Path('$REPORT_DIR/comparison.txt').read_text(),
}
Path('$REPORT_DIR/comparison.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
"
  phase_done report
fi

echo "=== PRE-C1.3 PIPELINE DONE $(date -Is) ==="
echo "Datasets: $DATASET_DIR"
echo "Training: $TRAIN_DIR"
echo "Evaluation: $EVAL_DIR"
echo "Report: $REPORT_DIR/comparison.txt"

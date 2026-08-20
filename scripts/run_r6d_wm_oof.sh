#!/usr/bin/env bash
# R6-D pre-registered world-model ablation: re-run the R6-C candidate-arm OOF
# with --wm-features and Pareto-compare against the no-WM baseline per seed and
# VLA.  Gated: only run after the R6-C no-WM stage gate passes (>=4/5 seeds on
# both VLAs).  The WM feature cache must already exist (see
# scripts/cache_r6d_wm_features.py); this script never touches the V-JEPA encoder.
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-runs/pre_c0_r6/r6b1_b1p2_v1}"
DATASET="${DATASET:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz}"
DATASET_REPORT="${DATASET_REPORT:-$DATASET_ROOT/r6c_candidate_arm_dataset.npz.report.json}"
BASELINE_ROOT="${BASELINE_ROOT:-runs/pre_c0_r6/r6c_candidate_arm_oof_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r6/r6d_wm_oof_v1}"
WM_CACHE="${WM_CACHE:?set WM_CACHE to the R6-D wm_features.jsonl (or a dir of them)}"
SEEDS="${SEEDS:-10 11 12 13 14}"
FOLD_SEED="${FOLD_SEED:-20260810}"
EPOCHS="${EPOCHS:-60}"
AUROC_GAIN_MIN="${AUROC_GAIN_MIN:-0.02}"
DOMINANT_SEEDS_MIN="${DOMINANT_SEEDS_MIN:-4}"

if [[ ! -f "$DATASET_REPORT" ]]; then
  echo "R6D ERROR: candidate-arm dataset missing; run run_r6c_candidate_arm_oof.sh first" >&2
  exit 2
fi
if [[ ! -e "$WM_CACHE" ]]; then
  echo "R6D ERROR: WM cache '$WM_CACHE' does not exist" >&2
  exit 2
fi

# A single merged jsonl keeps the trainer's group+elapsed alignment simple.
if [[ -d "$WM_CACHE" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  WM_JSONL="$OUTPUT_ROOT/wm_features_merged.jsonl"
  : > "$WM_JSONL"
  find "$WM_CACHE" -name '*.jsonl' -print0 | sort -z | xargs -0 cat >> "$WM_JSONL"
else
  WM_JSONL="$WM_CACHE"
fi
if [[ ! -s "$WM_JSONL" ]]; then
  echo "R6D ERROR: merged WM cache is empty" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
: > "$OUTPUT_ROOT/status.txt"
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$output"
  for policy in pi0fast_libero pi05_libero; do
    baseline="$BASELINE_ROOT/seed_${seed}/per_vla_${policy}.json"
    if [[ ! -f "$baseline" ]]; then
      echo "R6D ERROR: baseline report missing: $baseline" >&2
      exit 2
    fi
    "$PYTHON_BIN" scripts/train_r6c_candidate_arm_student.py \
      --dataset "$DATASET" \
      --dataset-report "$DATASET_REPORT" \
      --output "$output/per_vla_${policy}.json" \
      --mode per_vla --target-policy "$policy" \
      --seed "$seed" --folds 5 --fold-seed "$FOLD_SEED" \
      --members 3 --epochs "$EPOCHS" \
      --dwell 2 --lcb-z 1.6448536269514722 \
      --wm-features "$WM_JSONL" \
      --device cuda \
      > "$output/per_vla_${policy}.log" 2>&1
    echo "DONE $seed per_vla $policy (WM)" | tee -a "$OUTPUT_ROOT/status.txt"
    # Per-seed Pareto comparison, pre-registered gate.
    "$PYTHON_BIN" scripts/eval_r6d_wm_ablation.py \
      --baseline "$baseline" \
      --wm "$output/per_vla_${policy}.json" \
      --output "$output/pareto_${policy}.json" \
      --seed-count 1 --auroc-gain-min "$AUROC_GAIN_MIN" \
      --dominant-seeds-min "$DOMINANT_SEEDS_MIN" \
      > "$output/pareto_${policy}.log" 2>&1 || true
    echo "DONE $seed pareto $policy" | tee -a "$OUTPUT_ROOT/status.txt"
  done
done

"$PYTHON_BIN" - <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); seeds = sys.argv[2].split()
policies = ["pi0fast_libero", "pi05_libero"]
aggregate = {}
for policy in policies:
    keeps = []
    for seed in seeds:
        report = root / f"seed_{seed}" / f"pareto_{policy}.json"
        if report.exists():
            value = json.loads(report.read_text())
            keeps.append(value["decision"] == "keep")
        else:
            keeps.append(False)
    aggregate[policy] = {"seeds_keep": int(sum(keeps)), "seeds_total": len(keeps),
                         "kept": keeps,
                         "gate_passed": sum(keeps) >= int(sys.argv[3])}
for policy, value in aggregate.items():
    print(f"{policy}: {value['seeds_keep']}/{value['seeds_total']} keep  ->  "
          f"{'PASS' if value['gate_passed'] else 'FAIL'}")
(root / "wm_ablation_aggregate.json").write_text(json.dumps(
    {"schema_version": "rase-r6d-wm-ablation-aggregate/v1",
     "per_vla": aggregate,
     "decision": ("keep" if all(v["gate_passed"] for v in aggregate.values()) else "reject"),
     "note": "keep only if the WM arm Pareto-dominates the no-WM baseline on >=4/5 seeds "
             "per VLA AND all per-VLA safety gates hold; otherwise write an honest negative result."},
    indent=2, sort_keys=True) + "\n")
PY
echo complete > "$OUTPUT_ROOT/COMPLETE"

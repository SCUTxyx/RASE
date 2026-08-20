#!/usr/bin/env bash
# Run Ridge WM training on v4 collection and analyze R4-F gates
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-runs/pre_c0_r4/boundary_train_v4/boundary_transitions.jsonl}"
OUTPUT="${OUTPUT:-runs/pre_c0_r4/world_model_ridge_v5_full}"
LOG="${LOG:-runs/pre_c0_r4/world_model_ridge_v5_full.log}"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"

echo "===== Training Ridge + MLP world model on v4 data =====" | tee -a "$LOG"
echo "Dataset: $DATASET" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"

conda run -n smolvla python -u scripts/train_r4_safe_handback_wm_ridge.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT" \
    --n-folds 6 \
    --ensemble-size 5 \
    --hidden-dim 128 \
    --lr 0.0002 \
    --ridge-alpha 1000.0 \
    --cost-credit 0.05 \
    --epochs 2000 \
    --patience 200 \
    $@ 2>&1 | tee -a "$LOG"

echo ""
echo "===== Training complete =====" | tee -a "$LOG"
echo ""
echo "===== Analyzing gates =====" | tee -a "$LOG"

conda run -n smolvla python - "$OUTPUT/report.json" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    r = json.load(f)

print("=" * 60)
print("R4-F GATE ANALYSIS")
print("=" * 60)

gates = r.get("gates", {})
for key, passed in sorted(gates.items()):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {key}")

# Detailed metrics
sel = r.get("selector_oof", {})
print(f"\n--- Performance Metrics ---")
print(f"  Dynamics improvement: {r.get('dynamics_improvement', 0):.4f}")
print(f"  Dynamics MSE: {r.get('dynamics_mse', 0):.4f}")
print(f"  Persistence MSE: {r.get('persistence_mse', 0):.4f}")
print(f"  Handback AUC: {r.get('handback_auc', 0):.4f}")
print(f"  Handback AP: {r.get('handback_ap', 0):.4f}")
print(f"  Risk AP: {r.get('risk_ap', 0):.4f}")

print(f"\n--- Selector Performance ---")
print(f"  N states: {sel.get('n_states', 0)}")
print(f"  Success rate: {sel.get('success_rate', 0):.4f}")
print(f"  Persistent success: {sel.get('persistent_success_rate', 0):.4f}")
print(f"  Success delta: {sel.get('success_minus_persistent', 0):.4f}")
print(f"  False handbacks: {sel.get('false_handbacks', 0)}")
print(f"  False handback rate: {sel.get('false_handback_rate_persistent_rescuable', 0):.4f}")
print(f"  OFT savings: {sel.get('oft_step_savings_fraction', 0):.4f}")
print(f"  Executed OFT steps: {sel.get('executed_oft_steps', 0)}")
print(f"  Persistent OFT steps: {sel.get('persistent_executed_oft_steps', 0)}")

print(f"\n--- Overall Status ---")
print(f"  Status: {r.get('status', 'unknown')}")
print(f"  Gate evaluation: {r.get('gate_evaluation_status', 'unknown')}")

pass_count = sum(1 for v in gates.values() if v)
total_count = len(gates)
print(f"\n  Gates passed: {pass_count}/{total_count}")
if pass_count == total_count:
    print("  RESULT: ALL GATES PASSED - Ready for R4-G!")
elif pass_count >= 5:
    print(f"  RESULT: {pass_count}/{total_count} passed - Close but not ready")
else:
    print(f"  RESULT: {pass_count}/{total_count} passed - Needs improvement")
PY

echo "" | tee -a "$LOG"
echo "===== Analysis complete =====" | tee -a "$LOG"

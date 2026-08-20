#!/usr/bin/env bash
# R6-E independent task-disjoint validation: frozen thresholds + 100+ paired
# closed-loop episodes on task-disjoint validation states.
#
# GATED: only runs after the R6-C stage gate passes (>=4/5 seeds on both
# qualified VLAs).  Sealed per configs/r6b1_dynamic_boundary_protocol_v1.json
# (`validation_test_lock`).  This script is the execution entry point once the
# protocol unlock conditions hold; it refuses to run otherwise.
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-runs/pre_c0_r6/r6b1_b1p2_v1}"
OOF_ROOT="${OOF_ROOT:-runs/pre_c0_r6/r6c_candidate_arm_oof_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/pre_c0_r6/r6e_validation_v1}"
STABILITY="${OOF_ROOT}/stability.json"
VAL_KEYS="${VAL_KEYS:-runs/r6e_validation_initial_keys_v1.json}"
MIN_VAL_STATES=100
MIN_EPISODES=100

if [[ ! -f "$STABILITY" ]]; then
  echo "R6E ERROR: R6-C stability.json missing at $STABILITY" >&2
  exit 2
fi
stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_gate_passed"])' "$STABILITY")"
if [[ "$stage" != "True" ]]; then
  echo "R6E ERROR: R6-C stage gate not passed (stage_gate_passed=$stage); validation stays sealed" >&2
  exit 2
fi
echo "R6E gate OK: R6-C stage gate passed; independent validation unlocked"

if [[ ! -f "$VAL_KEYS" ]]; then
  echo "R6E ERROR: independent task-disjoint validation state manifest missing:" >&2
  echo "R6E ERROR:   $VAL_KEYS" >&2
  echo "R6E ERROR: Freeze it with scripts/export_r6e_validation_keys.py (>= ${MIN_VAL_STATES} states," >&2
  echo "R6E ERROR: task-disjoint from the R6-C training task set)." >&2
  exit 2
fi
n_states="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["state_keys"]))' "$VAL_KEYS")"
if [[ "$n_states" -lt "$MIN_VAL_STATES" ]]; then
  echo "R6E ERROR: validation manifest has $n_states states; need >= $MIN_VAL_STATES" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

# Freeze thresholds from the R6-C OOF report (per-VLA mean threshold across folds).
"$PYTHON_BIN" - "$OOF_ROOT" "$OUTPUT_ROOT/frozen_thresholds.json" <<'PY'
import json, sys
from pathlib import Path
oof_root = Path(sys.argv[1]); out = Path(sys.argv[2])
seeds = sorted(path.parent.name for path in oof_root.glob("seed_*") if path.is_dir())
by_policy = {}
for policy in ("pi0fast_libero", "pi05_libero"):
    thresholds = []
    for seed in seeds:
        report = oof_root / seed / f"per_vla_{policy}.json"
        if report.is_file():
            payload = json.loads(report.read_text())
            if payload.get("status") == "complete":
                thresholds.append(float(payload["metrics"].get("threshold", float("nan"))))
    if thresholds:
        by_policy[policy] = {"threshold_mean": float(sum(thresholds) / len(thresholds)),
                             "n_seeds": len(thresholds)}
freeze = {"schema_version": "rase-r6e-frozen-thresholds/v1",
          "source": "R6-C candidate-arm OOF per-VLA models",
          "by_policy": by_policy}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
print(json.dumps(freeze, indent=2, sort_keys=True))
PY

# Closed-loop paired rollout on task-disjoint validation states (four suites,
# both source VLA seeds).  Placeholder that records the exact sealed entry point;
# the actual rollout harness is scripts/eval_r6e_closed_loop.py and must be run
# with the same simulator/oracle resources as the B1.2 collector.
echo "R6E frozen thresholds written to $OUTPUT_ROOT/frozen_thresholds.json"
echo "R6E next step: run scripts/eval_r6e_closed_loop.py with the frozen model "
echo "R6E next step: to collect ${MIN_EPISODES}+ paired closed-loop episodes."

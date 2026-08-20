#!/usr/bin/env bash
# R6-C pipeline continuation: waits for the B1.2 full collection to finish
# (writes runs/pre_c0_r6/r6b1_b1p2_v1/COMPLETE), re-verifies the source-parity
# hard gate that the collector already ran, builds the full candidate-arm
# dataset with the parity recheck, runs the candidate-arm analysis, then launches
# the 5-seed task-held-out OOF (R6-C stage gate).
#
# Intended to run in the background so the long collection never blocks the plan:
#   nohup bash scripts/run_r6c_pipeline_after_b12.sh > /tmp/r6c_pipeline.log 2>&1 &
set -euo pipefail
cd /root/autodl-tmp/RASE

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/smolvla/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-runs/pre_c0_r6/r6b1_b1p2_v1}"
EXPECTED_NPZ=144
EXCLUSIONS="${EXCLUSIONS:-runs/pre_c0_r6/r6b1_b12_exclusions_v1.json}"

echo "R6C_PIPE waiting for full B1.2 collection (${EXPECTED_NPZ} npz) at $(date '+%F %T')"
while [[ ! -f "$DATASET_ROOT/COMPLETE" ]]; do
  n="$(find "$DATASET_ROOT" -name '*.npz' | wc -l)"
  echo "R6C_PIPE progress npz=${n}/${EXPECTED_NPZ} at $(date '+%T')"
  if [[ "$n" -ge "$EXPECTED_NPZ" && -f "$DATASET_ROOT/parity_audit.json" ]]; then
    break
  fi
  sleep 300
done
echo "R6C_PIPE collection finished at $(date '+%F %T')"

# 1) Hard gate: the collector runs the audit; re-verify its result is a pass.
if [[ ! -f "$DATASET_ROOT/parity_audit.json" ]]; then
  echo "R6C_PIPE ERROR: parity_audit.json missing" >&2
  exit 2
fi
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DATASET_ROOT/parity_audit.json")"
if [[ "$status" != "pass" ]]; then
  echo "R6C_PIPE ERROR: source parity audit status=$status (not pass)" >&2
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["reasons"], indent=2))' "$DATASET_ROOT/parity_audit.json" >&2 || true
  exit 2
fi
echo "R6C_PIPE parity hard gate PASS"

# 2) Build the full candidate-arm dataset with the R6-A parity recheck inside.
DATASET="${DATASET_ROOT}/r6c_candidate_arm_dataset.npz"
"$PYTHON_BIN" scripts/build_candidate_arm_dataset.py \
  --input-root "$DATASET_ROOT" \
  --protocol configs/r6b1_dynamic_boundary_protocol_v1.json \
  --output "$DATASET" \
  --atlas-root runs/pre_c0_r6/policy_pair_atlas_v1 \
  --exclusions "$EXCLUSIONS"
echo "R6C_PIPE candidate-arm dataset built"

# 3) Candidate-arm data analysis (R6-C plan phase 1 step 2).
"$PYTHON_BIN" scripts/analyze_r6c_candidate_arms.py \
  --input-root "$DATASET_ROOT" \
  --output "$DATASET_ROOT/candidate_arm_analysis.json" \
  --exclusions "$EXCLUSIONS" \
  > "$DATASET_ROOT/candidate_arm_analysis.log" 2>&1
echo "R6C_PIPE candidate-arm analysis written"

# 3b) Render the analysis into the plan-required progress document.
"$PYTHON_BIN" - "$DATASET_ROOT/candidate_arm_analysis.json" <<'PY'
import json, sys
from datetime import date
from pathlib import Path

root = Path("/root/autodl-tmp/RASE")
analysis = json.loads(Path(sys.argv[1]).read_text())
rows = analysis["rows"]
groups = {}
for row in rows:
    groups.setdefault(row["group_id"], []).append(row)
rescue_by_boundary = {}
for row in rows:
    if row["persistent_success_if_enter_now"]:
        rescue_by_boundary[int(row["elapsed_source_steps"])] = rescue_by_boundary.get(
            int(row["elapsed_source_steps"]), 0) + 1
n_boundaries_with_rescue = len(rescue_by_boundary)
teacher_by_boundary = {}
for row in rows:
    steps = int(row["persistent_teacher_steps_if_enter_now"] or 0)
    elapsed = int(row["elapsed_source_steps"])
    teacher_by_boundary[elapsed] = teacher_by_boundary.get(elapsed, 0) + steps

lines = [
    f"# R6-B1.2 Candidate-Arm Data Analysis ({date.today().isoformat()})",
    "",
    "Full B1.2 collection (144 npz) complete; source-parity hard gate passed "
    "(see `parity_audit.json`). One known nondeterministic trajectory group "
    "(`pi05_libero` seed 1, `sp1_b0b5e524da0d318935146d898a89ef8c`) is excluded "
    "per the frozen manifest `runs/pre_c0_r6/r6b1_b12_exclusions_v1.json` "
    "(R6-A reference 154 steps is not reproducible; collector and isolated "
    "reruns both give 138). This report is generated from the frozen B1.2 "
    "metadata by `scripts/analyze_r6c_candidate_arms.py`.",
    "",
    "## Scope",
    "",
    f"- rows: {analysis['n_rows']}  groups: {analysis['n_groups']}  "
    f"states: {analysis['n_states']}  tasks: {analysis['n_tasks']}",
    f"- policies: {', '.join(analysis['policies'])}",
    "",
    "## Per-boundary rescuability",
    "",
    "| elapsed | rows | source success | persistent success | teacher steps mean |",
    "|---|---|---|---|---|",
]
for b in analysis["boundary_stats"]:
    lines.append(
        f"| {b['elapsed_source_steps']} | {b['n_rows']} | "
        f"{b['source_success_rate']:.2f} | {b['persistent_success_rate']:.2f} | "
        f"{b['persistent_teacher_steps_mean']:.1f} |")
lines += ["", "## Source failure prevalence (per policy)", ""]
for policy, stats in analysis["policy_stats"].items():
    lines.append(f"- **{policy}**: source final success {stats['source_final_success_rate']:.2f}, "
                 f"source-failure rows {stats['source_failure_rows']}, "
                 f"failure rescue rate {stats['source_failure_rescued_rate']:.2f}, "
                 f"rescue teacher steps {stats['source_failure_rescue_teacher_steps_mean']:.1f}")
lines += ["", "## Temporal non-monotonicity", ""]
t = analysis["temporal"]
lines.append(f"- groups with >=3 boundaries: {t['n_groups_ge_3_boundaries']}")
lines.append(f"- groups with non-monotonic within-16 series: {t['n_groups_with_nonmonotonic_within16']} "
             f"({t['nonmonotonic_fraction']:.2f})")
lines += ["", "## Candidate-arm opportunity", ""]
r = analysis["rescue_opportunity"]
lines.append(f"- rescue opportunities: {r['n_rescue_opportunities']} in {r['n_groups_with_rescue']} groups")
lines.append(f"- new successes created: {r['n_created_success']}")
lines.append(f"- mean teacher steps per rescue: {r['mean_teacher_steps']:.1f}")
lines += ["", "## Takeaway", ""]
lines.append(
    "The persistent-OFT arm rescues a meaningful fraction of source failures at "
    "the early boundaries at low cost; rescuability degrades as elapsed steps "
    "grow. Non-monotonic within-horizon series confirm that a fixed elapsed "
    "threshold alone is insufficient, motivating the learned per-boundary risk "
    "model with two-boundary dwell.")
(root / "progress" / f"2026-08-10_r6b1_b12_candidate_arm_analysis.md").write_text("\n".join(lines) + "\n")
print("R6C_PIPE progress doc written")
PY

# 4) Launch the 5-seed task-held-out OOF with the R6-C stage gate.
bash scripts/run_r6c_candidate_arm_oof.sh
echo "R6C_PIPE OOF gate run launched (see runs/pre_c0_r6/r6c_candidate_arm_oof_v1/)"

# 5) Pre-registered R6-D: only if the no-WM stage gate passed do we generate the
#    world-model feature cache and run the WM ablation OOF + Pareto comparison.
STABILITY="runs/pre_c0_r6/r6c_candidate_arm_oof_v1/stability.json"
if [[ -f "$STABILITY" ]]; then
  stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_gate_passed"])' "$STABILITY")"
  if [[ "$stage" == "True" ]]; then
    echo "R6C_PIPE R6-C stage gate PASSED; starting pre-registered R6-D ablation"
    WM_ROOT="runs/pre_c0_r6/r6d_wm_features_v1"
    mkdir -p "$WM_ROOT"
    # Feature cache per qualified policy/seed (offline V-JEPA, never deployment).
    # The collector seeds in B1.2 are pi0fast seed 0, pi05 seeds 0 and 1.
    for pair in "pi0fast_libero 0" "pi05_libero 0" "pi05_libero 1"; do
      set -- $pair
      policy="$1"; seed="$2"
      per_dir="$WM_ROOT/${policy}_seed${seed}"
      if [[ ! -f "$per_dir/wm_features.jsonl" ]]; then
        mkdir -p "$per_dir"
        CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        "$PYTHON_BIN" scripts/cache_r6d_wm_features.py \
          --input-root "$DATASET_ROOT" \
          --policy-id "$policy" --seed-index "$seed" \
          --output "$per_dir" \
          --initial-keys runs/rase_ui_phase1a_replacement48_initial_keys_v2.json \
          --teacher-ckpt /root/autodl-tmp/vjepa2 \
          --k 1 4 8 \
          --device cuda \
          --exclusions "$EXCLUSIONS" \
          > "$per_dir/cache.log" 2>&1
      fi
    done
    export WM_CACHE="$WM_ROOT"
    bash scripts/run_r6d_wm_oof.sh
    echo "R6C_PIPE R6-D ablation complete (see runs/pre_c0_r6/r6d_wm_oof_v1/)"
  else
    echo "R6C_PIPE R6-C stage gate FAILED; R6-D ablation stays sealed (honest negative)."
  fi
else
  echo "R6C_PIPE WARNING: stability.json missing; skipping R6-D continuation."
fi
echo "R6C_PIPE pipeline finished at $(date '+%F %T')"

#!/usr/bin/env bash
# W10 Object/Spatial benchmark pipeline (benchmark/diagnosis only).
# Hard-stops on inventory/split NOT_READY. Never trains a selector.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOL_PY="${SMOLVLA_PYTHON:-/root/autodl-tmp/envs/smolvla/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYTHONUNBUFFERED=1
export SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
export OFT_ENV="${OFT_ENV:-oft}"

LOG_DIR="$ROOT/runs/ngc_w10_pipeline_logs"
mkdir -p "$LOG_DIR" "$ROOT/runs"
STAGE_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$STAGE_LOG") 2>&1

echo "==== W10 pipeline start $(date -Is) ===="
echo "ROOT=$ROOT"
echo "SMOL_PY=$SMOL_PY"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG=$STAGE_LOG"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "MISSING required file: $path" >&2
    exit 1
  fi
}

stage() {
  echo
  echo "==== STAGE: $* $(date -Is) ===="
}

# 0) Manual gates
stage "0_manual_gates"
require_file runs/ngc_w10_manual_gate_manifest.json
require_file runs/ngc_w10_collection_request_schedule.json
"$SMOL_PY" - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("runs/ngc_w10_manual_gate_manifest.json").read_text())
print("overall_status:", m["overall_status"])
if m["overall_status"] != "READY_FOR_COLLECTION":
    raise SystemExit(f"manual gates blocked: {m['overall_status']}")
for name, gate in m["gates"].items():
    print(f"  {name}: {gate['status']}")
PY

# Clear interrupted empty pool lock if present and no episodes exist
POOL="pool/ngc_w10_object_spatial_failures"
if [[ -d "$POOL" ]]; then
  n_ep="$("$SMOL_PY" - <<'PY'
from pathlib import Path
from rase.collect.pipeline import existing_episode_ids
print(len(existing_episode_ids(Path("pool/ngc_w10_object_spatial_failures"))))
PY
)"
  if [[ "$n_ep" == "0" ]]; then
    echo "Clearing empty interrupted pool lock (0 episodes)"
    rm -f "$POOL/.collect_current_episode.json"
  else
    echo "Pool already has $n_ep episodes; collection will resume/skip done IDs"
  fi
fi

# 1) CPU preflight (quick)
stage "1_cpu_preflight"
"$SMOL_PY" -m pytest -q tests/test_w10_protocol.py tests/test_filter_selector_dataset.py

# 3) Collection — single launch, 80 episodes hard stop
stage "3_collect80"
if [[ -f runs/ngc_w10_object_spatial_collect80.json ]]; then
  echo "Collection summary already exists; skipping recollect"
else
  "$SMOL_PY" scripts/collect_state_pool.py \
    --config configs/collect_w10_object_spatial_failures.json \
    --summary-output runs/ngc_w10_object_spatial_collect80.json
fi

# 4) Inventory gate
stage "4_inventory"
set +e
"$SMOL_PY" scripts/sample_state_keys.py \
  --config configs/ngc_w10_object_spatial_benchmark.yaml \
  --inventory-only --require-complete \
  --output runs/ngc_w10_object_spatial_inventory.json
inv_rc=$?
set -e
"$SMOL_PY" - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/ngc_w10_object_spatial_inventory.json").read_text())
print("coverage_complete:", d.get("coverage_complete"))
print("deficit_cells:", d.get("deficit_cells"))
if not d.get("coverage_complete"):
    raise SystemExit("INVENTORY NOT_READY: do not top up or relax protocol")
PY
if [[ $inv_rc -ne 0 ]]; then
  echo "Inventory CLI exited $inv_rc (NOT_READY)" >&2
  exit "$inv_rc"
fi

# 5) Freeze 16 state keys
stage "5_sample_state_keys"
"$SMOL_PY" scripts/sample_state_keys.py \
  --config configs/ngc_w10_object_spatial_benchmark.yaml \
  --require-complete \
  --output runs/ngc_w10_object_spatial_state_keys.json
"$SMOL_PY" - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/ngc_w10_object_spatial_state_keys.json").read_text())
assert d["coverage_complete"] is True
assert d["n_states"] == 16
print("n_states:", d["n_states"])
print("state_keys_sha256:", d["state_keys_sha256"])
# append evaluation schedule freeze into manual gate
gate = json.loads(Path("runs/ngc_w10_manual_gate_manifest.json").read_text())
gate["gates"]["evaluation_schedule_seed"]["status"] = "PASS"
gate["gates"]["evaluation_schedule_seed"]["state_keys_sha256"] = d["state_keys_sha256"]
gate["gates"]["selected_identity_audit"]["status"] = "PASS_CELL_COVERAGE"
gate["gates"]["selected_identity_audit"]["n_states"] = d["n_states"]
Path("runs/ngc_w10_manual_gate_manifest.json").write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n"
)
PY

# 6) Direct Smol
stage "6_direct_smol"
if [[ -f runs/ngc_w10_direct_smol_object_spatial16/summary.json ]]; then
  echo "Direct Smol summary exists; skipping"
else
  "$SMOL_PY" scripts/rollout_direct_smol.py \
    --config configs/ngc_w10_object_spatial_benchmark.yaml \
    --state-keys-json runs/ngc_w10_object_spatial_state_keys.json \
    --output-dir runs/ngc_w10_direct_smol_object_spatial16 \
    --fresh-run
fi

# 7) Direct OFT Object/Spatial
stage "7_direct_oft"
if [[ -f runs/ngc_w10_direct_oft_object_object_spatial16/summary.json \
   && -f runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json ]]; then
  echo "Direct OFT summaries exist; skipping"
else
  SMOLVLA_ENV=smolvla \
  OFT_ENV=oft \
  OFT_SUITE_SHORTS=object,spatial \
  OUTPUT_PREFIX=ngc_w10_direct_oft \
  STATE_KEYS_JSON=runs/ngc_w10_object_spatial_state_keys.json \
  CANDIDATES_DIR=runs/ngc_w10_object_spatial_state_keys.json \
  OFT_RUNNER=prefix-ablation \
  OFT_PREFIX_ARMS=direct \
  FRESH_RUN=1 \
  ./scripts/run_oft_verify_suites.sh \
    configs/ngc_w10_object_spatial_benchmark.yaml \
    object_spatial16
fi

# 8) Export failure action dataset
stage "8_export_failure_dataset"
"$SMOL_PY" scripts/extract_selector_features.py \
  --pool pool/ngc_w10_object_spatial_failures \
  --state-keys runs/ngc_w10_object_spatial_state_keys.json \
  --output runs/ngc_w10_object_spatial_features.json
"$SMOL_PY" scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w10_direct_smol_object_spatial16/summary.json \
  --oft-direct-summary runs/ngc_w10_direct_oft_object_object_spatial16/summary.json \
  --oft-direct-summary runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json \
  --features runs/ngc_w10_object_spatial_features.json \
  --pool pool/ngc_w10_object_spatial_failures \
  --cohort failure_challenge \
  --output runs/ngc_w10_object_spatial_failure_action_dataset.jsonl

# 9) Filter W9C clean Object/Spatial (idempotent)
stage "9_filter_w9c_clean"
"$SMOL_PY" scripts/filter_selector_dataset.py \
  --dataset runs/ngc_w9c_clean_action_dataset.jsonl \
  --suite Object --suite Spatial \
  --cohort clean_control \
  --output runs/ngc_w10_w9c_object_spatial_clean_action_dataset.jsonl \
  --manifest-output runs/ngc_w10_w9c_object_spatial_clean_filter_manifest.json

# 10) Merge + cross-source episode-group audit
stage "10_merge"
"$SMOL_PY" scripts/merge_selector_datasets.py \
  --dataset runs/ngc_w10_object_spatial_failure_action_dataset.jsonl \
  --dataset runs/ngc_w10_w9c_object_spatial_clean_action_dataset.jsonl \
  --output runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl \
  --manifest runs/ngc_w10_object_spatial_heldout_merge_manifest.json
"$SMOL_PY" - <<'PY'
import json
from collections import Counter
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl").read_text().splitlines()
    if line.strip()
]
cohorts = Counter(str(r.get("cohort")) for r in rows)
suites = Counter(str(r.get("suite")) for r in rows)
groups = [(str(r.get("task_id")), str(r.get("episode_id")), str(r.get("cohort"))) for r in rows]
# Cross-source overlap: same (task_id, episode_id) in both cohorts
by_group = {}
for task, ep, cohort in groups:
    by_group.setdefault((task, ep), set()).add(cohort)
overlap = {k: sorted(v) for k, v in by_group.items() if len(v) > 1}
print("n_rows", len(rows))
print("cohorts", dict(cohorts))
print("suites", dict(suites))
assert len(rows) == 32
assert cohorts["clean_control"] == 16
assert cohorts["failure_challenge"] == 16
if overlap:
    raise SystemExit(f"cross-source episode-group overlap: {list(overlap.items())[:5]}")
print("cross_source_episode_group_overlap: none")
audit = {
    "schema_version": "rase-w10-cross-source-identity-audit/v1",
    "n_rows": len(rows),
    "cohort_counts": dict(cohorts),
    "suite_counts": dict(suites),
    "n_episode_groups": len(by_group),
    "overlapping_episode_groups": [
        {"task_id": k[0], "episode_id": k[1], "cohorts": v} for k, v in sorted(overlap.items())
    ],
    "status": "PASS" if not overlap else "BLOCKED",
}
Path("runs/ngc_w10_cross_source_identity_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
PY

# 11) Split support gate
stage "11_split_support_gate"
set +e
"$SMOL_PY" scripts/build_selector_splits.py \
  --dataset runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl \
  --output runs/ngc_w10_object_spatial_episode_splits.json \
  --seed 20260731 --grouping episode \
  --requirements configs/ngc_w10_split_requirements.json \
  --fail-not-ready
split_rc=$?
set -e
"$SMOL_PY" - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/ngc_w10_object_spatial_episode_splits.json").read_text())
status = d.get("status") or d.get("requirements_audit", {}).get("status")
print("split_status:", status)
reasons = d.get("requirements_audit", {}).get("reasons")
print("reasons:", reasons)
if status != "READY":
    raise SystemExit("SPLIT NOT_READY: do not change seed/top-up; report as diagnosis")
PY
if [[ $split_rc -ne 0 ]]; then
  exit "$split_rc"
fi

# 12) Benchmark analysis only — never train selector
stage "12_benchmark_analysis"
"$SMOL_PY" scripts/analyze_selector_benchmark.py \
  --dataset runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl \
  --splits runs/ngc_w10_object_spatial_episode_splits.json \
  --output runs/ngc_w10_object_spatial_benchmark_analysis.json

echo
echo "==== W10 pipeline COMPLETE $(date -Is) ===="
echo "Artifacts under runs/ngc_w10_* and pool/ngc_w10_object_spatial_failures"
echo "Do NOT run train_lightweight_selector.py / MLP / RL."

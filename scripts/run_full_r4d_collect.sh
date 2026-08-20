#!/usr/bin/env bash
# PRE-C0-R4 FULL boundary collection (all 4 suites, all 71 states)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/pre_a3_recovery_duration120.yaml}"
KEYS="${KEYS:-runs/rase_pre_a3_keys120_v1.json}"
AUDIT="${AUDIT:-runs/pre_c0_r4/opportunity_audit_costaware_qc.json}"
OUTPUT="${OUTPUT:-runs/pre_c0_r4/boundary_train_v4}"
LOG="${LOG:-runs/pre_c0_r4/boundary_train_v4.log}"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
BASE_PY="${BASE_PY:-${CONDA_ROOT}/bin/python}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
OFT_ENV="${OFT_ENV:-oft}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
PYTHONPATH_OFT="${PYTHONPATH_OFT:-/root/autodl-tmp/src/openvla-oft}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
SERVER_CUDA="${SERVER_CUDA:-0}"
CLIENT_CUDA="${CLIENT_CUDA:-0}"

if [[ ! -f "$KEYS" || ! -f "$AUDIT" ]]; then
  echo "ERROR: missing frozen keys or QC audit" >&2
  exit 1
fi

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

SUITE_SHORTS=(spatial object goal 10)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"

  echo "===== Suite: ${suite} ($short) ====="
  
  # Start OFT server for this suite
  (
    conda activate "$OFT_ENV"
    export CUDA_VISIBLE_DEVICES="$SERVER_CUDA"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="${PYTHONPATH_OFT}:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/pre_c0_r4/oft_server_${short}_v4.log" 2>&1
  ) &
  server_pid=$!
  
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  # Wait for server to be ready
  ready=0
  for _ in $(seq 1 60); do
    if conda run -n "$SMOLVLA_ENV" python scripts/probe_oracle.py \
      --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server did not become ready for $suite" >&2
    exit 1
  fi

  # Collect ALL train states for this suite
  (
    conda activate "$SMOLVLA_ENV"
    export CUDA_VISIBLE_DEVICES="$CLIENT_CUDA"
    export MUJOCO_EGL_DEVICE_ID="$CLIENT_CUDA"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_PLUS_ROOT
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    python -u scripts/collect_r4_boundary_transitions.py \
      --config "$CONFIG" \
      --state-keys-json "$KEYS" \
      --opportunity-audit "$AUDIT" \
      --suite "$suite" \
      --endpoint "$ENDPOINT" \
      --output-dir "$OUTPUT/suite_${short}" \
      --split train \
      --max-states 0 \
      --num-shards 1 \
      --shard-index 0
  ) || {
    echo "ERROR: collection failed for $suite"
    cleanup
    trap - EXIT
    exit 1
  }

  # Kill OFT server before moving to next suite
  cleanup
  trap - EXIT
  sleep 2
done

# Merge all suite results
echo ""
echo "===== Merging all suites ====="
"$BASE_PY" - "$OUTPUT" <<'PY'
import glob, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reports = [json.load(open(p)) for p in sorted(glob.glob(str(root / "suite_*" / "report.json")))]
if not reports:
    raise SystemExit("no R4 suite reports found")
projection_hashes = {r["projection_sha256"] for r in reports}
if len(projection_hashes) != 1:
    raise SystemExit(f"projection mismatch across suites: {projection_hashes}")

rows = []
for path in sorted(glob.glob(str(root / "suite_*" / "boundaries_*.jsonl"))):
    rows.extend(json.loads(line) for line in open(path) if line.strip())
out = root / "boundary_transitions.jsonl"
out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

summary = {
    "schema_version": "rase-pre-c0-r4-boundary-merged/v4",
    "n_suites": len(reports),
    "n_states": sum(r["n_states"] for r in reports),
    "n_rows": len(rows),
    "persistent_replay_matches": sum(r["persistent_replay_matches"] for r in reports),
    "projection_sha256": next(iter(projection_hashes)),
    "suite_reports": reports,
    "output": str(out.resolve()),
    "historical_handback_label_matches": sum(
        int(report.get("historical_handback_label_matches", 0)) for report in reports
    ),
    "historical_handback_labels_compared": sum(
        int(report.get("historical_handback_labels_compared", 0)) for report in reports
    ),
}

# Compute oracle savings and finite-safe gate
states = [state for report in reports for state in report.get("state_summaries", [])]
persistent_success_states = [state for state in states if state.get("persistent_replay_success")]
persistent_steps = sum(int(state["executed_oft_steps"]) for state in states)

rows_by_state = {}
for row in rows:
    rows_by_state.setdefault(str(row["state_key"]), []).append(row)

oracle_steps = 0
for state in states:
    successful = [
        int(row["elapsed_oft_steps"])
        for row in rows_by_state.get(str(state["state_key"]), [])
        if bool(row["success_if_handback_now"])
    ]
    oracle_steps += min(successful, default=int(state["executed_oft_steps"]))

finite_safe = [state for state in states if state.get("finite_safe")]
finite_safe_tasks = {str(state["task_id"]) for state in finite_safe}
savings = 1.0 - oracle_steps / max(1, persistent_steps)

minimum_bins = {}
for state in finite_safe:
    boundary = int(state["minimum_successful_handback_boundary"])
    minimum_bins[str(boundary)] = minimum_bins.get(str(boundary), 0) + 1

gate_reasons = []
if len(finite_safe) < 20:
    gate_reasons.append(f"only {len(finite_safe)} live finite-safe states (<20)")
if len(finite_safe_tasks) < 3:
    gate_reasons.append(f"only {len(finite_safe_tasks)} true tasks have live finite-safe states (<3)")
if savings < 0.20:
    gate_reasons.append(f"live oracle OFT-step savings {savings:.4f} (<0.20)")

summary.update({
    "persistent_success_states": len(persistent_success_states),
    "live_finite_safe_states": len(finite_safe),
    "live_finite_safe_task_count": len(finite_safe_tasks),
    "live_finite_safe_tasks": sorted(finite_safe_tasks),
    "live_minimum_successful_boundary_counts": minimum_bins,
    "persistent_total_executed_oft_steps": persistent_steps,
    "live_oracle_minimum_total_executed_oft_steps": oracle_steps,
    "live_oracle_oft_step_savings_fraction": savings,
    "safe_handback_status": "ready" if not gate_reasons else "not_ready",
    "safe_handback_reasons": gate_reasons,
})

(root / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["safe_handback_status"] != "ready":
    print(f"WARNING: safe-handback gate not ready: {gate_reasons}")
PY

echo ""
echo "===== Collection complete ====="

#!/usr/bin/env bash
# Preregistered W9B clean-control pipeline. Do not run before protocol approval.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${SMOLVLA_ENV:-smolvla}"

COLLECT_CFG="${COLLECT_CFG:-configs/collect_w9b_clean_controls.json}"
CONTROL_CFG="${CONTROL_CFG:-configs/ngc_w9b_clean_controls.yaml}"
SCHEDULE="${SCHEDULE:-configs/w9b_clean_control_schedule.json}"
SCHEDULE_SHA256="${SCHEDULE_SHA256:-71e61d3cd4d36469652735293b7c8e23b93fb22aa450487c58d21e085e8e1943}"
CONTROL_KEYS="${CONTROL_KEYS:-runs/ngc_w9b_clean_control_state_keys.json}"
CONTROL_POOL="${CONTROL_POOL:-pool/ngc_w9b_clean_controls}"

if [[ "$CONTROL_POOL" != "pool/ngc_w9b_clean_controls" ]]; then
  echo "ERROR: W9B CONTROL_POOL must be pool/ngc_w9b_clean_controls" >&2
  exit 1
fi
if [[ "$CONTROL_POOL" == "pool/ngc_w9_clean_controls" ]]; then
  echo "ERROR: refusing to write W9B into legacy W9 pool" >&2
  exit 1
fi

python scripts/generate_w9b_schedule.py \
  --output "$SCHEDULE" \
  --seed 20260730 \
  --expected-sha256 "$SCHEDULE_SHA256" \
  --check

pytest -q \
  tests/test_w9b_schedule.py \
  tests/test_task_fingerprint_stability.py \
  tests/test_lerobot_collection_adapter.py \
  tests/test_perturb_sampler.py \
  tests/test_stratified_sample.py \
  tests/test_sample_state_keys_cli.py \
  tests/test_resume_idempotency.py \
  tests/test_pool_snapshot_roundtrip.py \
  tests/test_state_pool_schema.py

mkdir -p runs
exec 9>runs/ngc_w9b_clean_selector_pipeline.lock
if ! flock -n 9; then
  echo "ERROR: another W9B clean-selector pipeline is active" >&2
  exit 1
fi

coverage_ready() {
  python scripts/sample_state_keys.py \
    --config "$CONTROL_CFG" \
    --output "$CONTROL_KEYS" \
    --require-complete
}

python scripts/rollout_direct_smol.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --state-keys-json runs/ngc_w7_heldout24_state_keys.json \
  --output-dir runs/ngc_w9b_direct_smol_failure24 \
  --resume

if [[ -f "$CONTROL_KEYS" ]] && python - "$CONTROL_KEYS" >/dev/null 2>&1 <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("coverage_complete") and x.get("n_states")==32 else 1)
PY
then
  echo "SKIP_COMPLETED stage=w9b-clean-control-collection"
else
  ready=0
  for batch_id in 1 2 3; do
    python scripts/collect_state_pool.py \
      --config "$COLLECT_CFG" \
      --schedule-batch "$batch_id" \
      --summary-output "runs/ngc_w9b_clean_collect_batch${batch_id}.json"
    if coverage_ready; then
      ready=1
      break
    fi
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: W9B coverage incomplete after preregistered 140 episodes" >&2
    exit 2
  fi
fi

python scripts/rollout_direct_smol.py \
  --config "$CONTROL_CFG" \
  --state-keys-json "$CONTROL_KEYS" \
  --output-dir runs/ngc_w9b_direct_smol_clean32 \
  --resume

OUTPUT_PREFIX=ngc_w9b_direct_oft \
STATE_KEYS_JSON="$CONTROL_KEYS" \
CANDIDATES_DIR="$CONTROL_KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
./scripts/run_oft_verify_suites.sh "$CONTROL_CFG" clean32

conda activate "${SMOLVLA_ENV:-smolvla}"
python scripts/extract_selector_features.py \
  --pool pool/ngc_w5_l1_l2_camera_robot \
  --state-keys runs/ngc_w7_heldout24_state_keys.json \
  --output runs/ngc_w9b_failure24_features.json
python scripts/extract_selector_features.py \
  --pool "$CONTROL_POOL" \
  --state-keys "$CONTROL_KEYS" \
  --output runs/ngc_w9b_clean32_features.json

python scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w9b_direct_smol_failure24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_spatial_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_object_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_goal_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_10_heldout24/summary.json \
  --features runs/ngc_w9b_failure24_features.json \
  --pool pool/ngc_w5_l1_l2_camera_robot \
  --cohort failure_challenge \
  --output runs/ngc_w9b_failure_action_dataset.jsonl

python scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w9b_direct_smol_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9b_direct_oft_spatial_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9b_direct_oft_object_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9b_direct_oft_goal_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9b_direct_oft_10_clean32/summary.json \
  --features runs/ngc_w9b_clean32_features.json \
  --pool "$CONTROL_POOL" \
  --cohort clean_control \
  --output runs/ngc_w9b_clean_action_dataset.jsonl

python scripts/merge_selector_datasets.py \
  --dataset runs/ngc_w9b_failure_action_dataset.jsonl \
  --dataset runs/ngc_w9b_clean_action_dataset.jsonl \
  --output runs/ngc_w9b_selector_dataset.jsonl \
  --manifest runs/ngc_w9b_selector_dataset_manifest.json

for grouping in episode task; do
  python scripts/build_selector_splits.py \
    --dataset runs/ngc_w9b_selector_dataset.jsonl \
    --grouping "$grouping" \
    --seed 20260730 \
    --output "runs/ngc_w9b_selector_${grouping}_splits.json"
  set +e
  python scripts/train_lightweight_selector.py \
    --dataset runs/ngc_w9b_selector_dataset.jsonl \
    --splits "runs/ngc_w9b_selector_${grouping}_splits.json" \
    --output-dir "runs/ngc_w9b_selector_${grouping}" \
    --min-train-states 30
  status=$?
  set -e
  if [[ "$status" != "0" && "$status" != "2" ]]; then
    echo "ERROR: W9B selector ${grouping} failed status=${status}" >&2
    exit "$status"
  fi
done

python scripts/summarize_selector_gate.py \
  --dataset runs/ngc_w9b_selector_dataset.jsonl \
  --episode-dir runs/ngc_w9b_selector_episode \
  --task-dir runs/ngc_w9b_selector_task \
  --output-json runs/ngc_w9b_selector_gate_summary.json \
  --output-md runs/ngc_w9b_selector_gate_summary.md

echo "W9B_CLEAN_SELECTOR_PIPELINE_DONE"

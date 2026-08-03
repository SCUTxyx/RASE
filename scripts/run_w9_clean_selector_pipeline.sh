#!/usr/bin/env bash
# Collect clean controls, label direct policy arms, and gate a tiny selector.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh"
conda activate "${SMOLVLA_ENV:-smolvla}"

# Fail before any simulator/GPU work if the W9 contracts or the task-identity
# hotfix regress. Keep this list focused so it remains cheap on the server.
pytest -q \
  tests/test_task_fingerprint_stability.py \
  tests/test_lightweight_selector.py \
  tests/test_selector_features.py \
  tests/test_lerobot_collection_adapter.py \
  tests/test_perturb_sampler.py \
  tests/test_stratified_sample.py \
  tests/test_sample_state_keys_cli.py \
  tests/test_resume_idempotency.py \
  tests/test_pool_snapshot_roundtrip.py

COLLECT_CFG="${COLLECT_CFG:-configs/collect_w9_clean_controls.json}"
CONTROL_CFG="${CONTROL_CFG:-configs/ngc_w9_clean_controls.yaml}"
CONTROL_KEYS="${CONTROL_KEYS:-runs/ngc_w9_clean_control_state_keys.json}"
CONTROL_POOL="${CONTROL_POOL:-pool/ngc_w9_clean_controls}"

mkdir -p runs
exec 9>runs/ngc_w9_clean_selector_pipeline.lock
if ! flock -n 9; then
  echo "ERROR: another W9 clean-selector pipeline is active" >&2
  exit 1
fi

coverage_ready() {
  python scripts/sample_state_keys.py \
    --config "$CONTROL_CFG" \
    --output "$CONTROL_KEYS" \
    --require-complete
}

# Highest-information first: validate the true direct-Smol failure arm before
# spending hours collecting controls. This stage is resumable and only 24 runs.
python scripts/rollout_direct_smol.py \
  --config configs/ngc_w7_heldout24_screen.yaml \
  --state-keys-json runs/ngc_w7_heldout24_state_keys.json \
  --output-dir runs/ngc_w9_direct_smol_failure24 \
  --resume

if [[ -f "$CONTROL_KEYS" ]] && python - "$CONTROL_KEYS" >/dev/null 2>&1 <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("coverage_complete") and x.get("n_states")==32 else 1)
PY
then
  echo "SKIP_COMPLETED stage=clean-control-collection"
else
  batch_specs=("60:20260730" "40:20260731" "40:20260732")
  ready=0
  for spec in "${batch_specs[@]}"; do
    episodes="${spec%%:*}"
    seed="${spec##*:}"
    python scripts/collect_state_pool.py \
      --config "$COLLECT_CFG" \
      --episodes "$episodes" \
      --seed "$seed" \
      --summary-output "runs/ngc_w9_clean_collect_${seed}.json"
    if coverage_ready; then
      ready=1
      break
    fi
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: clean-control coverage incomplete after preregistered 140 episodes" >&2
    exit 2
  fi
fi

python scripts/rollout_direct_smol.py \
  --config "$CONTROL_CFG" \
  --state-keys-json "$CONTROL_KEYS" \
  --output-dir runs/ngc_w9_direct_smol_clean32 \
  --resume

OUTPUT_PREFIX=ngc_w9_direct_oft \
STATE_KEYS_JSON="$CONTROL_KEYS" \
CANDIDATES_DIR="$CONTROL_KEYS" \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
./scripts/run_oft_verify_suites.sh "$CONTROL_CFG" clean32

conda activate "${SMOLVLA_ENV:-smolvla}"
python scripts/extract_selector_features.py \
  --pool pool/ngc_w5_l1_l2_camera_robot \
  --state-keys runs/ngc_w7_heldout24_state_keys.json \
  --output runs/ngc_w9_failure24_features.json
python scripts/extract_selector_features.py \
  --pool "$CONTROL_POOL" \
  --state-keys "$CONTROL_KEYS" \
  --output runs/ngc_w9_clean32_features.json

python scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w9_direct_smol_failure24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_spatial_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_object_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_goal_heldout24/summary.json \
  --oft-direct-summary runs/ngc_w8_direct_oft_10_heldout24/summary.json \
  --features runs/ngc_w9_failure24_features.json \
  --pool pool/ngc_w5_l1_l2_camera_robot \
  --cohort failure_challenge \
  --output runs/ngc_w9_failure_action_dataset.jsonl

python scripts/export_selector_action_dataset.py \
  --smol-direct-summary runs/ngc_w9_direct_smol_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9_direct_oft_spatial_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9_direct_oft_object_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9_direct_oft_goal_clean32/summary.json \
  --oft-direct-summary runs/ngc_w9_direct_oft_10_clean32/summary.json \
  --features runs/ngc_w9_clean32_features.json \
  --pool "$CONTROL_POOL" \
  --cohort clean_control \
  --output runs/ngc_w9_clean_action_dataset.jsonl

python scripts/merge_selector_datasets.py \
  --dataset runs/ngc_w9_failure_action_dataset.jsonl \
  --dataset runs/ngc_w9_clean_action_dataset.jsonl \
  --output runs/ngc_w9_selector_dataset.jsonl \
  --manifest runs/ngc_w9_selector_dataset_manifest.json

for grouping in episode task; do
  python scripts/build_selector_splits.py \
    --dataset runs/ngc_w9_selector_dataset.jsonl \
    --grouping "$grouping" \
    --seed 20260730 \
    --output "runs/ngc_w9_selector_${grouping}_splits.json"
  set +e
  python scripts/train_lightweight_selector.py \
    --dataset runs/ngc_w9_selector_dataset.jsonl \
    --splits "runs/ngc_w9_selector_${grouping}_splits.json" \
    --output-dir "runs/ngc_w9_selector_${grouping}" \
    --min-train-states 30
  status=$?
  set -e
  if [[ "$status" != "0" && "$status" != "2" ]]; then
    echo "ERROR: selector ${grouping} run failed status=${status}" >&2
    exit "$status"
  fi
  echo "SELECTOR_GATE grouping=${grouping} status=${status}"
done

python scripts/summarize_selector_gate.py \
  --dataset runs/ngc_w9_selector_dataset.jsonl \
  --episode-dir runs/ngc_w9_selector_episode \
  --task-dir runs/ngc_w9_selector_task \
  --output-json runs/ngc_w9_selector_gate_summary.json \
  --output-md runs/ngc_w9_selector_gate_summary.md

echo "W9_CLEAN_SELECTOR_PIPELINE_DONE"

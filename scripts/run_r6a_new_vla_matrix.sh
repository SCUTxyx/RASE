#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
KEYS=runs/rase_ui_phase1a_replacement48_initial_keys_v2.json
ROOT=runs/pre_c0_r6/policy_pair_atlas_v1
TEXT_TOK=ckpts/paligemma_tokenizer_35e4f46
ACTION_TOK=ckpts/pi0fast_action_tokenizer_79ae83e

export LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

wait_for_file() {
  local path="$1"
  while [[ ! -s "$path" ]]; do
    echo "R6A_WAIT $path"
    sleep 30
  done
}

# Keep the GPU schedule deterministic: finish both already-running SmolVLA
# seeds before loading either 3B-class source policy.
wait_for_file "$ROOT/smolvla_libero/seed_0/summary.json"
wait_for_file "$ROOT/smolvla_libero/seed_1/summary.json"
wait_for_file ckpts/pi0fast_libero/model.safetensors
wait_for_file ckpts/pi05_libero/model.safetensors
wait_for_file "$TEXT_TOK/tokenizer.json"
wait_for_file "$TEXT_TOK/tokenizer.model"
wait_for_file "$ACTION_TOK/processor_config.json"

$PY scripts/rollout_lerobot_source_from_initial_states.py \
  --initial-keys "$KEYS" \
  --policy-path ckpts/pi0fast_libero \
  --tokenizer-path "$TEXT_TOK" \
  --action-tokenizer-path "$ACTION_TOK" \
  --policy-id pi0fast_libero \
  --seed-index 0 \
  --max-states 1 \
  --fresh-run \
  --output-dir "$ROOT/smoke/pi0fast_seed0_v2"

for seed in 0 1; do
  $PY scripts/rollout_lerobot_source_from_initial_states.py \
    --initial-keys "$KEYS" \
    --policy-path ckpts/pi0fast_libero \
    --tokenizer-path "$TEXT_TOK" \
    --action-tokenizer-path "$ACTION_TOK" \
    --policy-id pi0fast_libero \
    --seed-index "$seed" \
    --fresh-run \
    --output-dir "$ROOT/pi0fast_libero/seed_$seed"
done

$PY scripts/rollout_lerobot_source_from_initial_states.py \
  --initial-keys "$KEYS" \
  --policy-path ckpts/pi05_libero \
  --tokenizer-path "$TEXT_TOK" \
  --policy-id pi05_libero \
  --seed-index 0 \
  --max-states 1 \
  --fresh-run \
  --output-dir "$ROOT/smoke/pi05_seed0"

for seed in 0 1; do
  $PY scripts/rollout_lerobot_source_from_initial_states.py \
    --initial-keys "$KEYS" \
    --policy-path ckpts/pi05_libero \
    --tokenizer-path "$TEXT_TOK" \
    --policy-id pi05_libero \
    --seed-index "$seed" \
    --fresh-run \
    --output-dir "$ROOT/pi05_libero/seed_$seed"
done

set +e
$PY scripts/audit_r6a_policy_pair_atlas.py \
  --manifest configs/r6a_policy_pair_manifest_v1.json \
  --oft-analysis runs/rase_ui_phase1a_replacement48_analysis_v2.json \
  --source-summary "$ROOT/smolvla_libero/seed_0/summary.json" \
  --source-summary "$ROOT/smolvla_libero/seed_1/summary.json" \
  --source-summary "$ROOT/pi0fast_libero/seed_0/summary.json" \
  --source-summary "$ROOT/pi0fast_libero/seed_1/summary.json" \
  --source-summary "$ROOT/pi05_libero/seed_0/summary.json" \
  --source-summary "$ROOT/pi05_libero/seed_1/summary.json" \
  --output runs/pre_c0_r6/policy_pair_atlas_v1.json
status=$?
set -e

if [[ "$status" -ne 0 && "$status" -ne 2 ]]; then
  exit "$status"
fi
echo "R6A_PIPELINE_COMPLETE atlas_exit=$status"

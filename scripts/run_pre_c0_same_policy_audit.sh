#!/usr/bin/env bash
# PRE-C0 natural same-policy corrective pilot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
STAGE_KEYS="${STAGE_KEYS:-artifacts/pre_c0/pre_c0_48_state_manifest.json}"
OUTPUT="${OUTPUT:-runs/rase_pre_c0_same_policy_pilot48_v1}"
AUDIT="${AUDIT:-runs/rase_pre_c0_same_policy_audit_v1.json}"
DECISION="${DECISION:-runs/rase_pre_c0_decision_v1.json}"
LOG="${LOG:-runs/rase_pre_c0_same_policy_pilot48_v1.log}"
FRESH_RUN="${FRESH_RUN:-1}"
LIMIT="${LIMIT:-0}"

if [[ "$FRESH_RUN" != "0" && "$FRESH_RUN" != "1" ]]; then
  echo "ERROR: FRESH_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ ! -f "$STAGE_KEYS" ]]; then
  echo "ERROR: missing frozen stage keys: $STAGE_KEYS" >&2
  exit 1
fi
if [[ "$FRESH_RUN" == "1" && -e "$OUTPUT" ]]; then
  echo "ERROR: fresh output exists: $OUTPUT" >&2
  exit 1
fi

mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

mode=(--resume)
if [[ "$FRESH_RUN" == "1" ]]; then
  mode=(--fresh-run)
fi
limit_args=()
if [[ "$LIMIT" != "0" ]]; then
  limit_args=(--limit "$LIMIT")
fi

echo "=== PRE-C0 NATURAL SAME-POLICY CORRECTIVE ==="
"$PY" scripts/generate_smolvla_corrective_candidates.py \
  --config "$CONFIG" \
  --stage-keys "$STAGE_KEYS" \
  --output-dir "$OUTPUT" \
  --stages T1 T3 \
  "${limit_args[@]}" \
  "${mode[@]}"

echo "=== PRE-C0 NATURAL GATE A ==="
"$PY" scripts/analyze_same_policy_headroom.py \
  --rollout-dir "$OUTPUT" \
  --output "$AUDIT" \
  --decision-output "$DECISION"

echo "PRE_C0_NATURAL_DONE audit=$AUDIT decision=$DECISION"

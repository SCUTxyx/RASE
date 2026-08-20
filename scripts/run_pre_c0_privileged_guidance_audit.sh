#!/usr/bin/env bash
# PRE-C0 Gate B offline privileged upper-bound audit.
# Run only when Natural Gate A decision is run_privileged_guidance_upper_bound.
# Naming: privileged trust-region action refinement / Best-of-K ceiling.
# Do NOT describe this as SmolVLA flow API guidance.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
NATURAL_DECISION="${NATURAL_DECISION:-runs/rase_pre_c0_decision_v1.json}"
NATURAL_ROLLOUT="${NATURAL_ROLLOUT:-runs/rase_pre_c0_same_policy_pilot48_v1}"
AUDIT="${AUDIT:-runs/rase_pre_c0_guided_audit_v1.json}"
DECISION="${DECISION:-runs/rase_pre_c0_guided_decision_v1.json}"
TRUST_REGION="${TRUST_REGION:-artifacts/pre_c0/guidance_trust_region_audit.json}"
ARTIFACTS_JSON="${ARTIFACTS_JSON:-artifacts/pre_c0/gate_b_results.json}"
LOG="${LOG:-runs/rase_pre_c0_guided_audit_v1.log}"
FAMILY="${FAMILY:-strict_resample}"
K="${K:-8}"

mkdir -p runs artifacts/pre_c0 progress
exec > >(tee -a "$LOG") 2>&1

if [[ ! -f "$NATURAL_DECISION" ]]; then
  echo "ERROR: missing Gate A decision: $NATURAL_DECISION" >&2
  exit 1
fi

gate_decision="$("$PY" - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$NATURAL_DECISION").read_text())
print(payload.get("decision", ""))
PY
)"
if [[ "$gate_decision" != "run_privileged_guidance_upper_bound" ]]; then
  echo "ERROR: Gate B blocked; natural decision is '$gate_decision' (need run_privileged_guidance_upper_bound)" >&2
  exit 2
fi

echo "=== PRE-C0 GATE B OFFLINE PRIVILEGED UPPER BOUND ==="
echo "note=privileged trust-region action refinement / Best-of-K ceiling; not SmolVLA API injection"
"$PY" scripts/run_privileged_guidance_audit.py \
  --natural-rollout-dir "$NATURAL_ROLLOUT" \
  --family "$FAMILY" \
  --k "$K" \
  --output "$AUDIT" \
  --decision-output "$DECISION" \
  --trust-region-output "$TRUST_REGION"

"$PY" - <<PY
import json
from pathlib import Path

audit = json.loads(Path("$AUDIT").read_text())
decision = json.loads(Path("$DECISION").read_text())
artifact = {
    "schema_version": "rase-pre-c0-gate-b-artifacts/v1",
    "mode": "offline_best_of_k_upper_bound",
    "naming": "privileged trust-region action refinement",
    "not_smolvla_flow_api_guidance": True,
    "audit": "$AUDIT",
    "decision": "$DECISION",
    "trust_region_audit": "$TRUST_REGION",
    "guided_gain_pp": audit.get("guided_gain_pp"),
    "frozen_same_policy_recovery": audit.get("frozen_same_policy_recovery"),
    "gate_pass": audit.get("gate_pass"),
    "decision_label": decision.get("decision"),
}
Path("$ARTIFACTS_JSON").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(json.dumps(artifact, sort_keys=True))
PY

echo "PRE_C0_GATE_B_OFFLINE_DONE audit=$AUDIT decision=$DECISION trust_region=$TRUST_REGION"

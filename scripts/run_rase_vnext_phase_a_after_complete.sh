#!/usr/bin/env bash
# Run only after the frozen confirmation runner has written COMPLETE.json.
set -euo pipefail

cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
CONFIRMATION=runs/rase_vnext/confirmation_v1
MANIFEST=runs/rase_vnext/frozen/confirmation_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/phase_a_v1

if [[ ! -f "$CONFIRMATION/COMPLETE.json" ]]; then
  echo "Phase A remains locked: $CONFIRMATION/COMPLETE.json is absent." >&2
  exit 20
fi
for path in "$CONFIRMATION/branches.jsonl" "$MANIFEST" "$PROTOCOL"; do
  if [[ ! -f "$path" ]]; then
    echo "Required frozen input is absent: $path" >&2
    exit 21
  fi
done

mkdir -p "$OUT"
set +e
"$PY" scripts/audit_rase_vnext_phase_a.py \
  --branches "$CONFIRMATION/branches.jsonl" \
  --manifest "$MANIFEST" \
  --protocol "$PROTOCOL" \
  --output "$OUT/phase_a_audit.json" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 202708 \
  --tie-margin 0.0 \
  > "$OUT/phase_a_audit.stdout"
status=$?
set -e

"$PY" -c 'import json,sys; value=json.load(open(sys.argv[1])); print(json.dumps({"status":value["status"],"verdict":value["verdict"],"unlocks":value["unlocks"]},indent=2,sort_keys=True))' "$OUT/phase_a_audit.json"
exit "$status"

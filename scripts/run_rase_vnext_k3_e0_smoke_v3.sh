#!/usr/bin/env bash
# RASE K3-E0 native-capture smoke v3 (revised capture contract)
# Verifies: continue/requery/fallback executable + full native capture,
# abort control-only, resample deterministic capability mask.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/k3_e0_native_capture_smoke_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/k3_e0_native_capture_smoke_v3
mkdir -p "$OUT"

# Sanity: manifest protocol hash must match the frozen protocol.
python3 - "$MANIFEST" "$PROTOCOL" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
protocol_hash = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
assert manifest["protocol_sha256"] == protocol_hash, "protocol hash mismatch"
print(f"protocol ok: {protocol_hash[:16]}...")
PY

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

# Start OFT oracle server for Spatial.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/ckpts/oft_spatial" RASE_OFT_SUITE="libero_spatial" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_spatial.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 90); do
  if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 \
    --expect-suite libero_spatial >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  tail -100 "$OUT/oft_spatial.log" >&2 || true
  exit 31
fi
echo "oracle ready"

# Run the E0 smoke collection (single group, 5 operator slots).
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/collect_rase_vnext_discovery.py \
  --manifest "$MANIFEST" \
  --protocol "$PROTOCOL" \
  --output-dir "$OUT" \
  --policy-path ckpts/pi0fast_libero \
  --policy-id pi0fast.libero \
  --suite Spatial \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
  --candidate-capture-dir "$OUT/captures"

# Audit the captured group + write E0 verdict.
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
from rase.vnext.candidate_capture import audit_candidate_capture

out = pathlib.Path(sys.argv[1])
groups = sorted((out / "groups").glob("*.json"))
assert groups, "no group records produced"
group = json.loads(groups[0].read_text())
rows = {str(row["operator_id"]): row for row in group["rows"]}
expected = {
    "continue.source": "executable",
    "requery.source": "executable",
    "fallback.persistent": "executable",
    "abort.safe": "control_only_abort",
    "resample.source": "incapable_missing",
}
failures = []
if not group.get("prefix_available"):
    failures.append("prefix_available is False")
for operator, capability in expected.items():
    row = rows.get(operator)
    if row is None:
        failures.append(f"missing row {operator}")
        continue
    if row.get("capability_status") != capability:
        failures.append(f"{operator}: capability {row.get('capability_status')!r} != {capability!r}")
    if row.get("chunk_origin") in (None, ""):
        failures.append(f"{operator}: missing chunk_origin")
for operator in ("continue.source", "requery.source", "fallback.persistent"):
    row = rows[operator]
    if row.get("inference_event_id") is None and operator != "fallback.persistent":
        failures.append(f"{operator}: missing inference_event_id")
    if row.get("success") is None:
        failures.append(f"{operator}: missing success")

captures = sorted((out / "captures").glob("*.json"))
assert captures, "no capture metadata produced"
audit = audit_candidate_capture(captures[0])
if audit["status"] != "PASS":
    failures.append(f"capture audit: {audit['failures']}")

verdict = {
    "schema_version": "rase-vnext-k3-e0-smoke-verdict/v1",
    "status": "E0_CAPTURE_PASS" if not failures else "E0_CAPTURE_FAIL",
    "failures": failures,
    "prefix_available": bool(group.get("prefix_available")),
    "capability_observed": {op: rows[op].get("capability_status") for op in expected},
    "capture_audit": audit,
    "group_path": str(groups[0]),
    "capture_metadata_path": str(captures[0]),
}
(out / "E0_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
raise SystemExit(0 if not failures else 2)
PY

echo "E0 smoke v3 complete"

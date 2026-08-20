#!/usr/bin/env bash
# B0: pi0.5 capability / parity smoke — 1 group × 6 ops on Spatial.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/pi05_challenge_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/pi05_smoke_v1
mkdir -p "$OUT"

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

# Record checkpoint identity (B0 requirement).
python3 - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
root = pathlib.Path("/root/autodl-tmp/RASE/ckpts/pi05_libero")
records = []
for path in sorted(root.iterdir()):
    if path.is_file():
        records.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size})
tokenizer = pathlib.Path("/root/autodl-tmp/RASE/ckpts/paligemma_tokenizer_35e4f46")
tok_records = [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(tokenizer.iterdir()) if p.is_file()]
(out / "checkpoint_identity.json").write_text(json.dumps({
    "policy_checkpoint": str(root), "files": records,
    "tokenizer": str(tokenizer), "files": tok_records,
}, indent=2, sort_keys=True) + "\n")
print("checkpoint identity recorded:", len(records), "policy files,", len(tok_records), "tokenizer files")
PY

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

# 1-group smoke with pi0.5 policy (no action-tokenizer override for pi05).
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/collect_rase_vnext_discovery.py \
  --manifest "$MANIFEST" \
  --protocol "$PROTOCOL" \
  --output-dir "$OUT" \
  --policy-path ckpts/pi05_libero \
  --policy-id pi05.libero \
  --suite Spatial \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --max-groups 1 \
  --candidate-capture-dir "$OUT/captures"

# Audit the smoke group.
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
from rase.vnext.candidate_capture import audit_candidate_capture
out = pathlib.Path(sys.argv[1])
groups = sorted((out / "groups").glob("*.json"))
assert groups, "no group produced"
g = json.loads(groups[0].read_text())
print("prefix_available:", g.get("prefix_available"))
rows = {str(r["operator_id"]): r for r in g["rows"]}
for op in sorted(rows):
    r = rows[op]
    print(" %-28s cap=%-22s origin=%-18s event=%s cursor=%s success=%s" % (
        op, r.get("capability_status"), r.get("chunk_origin"),
        str(r.get("inference_event_id"))[:8], r.get("queue_cursor_at_boundary"), r.get("success")))
caps = sorted((out / "captures").glob("*.json"))
print("captures:", len(caps))
if caps:
    print("capture audit:", audit_candidate_capture(caps[0]))
fails = []
if not g.get("prefix_available"):
    fails.append("prefix_available=False")
if any(r.get("capability_status") == "execution_error" for r in g["rows"]):
    fails.append("execution_error present")
verdict = {
    "schema_version": "rase-vnext-pi05-smoke-verdict/v1",
    "status": "PI05_SMOKE_PASS" if not fails else "PI05_SMOKE_FAIL",
    "failures": fails,
    "prefix_available": bool(g.get("prefix_available")),
    "capability": {op: rows[op].get("capability_status") for op in rows},
}
(out / "PI05_SMOKE_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
raise SystemExit(0 if not fails else 2)
PY

echo "pi05 smoke complete"

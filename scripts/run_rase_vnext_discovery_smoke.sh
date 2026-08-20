#!/usr/bin/env bash
# One manifest-bound five-operator GPU smoke group; never enters scientific discovery data.
set -euo pipefail
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
OUT=runs/rase_vnext/discovery_smoke_fresh_env_v2
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
RASE_OFT_CHECKPOINT="$PWD/ckpts/oft_spatial" RASE_OFT_SUITE=libero_spatial \
"$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_spatial.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite libero_spatial >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then tail -100 "$OUT/oft_spatial.log" >&2 || true; exit 31; fi

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PY" -u scripts/collect_rase_vnext_discovery.py \
  --manifest runs/rase_vnext/frozen/discovery_manifest_v1.json \
  --protocol configs/rase_vnext_protocol_v1.json --output-dir "$OUT" \
  --suite Spatial --policy-id pi05.libero --policy-path ckpts/pi05_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --endpoint tcp://127.0.0.1:5555 --max-groups 1
cleanup
trap - EXIT
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); files=list((root/"groups").glob("*.json"))
assert len(files)==1
x=json.load(open(files[0])); rows=x["rows"]
assert len(rows)==5 and all(r["completed"] is True for r in rows)
print(json.dumps({"status":"PASS","operators":{r["operator_id"]:{"available":r["available"],"success":r["success"],"stop_reason":r.get("stop_reason"),"utility":r.get("utility")} for r in rows}},indent=2,sort_keys=True))
PY

#!/usr/bin/env bash
# Resume-safe full vNext discovery: 16 roots x 2 policies x 2 points x 5 ops x K3.
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/discovery_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/discovery_v1
mkdir -p "$OUT"

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
    *) echo "unknown suite $1" >&2; exit 2 ;;
  esac
}
checkpoint() {
  case "$1" in
    Spatial) echo ckpts/oft_spatial ;;
    Object) echo ckpts/oft_object ;;
    Goal) echo ckpts/oft_goal ;;
    Long) echo ckpts/oft_10 ;;
  esac
}
collect_policy() {
  local label="$1" policy="$2"
  local -a policy_args
  if [[ "$policy" == pi05.libero ]]; then
    policy_args=(--policy-path ckpts/pi05_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  else
    policy_args=(--policy-path ckpts/pi0fast_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  fi
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -u scripts/collect_rase_vnext_discovery.py \
    --manifest "$MANIFEST" --protocol "$PROTOCOL" --output-dir "$OUT" \
    --suite "$label" --policy-id "$policy" "${policy_args[@]}" \
    --endpoint tcp://127.0.0.1:5555
}

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/oft_${label,,}.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 60); do
    if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite "$suite" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then tail -100 "$server_log" >&2 || true; exit 31; fi
  echo "VNEXT OFT ready suite=$label pid=$server_pid"
  collect_policy "$label" pi05.libero
  collect_policy "$label" pi0fast.libero
  cleanup
  trap - EXIT
done

"$PY" scripts/collect_rase_vnext_discovery.py \
  --manifest "$MANIFEST" --protocol "$PROTOCOL" --output-dir "$OUT" --summarize
if "$PY" scripts/audit_rase_vnext_discovery_feasibility.py \
  --branches "$OUT/branches.jsonl" --manifest "$MANIFEST" --protocol "$PROTOCOL" \
  --output "$OUT/feasibility_audit.json"; then
  gate=PASS
else
  gate=FAIL
fi
"$PY" - "$OUT" "$gate" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, gate = Path(sys.argv[1]), sys.argv[2]
audit = root / "feasibility_audit.json"
payload = {
    "status": "DISCOVERY_COMPLETE", "feasibility_gate": gate,
    "audit": str(audit), "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
}
(root / "COMPLETE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

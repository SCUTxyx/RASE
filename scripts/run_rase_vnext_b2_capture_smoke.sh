#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
OFT_PY=/root/autodl-tmp/envs/oft/bin/python
MANIFEST=runs/rase_vnext/frozen/b2_capture_smoke_manifest_v1.json
PROTOCOL=configs/rase_vnext_protocol_v1.json
OUT=runs/rase_vnext/b2_capture_smoke_v1
CAPTURE="$OUT/captures"
mkdir -p "$OUT" "$CAPTURE"

actual_suite() {
  case "$1" in
    Spatial) echo libero_spatial ;;
    Object) echo libero_object ;;
    Goal) echo libero_goal ;;
    Long) echo libero_10 ;;
    *) return 2 ;;
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
cleanup_server() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

for label in Spatial Object Goal Long; do
  suite="$(actual_suite "$label")"
  ckpt="$(checkpoint "$label")"
  server_log="$OUT/oft_${label,,}.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
  RASE_OFT_CHECKPOINT="$PWD/$ckpt" RASE_OFT_SUITE="$suite" \
  "$OFT_PY" -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
    --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$server_log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 60); do
    if "$PY" scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 \
      --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -100 "$server_log" >&2 || true
    exit 31
  fi
  echo "B2_CAPTURE oracle_ready suite=$label pid=$server_pid"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/collect_rase_vnext_discovery.py \
    --manifest "$MANIFEST" --protocol "$PROTOCOL" --output-dir "$OUT" \
    --suite "$label" --policy-id pi0fast.libero \
    --policy-path ckpts/pi0fast_libero \
    --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
    --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e \
    --endpoint tcp://127.0.0.1:5555 \
    --candidate-capture-dir "$CAPTURE"
  cleanup_server
done

"$PY" scripts/collect_rase_vnext_discovery.py \
  --manifest "$MANIFEST" --protocol "$PROTOCOL" --output-dir "$OUT" --summarize
"$PY" scripts/audit_rase_vnext_b2_capture_smoke.py \
  --manifest "$MANIFEST" --output-dir "$OUT" --capture-dir "$CAPTURE" \
  --output "$OUT/B2_VERDICT.json"

"$PY" - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = [
    root / "manifest.bound.json", root / "collection_report.json",
    root / "branches.jsonl", root / "B2_VERDICT.json",
]
lines = []
for path in files + sorted((root / "captures").glob("*")):
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
(root / "HASHES.sha256").write_text("\n".join(lines) + "\n")
verdict = json.loads((root / "B2_VERDICT.json").read_text())
(root / "COMPLETE.json").write_text(json.dumps({
    "status": "COMPLETE", "verdict": verdict["status"],
    "scientific_scope": "PARITY_ONLY_NOT_AN_EFFECT_RESULT",
}, indent=2, sort_keys=True) + "\n")
PY

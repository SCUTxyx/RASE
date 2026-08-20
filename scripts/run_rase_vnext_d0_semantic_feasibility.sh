#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RASE

PY=/root/autodl-tmp/envs/smolvla/bin/python
B2=runs/rase_vnext/b2_capture_smoke_v1
MANIFEST=runs/rase_vnext/frozen/b2_capture_smoke_manifest_v1.json
OUT=runs/rase_vnext/d0_semantic_feasibility_v3
mkdir -p "$OUT"

"$PY" - "$B2/B2_VERDICT.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit("B2 verdict is missing; D0 remains locked")
status = json.loads(path.read_text()).get("status")
if status != "B2_CAPTURE_PASS":
    raise SystemExit(f"B2 verdict is {status!r}; D0 remains locked")
PY

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
"$PY" -u scripts/collect_rase_vnext_d0_semantic_feasibility.py \
  --manifest "$MANIFEST" \
  --capture-dir "$B2/captures" \
  --output-dir "$OUT" \
  --policy-path ckpts/pi0fast_libero \
  --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 \
  --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e

"$PY" - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
required = [root / "PROTOCOL.json", root / "SUMMARY.json"]
required.extend(sorted((root / "trials").glob("*.json")))
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"D0 outputs missing: {missing}")
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}" for path in required]
(root / "HASHES.sha256").write_text("\n".join(lines) + "\n")
summary = json.loads((root / "SUMMARY.json").read_text())
(root / "COMPLETE.json").write_text(json.dumps({
    "status": "COMPLETE",
    "verdict": summary["status"],
    "scientific_scope": "A_PARTIAL_PI0FAST_FEASIBILITY_NOT_D_GATE",
}, indent=2, sort_keys=True) + "\n")
PY

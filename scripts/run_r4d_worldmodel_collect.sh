#!/usr/bin/env bash
# R4-D world-model window collection (all 4 suites).
# Mirrors run_full_r4d_collect.sh but uses collect_r4d_worldmodel_windows.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/pre_a3_recovery_duration120.yaml}"
KEYS="${KEYS:-runs/rase_pre_a3_keys120_v1.json}"
DESIGN="${DESIGN:-runs/pre_c0_r4/r4d_train_design.json}"
OUTPUT="${OUTPUT:-runs/pre_c0_r4/worldmodel_windows}"
LOG="${LOG:-runs/pre_c0_r4/worldmodel_windows.log}"
WINDOW="${WINDOW:-8}"
STRIDE="${STRIDE:-4}"
MAX_WINDOWS="${MAX_WINDOWS:-20}"
MAX_STATES="${MAX_STATES:-0}"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
BASE_PY="${BASE_PY:-${CONDA_ROOT}/bin/python}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
OFT_ENV="${OFT_ENV:-oft}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/root/autodl-tmp/src/LIBERO-plus}"
PYTHONPATH_OFT="${PYTHONPATH_OFT:-/root/autodl-tmp/src/openvla-oft}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
SERVER_CUDA="${SERVER_CUDA:-0}"
CLIENT_CUDA="${CLIENT_CUDA:-0}"

if [[ ! -f "$KEYS" || ! -f "$DESIGN" ]]; then
  echo "ERROR: missing frozen keys or R4-D design" >&2
  exit 1
fi

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

SUITE_SHORTS=(spatial object goal 10)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"

  echo "===== Suite: ${suite} ($short) ====="

  # Start OFT server for this suite
  (
    conda activate "$OFT_ENV"
    export CUDA_VISIBLE_DEVICES="$SERVER_CUDA"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONPATH="${PYTHONPATH_OFT}:${PYTHONPATH:-}"
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "runs/pre_c0_r4/oft_server_${short}_wm.log" 2>&1
  ) &
  server_pid=$!

  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  ready=0
  for _ in $(seq 1 60); do
    if conda run -n "$SMOLVLA_ENV" python scripts/probe_oracle.py \
      --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server did not become ready for $suite" >&2
    exit 1
  fi

  (
    conda activate "$SMOLVLA_ENV"
    export CUDA_VISIBLE_DEVICES="$CLIENT_CUDA"
    export MUJOCO_EGL_DEVICE_ID="$CLIENT_CUDA"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_PLUS_ROOT
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    python -u scripts/collect_r4d_worldmodel_windows.py \
      --config "$CONFIG" \
      --design "$DESIGN" \
      --suite "$suite" \
      --endpoint "$ENDPOINT" \
      --output-dir "$OUTPUT/suite_${short}" \
      --window "$WINDOW" \
      --stride "$STRIDE" \
      --max-windows "$MAX_WINDOWS" \
      --max-states "$MAX_STATES"
  ) || {
    echo "ERROR: collection failed for $suite"
    cleanup
    trap - EXIT
    exit 1
  }

  cleanup
  trap - EXIT
  sleep 2
done

echo ""
echo "===== Merging all suites ====="
"$BASE_PY" - "$OUTPUT" <<'PY'
import glob, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reports = [json.load(open(p)) for p in sorted(glob.glob(str(root / "suite_*" / "report.json")))]
if not reports:
    raise SystemExit("no R4-D world-model reports found")

rows = []
for path in sorted(glob.glob(str(root / "suite_*" / "worldmodel_windows_*.jsonl"))):
    rows.extend(json.loads(line) for line in open(path) if line.strip())
out = root / "worldmodel_windows.jsonl"
out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

summary = {
    "schema_version": "rase-pre-c0-r4d-worldmodel-merged/v1",
    "n_suites": len(reports),
    "n_states": sum(r["n_states"] for r in reports),
    "n_windows": len(rows),
    "window": reports[0]["window"] if reports else None,
    "stride": reports[0]["stride"] if reports else None,
    "vla_name": reports[0]["vla_name"] if reports else None,
    "suite_reports": reports,
    "output": str(out.resolve()),
}
(root / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo ""
echo "===== R4-D world-model collection complete ====="

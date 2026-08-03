#!/usr/bin/env bash
# Conditional, resumable causal attribution for W7 held-out OFT-only states.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
SMOLVLA_ENV="${SMOLVLA_ENV:-smolvla}"
CFG="${CFG:-configs/ngc_w7_heldout24_screen.yaml}"
MATRIX="${MATRIX:-runs/ngc_w7_heldout24_policy_matrix.json}"
STATE_KEYS_JSON="${STATE_KEYS_JSON:-runs/ngc_w7_heldout24_oft_only_state_keys.json}"
CANDIDATES_DIR="${CANDIDATES_DIR:-runs/ngc_w7_heldout24_candidates_t07}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ngc_w7_heldout24_prefix_ablation}"
TAG="${TAG:-causal}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/ngc_w7_heldout24_prefix_ablation.json}"
OUTPUT_MD="${OUTPUT_MD:-runs/ngc_w7_heldout24_prefix_ablation.md}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"

mkdir -p runs
exec 9>runs/ngc_w7_heldout24_attribution.lock
if ! flock -n 9; then
  echo "ERROR: another W7 held-out attribution runner is active" >&2
  exit 1
fi

if ! [[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: WAIT_SECONDS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_WAIT_SECONDS must be a positive integer" >&2
  exit 1
fi

elapsed=0
matrix_complete() {
  [[ -f "$MATRIX" ]] && python - "$MATRIX" >/dev/null 2>&1 <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
}

while ! matrix_complete; do
  if (( elapsed >= MAX_WAIT_SECONDS )); then
    echo "ERROR: timed out waiting for complete policy matrix: $MATRIX" >&2
    exit 2
  fi
  echo "WAIT_POLICY_MATRIX elapsed=${elapsed}s path=${MATRIX}"
  sleep "$WAIT_SECONDS"
  elapsed=$((elapsed + WAIT_SECONDS))
done

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$SMOLVLA_ENV"

readarray -t matrix_info < <(
  python - "$MATRIX" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
aliases = {
    "Spatial": "spatial", "libero_spatial": "spatial",
    "Object": "object", "libero_object": "object",
    "Goal": "goal", "libero_goal": "goal",
    "Long": "10", "libero_10": "10",
}
rows = [
    row for row in payload.get("per_state", [])
    if row.get("state_pair_label") == "oft_only"
]
suites = {aliases[str(row.get("suite"))] for row in rows}
ordered = [name for name in ("spatial", "object", "goal", "10") if name in suites]
print(len(rows))
print(",".join(ordered))
PY
)
n_states="${matrix_info[0]}"
suite_shorts="${matrix_info[1]:-}"

if [[ "$n_states" == "0" ]]; then
  echo "NO_OFT_ONLY_STATES attribution_skipped=true"
  exit 0
fi
if [[ -z "$suite_shorts" ]]; then
  echo "ERROR: OFT-only states found but no suites resolved" >&2
  exit 1
fi

python scripts/export_policy_matrix_split_keys.py \
  --matrix "$MATRIX" \
  --label oft_only \
  --output "$STATE_KEYS_JSON"

echo "ATTRIBUTION_START n_states=${n_states} suites=${suite_shorts}"
OUTPUT_PREFIX="$OUTPUT_PREFIX" \
STATE_KEYS_JSON="$STATE_KEYS_JSON" \
CANDIDATES_DIR="$CANDIDATES_DIR" \
OFT_RUNNER=prefix-ablation \
OFT_SUITE_SHORTS="$suite_shorts" \
./scripts/run_oft_verify_suites.sh "$CFG" "$TAG"

summary_args=()
IFS=',' read -r -a selected_suites <<< "$suite_shorts"
for short in "${selected_suites[@]}"; do
  case "$short" in
    spatial) suite=libero_spatial ;;
    object) suite=libero_object ;;
    goal) suite=libero_goal ;;
    10) suite=libero_10 ;;
    *) echo "ERROR: unresolved suite short: $short" >&2; exit 1 ;;
  esac
  summary_args+=(
    --summary
    "${suite}=runs/${OUTPUT_PREFIX}_${short}_${TAG}/summary.json"
  )
done

conda activate "$SMOLVLA_ENV"
python scripts/summarize_prefix_ablation.py \
  --state-keys "$STATE_KEYS_JSON" \
  "${summary_args[@]}" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"

echo "W7_HELDOUT_ATTRIBUTION_DONE n_states=${n_states}"

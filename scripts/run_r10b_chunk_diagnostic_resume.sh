#!/usr/bin/env bash
# Resume the frozen 54-trajectory diagnostic and audit only after full completion.
set -euo pipefail
cd /root/autodl-tmp/RASE

SESSION=${R10B_SESSION:-r10b_chunk_full}
MANIFEST=${R10B_MANIFEST:-runs/pre_c0_r10/r10b_oft_trace_diagnostic_manifest_v1.json}
OUT=${R10B_OUT:-runs/pre_c0_r10/r10b_chunk_input_diagnostic_collect_v1}
AUDIT=${R10B_AUDIT:-runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json}
LOG=${R10B_LOG:-runs/pre_c0_r10/r10b_chunk_input_diagnostic.log}
PY=/root/autodl-tmp/envs/smolvla/bin/python

test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
"$PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get("status") == "frozen_diagnostic", "manifest is not frozen_diagnostic"
assert m.get("expected_groups") == 18 and m.get("expected_trajectories") == 54
assert len(m.get("records", [])) == 18
PY

if test -f "$AUDIT"; then
  echo "audit already exists: $AUDIT"
  "$PY" scripts/status_r10b_chunk_diagnostic.py
  exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already running: $SESSION"
  "$PY" scripts/status_r10b_chunk_diagnostic.py
  exit 0
fi

if [[ ${RASE_DRY_RUN:-0} != 1 ]]; then
  free_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  if (( free_kb < 20 * 1024 * 1024 )); then
    echo "refusing to start: /root/autodl-tmp has less than 20 GiB free" >&2
    exit 20
  fi
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "refusing to start: no visible NVIDIA GPU" >&2
    exit 21
  fi
fi

job=$(printf '%q ' env \
  R10B_MANIFEST="$MANIFEST" R10B_OUT="$OUT" R10B_REPLICA_LIMIT=3 \
  R10B_RECORD_OFT_TRACE_HASH=1 R10B_RECORD_OFT_CHUNK_TRACE=1 \
  bash scripts/run_r10b_case_control_collect.sh)
audit=$(printf '%q ' "$PY" scripts/audit_r10b_chunk_input_divergence.py \
  --manifest "$MANIFEST" --collect-root "$OUT" --output "$AUDIT")
command="set -o pipefail; ${job} 2>&1 | tee -a $(printf '%q' "$LOG"); test -f $(printf '%q' "$OUT/COMPLETE"); ${audit} 2>&1 | tee -a $(printf '%q' "$LOG")"

if [[ ${RASE_DRY_RUN:-0} == 1 ]]; then
  echo "tmux new-session -d -s $SESSION bash -lc $(printf '%q' "$command")"
  exit 0
fi
tmux new-session -d -s "$SESSION" bash -lc "$command"
echo "started resumable session: $SESSION"
echo "progress: $PY scripts/status_r10b_chunk_diagnostic.py --watch --interval 30"
echo "logs: tmux attach -t $SESSION"

#!/usr/bin/env bash
# Evaluate revised adapter with R(k) handover grid vs C1.1 baseline curves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"
PROTOCOL="${PROTOCOL:-artifacts/pre_c1/pre_c1_2_protocol_lock.yaml}"
CONFIG="${CONFIG:-configs/collect_pre_c0_deviation_pilot24.json}"
ADAPTER="${ADAPTER:-runs/rase_pre_c1_2_lora_revised_r1_v1/adapter_final}"
KEYS="${KEYS:-artifacts/pre_c1/pre_c1_2_locked_9_failure_keys.json}"
FAILURES="${FAILURES:-runs/rase_pre_c0_same_policy_pilot48_v1}"
ENDPOINT="${ENDPOINT:-tcp://127.0.0.1:5555}"
OUT="${OUT:-runs/rase_pre_c1_2_r0_recoverability_revised_v1}"
LOG_DIR="${LOG_DIR:-runs/rase_pre_c1_2_pipeline_logs}"

mkdir -p "$OUT" "$LOG_DIR"

kill_oft() {
  pkill -f 'python -m rase.oracle.server' 2>/dev/null || true
  sleep 2
}

start_oft() {
  local short="$1" local suite="$2" local ckpt="$3"
  kill_oft
  echo "=== Start OFT server suite=${suite} ==="
  (
    conda activate oft
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/root/autodl-tmp/src/openvla-oft:${PYTHONPATH:-}
    export RASE_OFT_CHECKPOINT="$ROOT/$ckpt"
    export RASE_OFT_SUITE="$suite"
    exec python -m rase.oracle.server --endpoint "$ENDPOINT" \
      --adapter rase.oracle.openvla_oft_adapter:create_adapter \
      > "$LOG_DIR/oft_server_revised_${short}.log" 2>&1
  ) &
  local ready=0
  for _ in $(seq 1 150); do
    if "$PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite "$suite" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: OFT server not ready for $suite" >&2
    exit 1
  fi
  echo "OFT ready suite=$suite"
}

SUITE_SHORTS=(spatial object goal 10)
SUITE_LABELS=(Spatial Object Goal Long)
SUITE_NAMES=(libero_spatial libero_object libero_goal libero_10)
CKPTS=(ckpts/oft_spatial ckpts/oft_object ckpts/oft_goal ckpts/oft_10)

for idx in "${!SUITE_SHORTS[@]}"; do
  short="${SUITE_SHORTS[$idx]}"
  label="${SUITE_LABELS[$idx]}"
  suite="${SUITE_NAMES[$idx]}"
  ckpt="${CKPTS[$idx]}"
  n_keys=$("$PY" -c "import json;from pathlib import Path;d=json.loads(Path('$KEYS').read_text());print(len(d.get('by_suite',{}).get('$label',[])))")
  if [[ "$n_keys" == "0" ]]; then
    echo "SKIP recoverability suite=$label"
    continue
  fi
  start_oft "$short" "$suite" "$ckpt"
  "$PY" scripts/eval_pre_c1_2_student_prefix_teacher_handover.py \
    --protocol-lock "$PROTOCOL" \
    --config "$CONFIG" \
    --adapter-dir "$ADAPTER" \
    --failure-rollout-dir "$FAILURES" \
    --state-keys-json "$KEYS" \
    --suite "$label" \
    --endpoint "$ENDPOINT" \
    --output-dir "$OUT" \
    --resume
done
kill_oft

echo "=== Re-aggregate summary from all trial files $(date -Is) ==="
"$PY" -c "
import json, numpy as np
from pathlib import Path
out=Path('$OUT')
trials=[json.loads(p.read_text()) for p in sorted((out/'trials').glob('*.json'))]
ks=sorted({int(t['k']) for t in trials})
def rate(arm,k):
    s=[t for t in trials if t.get('arm')==arm and int(t.get('k',-1))==int(k)]
    return float(np.mean([1.0 if t.get('teacher_success') else 0.0 for t in s])) if s else float('nan')
curves={'oft_k0':rate('oft_k0',0),'oft_replan_k1':rate('oft_replan_k1',1),
        'base':{int(k):rate('base',k) for k in ks if int(k)>0},
        'adapted':{int(k):rate('adapted',k) for k in ks if int(k)>0}}
curves['base'][0]=curves['oft_k0']; curves['adapted'][0]=curves['oft_k0']
summary={'schema_version':'rase-pre-c1-2-r0-recoverability/v1','adapter':'$ADAPTER',
         'n_anchors':len({t['state_key'] for t in trials}),'n_trials':len(trials),'ks':ks,
         'curves':curves,'R_oft_k0':curves['oft_k0'],'R_oft_replan_1':curves['oft_replan_k1'],
         'coverage':{'anchors':sorted({t['state_key'] for t in trials}),'arms':sorted({t['arm'] for t in trials})},
         'compared_to_c11_adapter':'runs/rase_pre_c1_2_r0_recoverability_v1/summary.json'}
(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
with (out/'trials.jsonl').open('w') as f:
    for t in trials: f.write(json.dumps(t,sort_keys=True)+'\n')
print(json.dumps(summary,sort_keys=True))
print('REVISED_RK_EVAL_DONE')
"

# Compare against C1.1 baseline
echo "=== Comparison vs C1.1 baseline ==="
"$PY" -c "
import json
from pathlib import Path
rev=json.loads(Path('runs/rase_pre_c1_2_r0_recoverability_revised_v1/summary.json').read_text())
c11=json.loads(Path('runs/rase_pre_c1_2_r0_recoverability_v1/summary.json').read_text())
print('=== R(k) comparison ===')
print(f\"{'k':>4s}  {'base(C1.1)':>12s}  {'adapt(C1.1)':>12s}  {'adapt(rev)':>12s}  {'rev-C1.1':>10s}\")
for k in sorted(set(c11['curves']['base'].keys()) | set(rev['curves']['adapted'].keys())):
    kb=float(c11['curves']['base'].get(str(k), c11['curves']['base'].get(int(k), float('nan'))))
    kc=float(c11['curves']['adapted'].get(str(k), c11['curves']['adapted'].get(int(k), float('nan'))))
    kr=float(rev['curves']['adapted'].get(str(k), rev['curves']['adapted'].get(int(k), float('nan'))))
    delta=kr-kc
    print(f'{int(k):>4d}  {kb:>12.3f}  {kc:>12.3f}  {kr:>12.3f}  {delta:>+10.3f}')
print()
# Check primary gates
r1c=float(c11['curves']['adapted'].get(1,0)); r1r=float(rev['curves']['adapted'].get(1,0))
r4c=float(c11['curves']['adapted'].get(4,0)); r4r=float(rev['curves']['adapted'].get(4,0))
print(f'R(1) revised: {r1r:.3f}  C1.1: {r1c:.3f}  gain: {r1r-r1c:+.3f}  >base={r1r>float(c11[\"curves\"][\"base\"][1])}')
print(f'R(4) revised: {r4r:.3f}  C1.1: {r4c:.3f}  gain: {r4r-r4c:+.3f}  >base={r4r>float(c11[\"curves\"][\"base\"][4])}')
print()
print('=== Gate status ===')
print(f'R_adapted(1) > R_base(1): {r1r > float(c11[\"curves\"][\"base\"][1]) or r1r == float(c11[\"curves\"][\"base\"][1])}')
print(f'R_adapted(4) > R_base(4): {r4r > float(c11[\"curves\"][\"base\"][4]) or r4r == float(c11[\"curves\"][\"base\"][4])}')
print(f'Revised > C1.1 at k=1: {r1r > r1c or r1r == r1c}')
print(f'Revised > C1.1 at k=4: {r4r > r4c or r4r == r4c}')
print(f'Terminal 8pp gate: not yet (final gate only)')
"

echo "=== REVISED RK DONE $(date -Is) ==="
echo "Adapter: $ADAPTER"
echo "Summary: $OUT/summary.json"
echo "Trials: $OUT/trials/*.json"
echo "Compare: runs/rase_pre_c1_2_r0_recoverability_v1/summary.json"
echo "Next: if short Rk improves, continue short corrective loop; terminal 8pp is final gate only"

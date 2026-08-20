#!/usr/bin/env bash
# Chain PRE-A3 staged live-duration: wait for train → val → (GO?) → test → method gate.
# Waits on log completion markers (not shell command text in tmux panes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/envs/smolvla/bin/python}"

count_live() {
  local log="$1"
  if [[ -f "$log" ]]; then
    grep -c 'LIVE_DURATION state=' "$log" || true
  else
    echo 0
  fi
}

wait_duration_done() {
  local log="$1"
  local label="$2"
  local gone_polls=0
  echo "[$(date -Iseconds)] waiting for PRE_A3_DURATION_DONE in $log ($label)"
  while true; do
    if [[ -f "$log" ]] && grep -q 'PRE_A3_DURATION_DONE' "$log"; then
      echo "[$(date -Iseconds)] $label DONE"
      tail -30 "$log"
      return 0
    fi
    if pgrep -f 'run_pre_a3_recovery_duration.sh' >/dev/null 2>&1 \
       || pgrep -f 'rollout_live_oft_duration_to_smol.py' >/dev/null 2>&1; then
      gone_polls=0
    else
      gone_polls=$((gone_polls + 1))
      # Require prior progress + ~6 min of no runner before declaring failure.
      if [[ "$(count_live "$log")" -gt 0 && "$gone_polls" -ge 3 ]]; then
        echo "[$(date -Iseconds)] $label processes gone without DONE marker" >&2
        tail -50 "$log" || true
        return 1
      fi
    fi
    echo "[$(date -Iseconds)] $label live_done=$(count_live "$log") gone_polls=$gone_polls"
    sleep 120
  done
}

launch_split() {
  local window="$1"
  local split="$2"
  local output="$3"
  local analysis="$4"
  local log="$5"
  local marker="$6"
  # kill old window if leftover
  tmux kill-window -t "0:$window" 2>/dev/null || true
  tmux new-window -t 0: -n "$window" -c "$ROOT"
  # Write a tiny launcher so the marker string is not present until completion.
  local launcher
  launcher="$(mktemp /tmp/pre_a3_${split}_XXXXXX.sh)"
  cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
KEYS=runs/rase_pre_a3_keys120_v1.json \\
CONFIG=configs/pre_a3_recovery_duration120.yaml \\
OUTPUT=$output \\
ANALYSIS=$analysis \\
LOG=$log \\
SPLIT_FILTER=$split \\
FRESH_RUN=1 \\
./scripts/run_pre_a3_recovery_duration.sh
ec=\$?
echo "${marker}:\${ec}"
exit "\$ec"
EOF
  chmod +x "$launcher"
  tmux send-keys -t "0:$window" "bash $launcher" Enter
}

# --- Step 2: wait for already-running train ---
TRAIN_LOG=runs/rase_pre_a3_recovery_duration120_train_v1.log
wait_duration_done "$TRAIN_LOG" train

# Sanity: train summary exists
test -f runs/rase_pre_a3_recovery_duration120_train_v1/summary.json
live_n="$(count_live "$TRAIN_LOG")"
echo "train live lines=$live_n"

cat > progress/2026-08-04_rase_pre_a3_train72_live_duration.md <<EOF
# PRE-A3 train72 live duration — 2026-08-04

## Status
**Step 2 complete** (plumbing).

## Outputs
- run: \`runs/rase_pre_a3_recovery_duration120_train_v1/\`
- audit: \`runs/rase_pre_a3_recovery_duration_audit120_train_v1/\`
- LIVE_DURATION lines: ${live_n}

## Note
Train-only results are for plumbing / resume / cost checks. Not confirmatory.
Do not retune h / cohort / gates from train.

## Next
Val24 live duration go/no-go.
EOF

# --- Step 3: val ---
launch_split pre-a3-val val \
  runs/rase_pre_a3_recovery_duration120_val_v1 \
  runs/rase_pre_a3_recovery_duration_audit120_val_v1 \
  runs/rase_pre_a3_recovery_duration120_val_v1.log \
  VAL_FINISHED_EXIT
wait_duration_done runs/rase_pre_a3_recovery_duration120_val_v1.log val
test -f runs/rase_pre_a3_recovery_duration120_val_v1/summary.json

"$PY" - <<'PY'
import json
from pathlib import Path

audit_dir = Path("runs/rase_pre_a3_recovery_duration_audit120_val_v1")
candidates = [audit_dir / "audit_val.json", *sorted(audit_dir.glob("audit*.json"))]
payload = None
used = None
for p in candidates:
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
        used = p
        break
if payload is None:
    raise SystemExit("missing val audit")

ok = payload.get("gate_pass")
if ok is None and isinstance(payload.get("pass_conditions"), dict):
    ok = all(bool(v) for v in payload["pass_conditions"].values())
decision = "GO" if ok else "NOGO"
Path("runs/rase_pre_a3_val_go_nogo.txt").write_text(decision + "\n", encoding="utf-8")
Path("runs/rase_pre_a3_val_gating_snapshot.json").write_text(
    json.dumps(
        {
            "audit": str(used),
            "decision": decision,
            "gate_pass": payload.get("gate_pass"),
            "pass_conditions": payload.get("pass_conditions"),
            "status": payload.get("status"),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"decision": decision, "audit": str(used), "gate_pass": payload.get("gate_pass")}, sort_keys=True))
PY

cat > progress/2026-08-04_rase_pre_a3_val24_live_duration.md <<'EOF'
# PRE-A3 val24 live duration — 2026-08-04

## Status
**Step 3 complete.**

## Outputs
- run: `runs/rase_pre_a3_recovery_duration120_val_v1/`
- audit: `runs/rase_pre_a3_recovery_duration_audit120_val_v1/`
- go/no-go: `runs/rase_pre_a3_val_go_nogo.txt`

## Constraint
No h / cohort / gate retune after looking at val; go/no-go only for blind test.
EOF

decision="$(tr -d '[:space:]' < runs/rase_pre_a3_val_go_nogo.txt)"
if [[ "$decision" != "GO" ]]; then
  echo "VAL NO-GO — freeze benchmark_diagnosis_only; skip test unblinding"
  "$PY" - <<'PY'
import json
from pathlib import Path
Path("runs/rase_pre_a3_method_gate_confirmatory_v1.json").write_text(
    json.dumps(
        {
            "status": "FAIL",
            "track": "benchmark_diagnosis_only",
            "reason": "val go/no-go failed; hidden test not unblinded",
            "world_model_gate": "closed",
            "critic_gate": "closed",
            "pre_b_allowed": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print("wrote method_gate FAIL without test")
PY
  cat > progress/2026-08-04_rase_pre_a3_test24_and_method_gate.md <<'EOF'
# PRE-A3 method gate — 2026-08-04

## Status
**FAIL / NO-GO at val.** Hidden test not unblinded.
Track remains `benchmark_diagnosis_only`. PRE-B closed.
EOF
  echo CHAIN_DONE
  exit 0
fi

# --- Step 4: blind test ---
launch_split pre-a3-test test \
  runs/rase_pre_a3_recovery_duration120_test_v1 \
  runs/rase_pre_a3_recovery_duration_audit120_test_v1 \
  runs/rase_pre_a3_recovery_duration120_test_v1.log \
  TEST_FINISHED_EXIT
wait_duration_done runs/rase_pre_a3_recovery_duration120_test_v1.log test
test -f runs/rase_pre_a3_recovery_duration120_test_v1/summary.json

VAL_AUDIT="$(ls runs/rase_pre_a3_recovery_duration_audit120_val_v1/audit_val.json 2>/dev/null || ls runs/rase_pre_a3_recovery_duration_audit120_val_v1/audit*.json | head -1)"
TEST_AUDIT="$(ls runs/rase_pre_a3_recovery_duration_audit120_test_v1/audit_test.json 2>/dev/null || ls runs/rase_pre_a3_recovery_duration_audit120_test_v1/audit*.json | head -1)"
echo "VAL_AUDIT=$VAL_AUDIT TEST_AUDIT=$TEST_AUDIT"

"$PY" scripts/decide_pre_a3_method_gate.py \
  --hidden-audit "$TEST_AUDIT" \
  --val-audit "$VAL_AUDIT" \
  --output runs/rase_pre_a3_method_gate_confirmatory_v1.json

cat > progress/2026-08-04_rase_pre_a3_test24_and_method_gate.md <<'EOF'
# PRE-A3 test24 + confirmatory method gate — 2026-08-04

## Status
**Step 4 complete.**

## Outputs
- test run: `runs/rase_pre_a3_recovery_duration120_test_v1/`
- test audit: `runs/rase_pre_a3_recovery_duration_audit120_test_v1/`
- method gate: `runs/rase_pre_a3_method_gate_confirmatory_v1.json`

## Constraint
No post-hoc h / cohort / gate edits after test unblinding.
EOF

echo CHAIN_DONE
cat runs/rase_pre_a3_method_gate_confirmatory_v1.json

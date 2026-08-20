#!/usr/bin/env bash
# R6-B1.0 diagnostic: bisect the in-loop boundary bookkeeping to find which
# operation perturbs the frozen R6-A source trajectory.
#
# For each (policy, state) pair run the four bookkeeping modes in independent
# processes and compare env_steps/success and the per-step source action trace.
set -euo pipefail
cd /root/autodl-tmp/RASE

OFT_PY=/root/autodl-tmp/envs/oft/bin/python
VLA_PY=/root/autodl-tmp/envs/smolvla/bin/python
ENDPOINT=tcp://127.0.0.1:5555
OUT=runs/pre_c0_r6/r6b1_bisect_v1
mkdir -p "$OUT"

# One suite-specific OFT server serves all server-backed runs (all Spatial).
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/root/autodl-tmp/src/openvla-oft:$PWD" \
RASE_OFT_CHECKPOINT="$PWD/ckpts/oft_spatial" RASE_OFT_SUITE=libero_spatial \
"$OFT_PY" -m rase.oracle.server --endpoint "$ENDPOINT" \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter > "$OUT/oft_server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  if "$VLA_PY" scripts/probe_oracle.py --endpoint "$ENDPOINT" --expect-suite libero_spatial >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == 1 ]]
echo "R6B1_BISECT OFT server ready pid=$server_pid"

run_one() {
  local label="$1" policy="$2" state="$3" mode="$4" rep="$5" use_server="$6"
  local suite
  suite=libero_spatial
  local -a policy_args
  if [[ "$policy" == pi0fast_libero ]]; then
    policy_args=(--policy-path ckpts/pi0fast_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46 --action-tokenizer-path ckpts/pi0fast_action_tokenizer_79ae83e)
  else
    policy_args=(--policy-path ckpts/pi05_libero --tokenizer-path ckpts/paligemma_tokenizer_35e4f46)
  fi
  local oracle_args=()
  if [[ "$use_server" == 1 ]]; then
    oracle_args=(--endpoint "$ENDPOINT")
  else
    oracle_args=(--no-oracle)
  fi
  local dir="$OUT/$label/$mode/$use_server/rep$rep"
  mkdir -p "$dir"
  echo "R6B1_BISECT $label policy=$policy state=$state mode=$mode rep=$rep server=$use_server"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_PLUS_ROOT=/root/autodl-tmp/src/LIBERO-plus HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLA_PY" -u scripts/collect_r6b1_dynamic_boundaries.py \
    --initial-keys runs/rase_ui_phase1a_replacement48_initial_keys_v2.json \
    --policy-id "$policy" "${policy_args[@]}" --suite "$suite" --seed-index 0 \
    "${oracle_args[@]}" --output-dir "$dir" \
    --boundary 0 --state-key "$state" \
    --bookkeeping-mode "$mode" --skip-oft --debug
  echo "R6B1_BISECT complete $label/$mode/$use_server/rep$rep"
}

# Pi0Fast success state (Spatial): R6-A 116 steps; failing B1 run produced 149.
PIF0=sp1_0660d272e7256c6b204caf666e94c875
# Pi0.5 success state (Spatial): R6-A 77 steps (seed 0).
PI050=sp1_4e5fbfad8277abf157439e0b7a22c06a

for rep in 1 2 3; do
  for mode in none snapshot_only obs_only full; do
    run_one pi0fast_libero pi0fast_libero "$PIF0" "$mode" "$rep" 1
  done
  # no-OFT-server control on the Pi0Fast success state
  run_one pi0fast_libero pi0fast_libero "$PIF0" none "$rep" 0
  run_one pi0fast_libero pi0fast_libero "$PIF0" full "$rep" 0
done

for rep in 1 2 3; do
  for mode in none snapshot_only obs_only full; do
    run_one pi05_libero pi05_libero "$PI050" "$mode" "$rep" 1
  done
done

echo "R6B1_BISECT all runs complete"

"$VLA_PY" scripts/analyze_r6b1_bisect.py \
  --input-root "$OUT" --output "$OUT/summary.json" | tee "$OUT/analysis.txt"

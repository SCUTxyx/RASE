# W2–W3 NGC pilot runbook

This runbook covers candidate artifacts (W2), research-grade forked continuation
(W3), sequential triage, and suite-matched OpenVLA-OFT cross-oracle verify.
GPU jobs are opt-in; contract tests and Monte Carlo calibration need no GPU.

**Do not** treat 2-state / 20-rollout smoke outputs as NGC yield or “candidate
recovery success.” Statistical claims require the stratified 16-state pilot
after calibration freezes the protocol.

## Frozen research protocol (W3)

| Parameter | Value | Notes |
|---|---|---|
| Primary estimator | SmolVLA continuation `temperature=0.5` | Bernoulli samples for Wilson / sequential bounds |
| Cross-oracle | Suite-matched OpenVLA-OFT L1 | Deterministic; one verify per `(state, candidate)` |
| Candidates | `K=8, T=10, action_dim=7, temp=0.7` | Env-space in `.npz`; no second denorm |
| Horizon | Snapshot `env_counters.timestep` | Candidate mid-success ends early |
| `tau` | `0.5` | Main NGC threshold |
| Look schedule | `n1=6 → n2=20` | Replaces invalid legacy `3→10` two-sided early-stop |
| Alpha spending | `0.01 + 0.04` | One-sided 95%; total type-I ≤ 0.05 |
| Set A | ≥3 candidates with lower > τ | |
| Smoke | 2 states / 20 rollouts | Engineering gate only |
| Pilot | 16 states, `4 suites × {camera,robot} × 2` | L3–L5 preferred |

Legacy two-sided `adaptive_sample` (`3→10`) remains for protocol-fidelity tests
only. Production W3 configs use `wilson-onesided-alpha-spend-v1`.

## 1. Preflight (no GPU)

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE
python -m pytest -q
python scripts/calibrate_ngc_statistics.py \
  --config configs/ngc_w3_pilot.yaml \
  --replications 100000 \
  --output runs/ngc_w3_calibration_v1/summary.json
```

Hard gate: calibration `Set C` false-positive rate at `p=0.5` (or the script’s
documented null) must be **≤ 5%**.

Configs: `configs/ngc_w3_smoke.yaml`, `configs/ngc_w3_pilot.yaml`. **Do not**
reuse an old run root after changing pool/candidates/policy fingerprints;
`run_manifest.json` rejects stale resume.

## 2. W2 candidates (if regenerating)

```bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
python -u scripts/generate_pool_candidates.py \
  --config configs/candidates_w2_pilot.json
```

Artifacts: `runs/ngc_w2_candidates_pilot/candidates/{state_key}.npz`, shape
`[8, T, 7]`, temperature 0.7, immutable `policy_hash`.

## 3. OpenVLA-OFT oracle (one suite per GPU)

Download suite checkpoints once:

```bash
conda activate oft
cd /data/data2/yuxuan/RASE
export HF_HOME=/data/data2/yuxuan/hf_cache
for suite in spatial object goal 10; do
  huggingface-cli download "moojink/openvla-7b-oft-finetuned-libero-${suite}" \
    --local-dir "ckpts/oft_${suite}"
done
```

Start **one** suite server per 4090 (example Goal):

```bash
conda activate oft
cd /data/data2/yuxuan/RASE
export CUDA_VISIBLE_DEVICES=0
export RASE_OFT_CHECKPOINT=/data/data2/yuxuan/RASE/ckpts/oft_goal
export RASE_OFT_SUITE=libero_goal
# Ensure official openvla-oft code is importable (install or PYTHONPATH).
python -m rase.oracle.server \
  --endpoint tcp://127.0.0.1:5555 \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter
```

Probe from the collection env:

```bash
conda activate smolvla
python scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite libero_goal
```

RPC wire: `agentview` / `wrist` uint8 `[B,H,W,3]`, proprio `policy_state [B,8]`
or `raw_quat [B,9]`, instructions list; actions env-ready `[B,8,7]`. Client
rebuilds the REQ socket after timeout. Adapter batching is serial with
`max_batch≤2`; do not claim GPU batch-8.

Golden parity (oft env): adapter actions vs official `get_vla_action` should
match with `atol≤1e-4` before forked OFT verify.

## 4. W3 forked rollouts

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
export HF_HOME=/data/data2/yuxuan/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RASE_TOKENIZER_PATH=/data/data2/yuxuan/RASE/ckpts/SmolVLM2-500M-Instruct

# Engineering smoke (≤20 rollouts); not a statistical conclusion.
# Use early-horizon state_keys in ngc_w3_smoke.yaml (W2 pilot keys sit near t=290).
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_smoke.yaml --mode smoke --max-rollouts 20

# After smoke + calibration: stratified SmolVLA primary
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_pilot.yaml --mode smolvla-primary

# Cross-oracle verify (group by suite; restart OFT server per suite)
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_pilot.yaml --mode oft-verify \
  --suite libero_goal --endpoint tcp://127.0.0.1:5555
```

Each rollout: checksum → task bind → restore → derived seed → `policy.reset()` →
candidate chunk → continuation. Scheduler key is `(state, candidate, rollout)`.
Resume is safe only when `run_manifest.json` fingerprints match.

Rebuild triage anytime from durable records:

```python
from rase.collect.scheduler import DiskRolloutScheduler
from rase.collect.triage_report import summarize_run

scheduler = DiskRolloutScheduler("runs/ngc_w3_pilot/scheduler")
print(summarize_run(scheduler, state_keys, k=8, n_first=6, n_total=20))
```

Triage (strict inequalities):

- **Set C**: all eight Wilson upper bounds `< τ`
- **Set A**: ≥3 lower bounds `> τ`
- **Set B**: 1–2 lower bounds `> τ`
- **uncertain**: otherwise

## 5. Acceptance and stop conditions

Before long collection, complete the 20-rollout smoke and record median/p90 wall
time (restore / render / inference breakdown if available), oracle `model_info`,
resolved config, retries, and triage counts.

**After the 16-state `smolvla-primary` pilot**, do **not** expand the pool yet.
First: (1) inspect `summary.json` + remaining-horizon diagnosis (narrow vs
adequate remaining steps); (2) run suite-grouped `oft-verify` with a **separate**
`--output-dir` per suite (`runs/ngc_w3_oft_<suite>`); (3) only then decide on
pool growth or candidate-protocol changes. A pilot that is 16× Set C with zero
raw successes is a scientific finding under the frozen protocol, not a license
to claim global NGC yield is zero.

**Stop** if fork replay is nondeterministic, candidate provenance fails, OFT
`model_info` mismatches the expected suite/checkpoint hash, adapter parity fails,
or retries are exhausted.

After a crash: rerun the same command on the same run root. After a
deterministic input change: use a **new** run root.

### Post-pilot OFT verify (example: spatial)

```bash
# Terminal A (oft env, one suite per GPU)
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/data2/yuxuan/openvla-oft:$PYTHONPATH
export RASE_OFT_CHECKPOINT=$PWD/ckpts/oft_spatial
export RASE_OFT_SUITE=libero_spatial
python -m rase.oracle.server --endpoint tcp://127.0.0.1:5555 \
  --adapter rase.oracle.openvla_oft_adapter:create_adapter

# Terminal B (smolvla env)
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
python scripts/probe_oracle.py --endpoint tcp://127.0.0.1:5555 --expect-suite libero_spatial
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_pilot.yaml --mode oft-verify \
  --suite libero_spatial --endpoint tcp://127.0.0.1:5555 \
  --output-dir runs/ngc_w3_oft_spatial
```

In `oft-verify` mode the CLI **forces** `scheduler` under `--output-dir`
(ignores `scheduler.root` in the YAML) so SmolVLA adaptive records are never
reused or overwritten.

Repeat for `libero_object` / `libero_goal` / `libero_10` with matching
`ckpts/oft_{object,goal,10}` and a **separate** `--output-dir` each time.
Kill the previous OFT server process before reloading another checkpoint
(otherwise GPU OOM). Do not `pkill -f 'rase.oracle.server'` in the same shell
that starts the server (it can kill itself); kill by exact PID.

### ADEQUATE-only pilot (`min_remaining_steps: 100`)

Config: `configs/ngc_w3_pilot_adequate.yaml` (frozen `sample.state_keys`).
Candidates live at `runs/ngc_w3_adequate_candidates/<state_key>.npz` — **no**
nested `candidates/` subdirectory (unlike `runs/ngc_w3_pilot_candidates/candidates/`).

```bash
cd /data/data2/yuxuan/RASE   # required
conda activate smolvla
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
export HF_HOME=/data/data2/yuxuan/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RASE_TOKENIZER_PATH=$PWD/ckpts/SmolVLM2-500M-Instruct

python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_pilot_adequate.yaml \
  --mode smolvla-primary \
  --force-new-run
```

OFT verify on the same config: `--mode oft-verify --suite <suite>` with
`--output-dir runs/ngc_w3_oft_<short>_adequate` (spatial/object/goal/10).

W3 status (2026-07-19): old pilot 16×Set C / 0/768; OFT spatial 8/32; libero_10
0/32; cont-temp ablation **0/192** vs OFT 8/8. ADEQUATE-only pilot: SmolVLA
still 16×C / 0/768 (0 NARROW); OFT spatial **7/32**, object/goal/10 **0/32**.
Do **not** expand the pool yet. See
`progress/2026-07-19_ngc_w3_pilot_adequate.md`.

### Continuation-temperature ablation

```bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
# dirs: t00→0.0, t02→0.2, t05→0.5, t10→1.0
python -u scripts/rollout_pool_candidates.py \
  --config configs/ngc_w3_cont_ablation.yaml \
  --mode smolvla-primary \
  --continuation-temperature 0.2 \
  --output-dir runs/ngc_w3_cont_ablation_t02 \
  --force-new-run
```

After a SIGFPE/exit 136 crash, rerun the **same** `--output-dir` without
`--force-new-run` to resume. Aggregate: `runs/ngc_w3_cont_ablation_summary.json`.


# W9 clean-control coverage gate（preregistered 140 episodes）

## Status

**Blocked at Gate W9-B (exit 2, as designed).** The preregistered clean
collection ran to the 140-episode hard stop. Suite×`t0_bin` coverage for a
32-state success control freeze is incomplete. Direct-OFT / clean32 labeling
and selector training were **not** started.

| 项 | 内容 |
|---|---|
| 日期 | 2026-07-29（CST） |
| 入口 | `scripts/run_w9_clean_selector_pipeline.sh` |
| 日志 | `runs/ngc_w9_clean_selector_pipeline.log` |
| Git SHA | `ea7ad40` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| 计划文档 | [W9 pipeline](2026-07-29_w9_clean_selector_pipeline.md) |

## Research question

Can SmolVLA under explicit clean controls (`perturb_dim=clean`,
`perturb_sub=none`, `level=0`) produce enough **distinct success episode
groups** to freeze a 32-state balanced control cohort
(4 suites × {early, mid} × 4), so that a three-action selector can learn
clean-regret rather than collapsing to always-escalate?

## Checkpoint / config

- Adapter: `rase.collect.lerobot_libero_plus_adapter:make_adapter`
- Collect configs: `configs/collect_w9_clean_controls.json`,
  `configs/ngc_w9_clean_controls.yaml`
- Policy: frozen SmolVLA (same W9 pipeline path as failure arms)
- Seeds: collect batches `20260730` / `20260731` / `20260732`
- Quotas: clean only; suites Spatial / Object / Goal / Long balanced per batch
- Preregistered schedule: 60 + optional 40 + optional 40 (**max 140**);
  no adaptive resampling beyond this limit

Mid-run incident: fingerprint v1 raised a false-positive
`UnsupportedEnvironmentError` on mutable mujoco XML. Hotfix
`rase-task-identity/v2` landed; pipeline resumed and finished the remaining
preregistered episodes. Already-published pool states were retained.

## What completed before the gate

| Stage | Result |
|---|---|
| Direct Smol on W7 failure24 | **0/24** hits (`runs/ngc_w9_direct_smol_failure24/summary.json`) |
| Clean collect batch 60 (`20260730`) | 54 new ep (6 skipped already-in-pool): **7 success / 47 failure**; 847 states |
| Clean collect batch 40 (`20260731`) | **4 success / 36 failure**; 650 states |
| Clean collect batch 40 (`20260732`) | **1 success / 39 failure**; 678 states |
| Freeze clean32 | **skipped** (`coverage_complete: false`, `n_states: 0`) |
| Direct OFT / Smol on clean32 | **not run** |
| Selector export / ridge | **not run** |

## Aggregate pool result (authoritative)

Pool: `pool/ngc_w9_clean_controls/`

| 量 | 值 |
|---:|---|
| Episodes in pool | **138** |
| Success / failure episodes | **10 / 128** |
| Episode success rate | **7.2%** |
| States retained | **2261** |
| Success states (any suite/t0) | **26** |

### Success by suite (episode-level)

| Suite | Episodes | Successes |
|---|---:|---:|
| Spatial (`libero_spatial`) | 33 | **9** |
| Long (`libero_10`) | 35 | **1** |
| Object (`libero_object`) | 35 | **0** |
| Goal (`libero_goal`) | 35 | **0** |

Almost all clean successes are Spatial (tasks `000005`–`000010` dominate).
Object and Goal are **0/35**. The single Long success has `step=48`, which
falls **outside** the preregistered early `[0,10)` / mid `[10,40)` bins
(`outside_t0_bins=1` in the coverage audit).

### Coverage freeze inventory

Target: `per_cell=4`, strata = suite × `t0_bin`, distinct success episodes,
`dims=[clean]`, `levels=[0]`.

| Cell | Available `n` | Missing to 4 |
|---|---:|---:|
| Spatial / early | 8 | 0 |
| Spatial / mid | 5 | 0 |
| Object / early | 0 | 4 |
| Object / mid | 0 | 4 |
| Goal / early | 0 | 4 |
| Goal / mid | 0 | 4 |
| Long / early | 0 | 4 |
| Long / mid | 0 | 4 |

**6/8 cells empty.** Gate message:

```text
coverage gate failed: 6/8 cells cannot satisfy per_cell=4; ...
ERROR: clean-control coverage incomplete after preregistered 140 episodes
```

Artifact: `runs/ngc_w9_clean_control_state_keys.json`
(`coverage_complete: false`, `n_states: 0`).

## Contrast with frozen clean baseline

| Setting | Success |
|---|---:|
| SmolVLA clean LIBERO baseline (seed 0, nas10, 2000 ep) | **70.0%** |
| W9 clean-control collect (this run, 138 ep) | **7.2%** |

The gap is large enough that the failure mode is **not** “need a few more
episodes under the same recipe.” Preregistered kill rule applies: stop and
audit task/environment mapping before any further sampling.

## Decision (frozen)

1. **Do not** lower `per_cell`, drop Object/Goal/Long, or train on
   Spatial-only / failure-only labels.
2. **Do not** start ridge / MLP / RL selector until a valid clean32 (or
   revised protocol with documented scope change) exists.
3. Next work is **diagnosis**, then a new collection record:
   - Confirm adapter truly selects original LIBERO clean tasks (catalog ids
     used here span `*_000001`–`*_000010` within Plus suites).
   - Reconcile why clean success ≪ 70% baseline (env wrapper, horizon,
     observation stack, checkpoint path, seed / init).
   - Only then recollect a success-retained control pool under a corrected
     config; cite this record as superseded collection attempt.

## Artifacts

- `pool/ngc_w9_clean_controls/manifest.json`
- `runs/ngc_w9_clean_collect_{20260730,20260731,20260732}.json`
- `runs/ngc_w9_clean_control_state_keys.json`
- `runs/ngc_w9_direct_smol_failure24/summary.json`
- `runs/ngc_w9_clean_selector_pipeline.log`

## Follow-up

Gate **W9-B remains open**. Selector Gate **W9-C must not start**. Paper-facing
claim stays: failure-frontier recoverability is policy-relative; clean-regret
selector evidence is still missing.

# W9C: clean task-identity fix

**Date:** 2026-07-31  
**Status:** code + tests + probe + W9C collect/readiness complete; ridge method branch killed (see `2026-07-31_w9c_selector_gate_result.md`)

## Root cause (W9A/W9B)

W9B fixed init diversity (140 unique suite×task×init) but SR stayed ~8.6% with Object/Goal at 0. The remaining bug was **task identity**:

- Adapter assumed “Plus keeps the ten original LIBERO tasks first.”
- Plus suites have 2400+ tasks; indices **0–9 are `_table_*` / layout variants** of one language task, not the official clean-10 set.
- Official clean names exist as unsuffixed BDDL/init files, but are **not** registered as Plus `suite.tasks[0:10]`.
- W9B metadata said `clean/none/L0` correctly while rolling out the wrong environments.
- Pool marked: `pool/ngc_w9b_clean_controls/VALIDITY.json` → `diagnostic_wrong_task_identity` (retain, do not train).

## Fix (W9C)

| Piece | Location |
|---|---|
| Frozen clean-10 names | `configs/clean_libero_task_names.json` |
| Clean path + 10-task suite | `rase/backends/libero_clean.py` |
| Adapter clean branch | `rase/collect/lerobot_libero_plus_adapter.py` |
| Factory `libero_flavor=clean\|plus` | `rase/collect/libero_env_factory.py` |
| Schedule / pool isolation | `W9C-clean-control/v1`, `pool/ngc_w9c_clean_controls`, seed `20260731` |
| Schedule SHA | `f4c944975385e2088f2e9a2dd10423231b6b10e819cc53e32f6c5907cbe99fd1` |

Clean controls load **vanilla exact-name** BDDL/init (`n_tasks==10`). Plus remains for failure/challenge only. Metadata adds `clean_task_name`, `bddl_stem`, `libero_flavor`.

## Tests

- `tests/test_clean_task_identity.py` — reject Plus suffixes; assert frozen names; `n_tasks==10`
- Strengthened `tests/test_lerobot_collection_adapter.py`
- `tests/test_w9c_schedule.py`

## Hard stops

- Do not re-run W9/W9B collect unchanged
- Do not train on W9A/W9B pools as control
- Do not lower coverage / Spatial-only selector
- Probe must pass before `scripts/run_w9c_clean_selector_pipeline.sh`

## Probe gate (preregistered) — PASSED

Per suite ≥20 ep; suite SR ≥ 0.5× baseline (Spatial≥34%, Object≥45%, Goal≥39%, Long≥22%); mean ≳35%; Object/Goal must not be ~0.

| Suite | SR | Floor | Notes |
|---|---:|---:|---|
| Spatial | 0.45 | 0.34 | official clean-10 names |
| Object | 0.85 | 0.45 | was 0 under W9B wrong identity |
| Goal | 0.80 | 0.39 | was 0 under W9B wrong identity |
| Long | 0.35 | 0.22 | after language fix (see below) |
| Mean | 0.6125 | 0.35 | |

Secondary bug found in first probe: Long language used `name.replace('_',' ')`, producing `LIVING ROOM SCENE2 put both...` instead of official `put both...`. Fixed via `grab_language_from_filename(suite, name+".bddl")`. Long re-probe 7/20.

Artifacts: `runs/ngc_w9c_clean_probe_combined_audit.json`.

```bash
bash scripts/run_w9c_clean_alignment_probe.sh
# only if exit 0:
bash scripts/run_w9c_clean_selector_pipeline.sh
```

## Fallback

If probe fails → diagnosis/benchmark paper from W7/W8; selector = future work. No 140-ep recollect.

## Final selector gate backlink

W9C selector pipeline subsequently exited 0 and produced
`runs/ngc_w9c_selector_gate_summary.{json,md}`. Clean32 coverage and both readiness
audits passed, but the preregistered method comparison did not: learned minus
action-matched random had Δutility=0 on both episode- and task-held-out tests.
The result is **`kill_method_branch`**: do not promote ridge and do not train
MLP/RL. The task-held-out test contains only 8 clean-control states and no
failure-challenge states, so it cannot establish failure-routing generalization.

Canonical result: [W9C selector gate result](2026-07-31_w9c_selector_gate_result.md).
Claim posture: [paper claim freeze](2026-07-31_paper_claim_freeze.md).

# W10 Object/Spatial failure benchmark result

**Date:** 2026-07-31  
**Status:** COMPLETED — pipeline hard-stopped at split `NOT_READY` (by protocol)  
**Protocol:** `W10-object-spatial-benchmark/v1`  
**Runbook:** `docs/runbooks/w10_object_spatial_benchmark.md`  
**Pipeline:** `scripts/run_w10_object_spatial_pipeline.sh`  
**Git HEAD at run:** `454f763` (dirty working tree; local research artifacts)

## Research question

On an independently frozen Object/Spatial failure cohort (camera/robot, L1/L2), does
direct OFT escalation recover states where direct Smol continuation fails? After
pairing with frozen identity-valid W9C Object/Spatial clean controls, is there
enough clean/failure and escalate-oracle support for a held-out routing split?

## Protocol freeze

- New collection only: Object/Spatial failure extension, 80 attempted episodes,
  seed `20260731`, camera/robot 40/40, L1/L2, `successful_snapshot_retention=0.0`.
- Benchmark sample: 16 failure states, 2 per `suite × dim × level` cell, distinct
  episode groups, `min_remaining_steps=100`, sample seed `20260731`.
- Direct arms only: Smol continuation temperature `0.5`; suite-matched OFT
  object/spatial checkpoints; one rollout per state and policy.
- Clean source: frozen W9C clean action rows filtered to Object/Spatial
  (`clean_control`), not a new clean collection.
- Explicitly **no** selector / MLP / RL training.
- Hard stop: no top-up, seed change, or outcome-adaptive resampling after 80
  episodes or after `NOT_READY`.

## Execution summary

| Stage | Result |
|---|---|
| Manual gate manifest | `READY_FOR_COLLECTION` |
| Collect 80 | complete: **80 failure / 0 success**, 1120 snapshots retained |
| Inventory | `coverage_complete=true`, deficit `[]` |
| Frozen keys | n=16, `state_keys_sha256=d7dc19cfd36f655017ec19341f8cb543a928cc042c477d31e4d9043911401ccc` |
| Direct Smol | **0/16** |
| Direct OFT Object | **1/8** |
| Direct OFT Spatial | **0/8** |
| Failure action dataset | 16 rows |
| W9C clean Object/Spatial filter | 16 rows (Object 8 / Spatial 8) |
| Merge | 32 rows; cross-source episode-group audit **PASS** (0 overlaps) |
| Episode split + support gate | **`NOT_READY`** |
| Benchmark analysis JSON | not written (hard-stop after split failure) |

## Collection

- Config: `configs/collect_w10_object_spatial_failures.json`
- Pool: `pool/ngc_w10_object_spatial_failures`
- Summary: `runs/ngc_w10_object_spatial_collect80.json`
- Outcomes: `failure=80`, `success=0`
- Quotas realized: Object 40 / Spatial 40; camera 40 / robot 40

All-failure collection is consistent with prior SmolVLA L1–L2 Object/Spatial
pools and is favorable for a failure cohort. It is **not** evidence that
escalation recovers those failures.

## Frozen failure cohort and direct outcomes

State keys: `runs/ngc_w10_object_spatial_state_keys.json`

| Arm | Result | Artifact |
|---|---|---|
| Direct Smol | 0/16 | `runs/ngc_w10_direct_smol_object_spatial16/summary.json` |
| Direct OFT Object | 1/8 | `runs/ngc_w10_direct_oft_object_object_spatial16/summary.json` |
| Direct OFT Spatial | 0/8 | `runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json` |

### Failure-challenge paired table (n=16)

| Suite | n | both | Smol-only | OFT-only | neither |
|---|---:|---:|---:|---:|---:|
| Object | 8 | 0 | 0 | 1 | 7 |
| Spatial | 8 | 0 | 0 | 0 | 8 |
| **All** | **16** | **0** | **0** | **1** | **15** |

Exact McNemar on the single discordant pair: **p=1.0**.

Sole OFT-only recovery:

- `state_key=sp1_2429cb5c5d6adfebd1399d85b36f8dc5`
- suite=Object, dim=robot, level=2
- episode=`ep-0135277b-00000068`
- task=`libero_object_000578`

### Clean-control paired table (W9C Object/Spatial, n=16)

| Suite | n | both | Smol-only | OFT-only | neither |
|---|---:|---:|---:|---:|---:|
| Object | 8 | 4 | 0 | 0 | 4 |
| Spatial | 8 | 5 | 0 | 0 | 3 |
| **All** | **16** | **9** | **0** | **0** | **7** |

Clean controls show no Smol/OFT asymmetry on this Object/Spatial subset: successes
are joint, and the remainder is both-fail.

## Oracle action support (full-information utility)

Cost-aware oracle over `{continue_smol, escalate_oft, abstain}`:

| Cohort | continue_smol | escalate_oft | abstain |
|---|---:|---:|---:|
| failure_challenge | 0 | 1 | 15 |
| clean_control | 9 | 0 | 7 |
| **All** | **9** | **1** | **22** |

There is essentially no escalation-positive mass on Object/Spatial failures: 15/16
failure states are oracle-abstain.

## Split support gate

Requirements: `configs/ngc_w10_split_requirements.json`  
Artifact: `runs/ngc_w10_object_spatial_episode_splits.json`

Status: **`NOT_READY`**

Reasons:

1. `split test optimal action escalate_oft has 0 states; requires at least 1`
2. `split val optimal action escalate_oft has 0 states; requires at least 1`

Per-split oracle support:

| Split | n | clean / failure | escalate_oft oracle |
|---|---:|---:|---:|
| train | 22 | 12 / 10 | 1 |
| val | 5 | 2 / 3 | 0 |
| test | 5 | 2 / 3 | 0 |

The single escalate-oracle state fell into train. This is a structural support
failure, not a tooling bug. Protocol forbids changing the seed, topping up, or
resampling to force `READY`.

## Conclusions

1. **W10 completed as a diagnostic benchmark, not a method success.**
2. On this preregistered Object/Spatial L1/L2 failure cohort, direct Smol recovery
   is **0/16** and direct OFT recovery is **1/16** (Spatial **0/8**).
3. Policy-relative recoverability is **not supported as a broad Object/Spatial
   phenomenon** under this regime. The earlier W7/W8 recoverable positives remain
   concentrated in Goal/Long evidence, not generalized by W10.
4. Because escalate-oracle support is almost absent, a cost-aware routing split
   cannot be well-posed here; `NOT_READY` is the correct scientific outcome.
5. W9C ridge `kill_method_branch` is **not** reopened. W10 adds suite-coverage
   diagnosis, not permission to train MLP/RL.
6. Clean Object/Spatial controls remain useful for clean regret / both-fail
   accounting, but they do not create failure-routing labels.

## Allowed / forbidden claims after W10

Allowed:

- Report Object/Spatial failure direct-matrix as mostly both-fail.
- State that escalate-positive support was insufficient for held-out routing.
- Treat W10 as negative suite-coverage evidence relative to Goal/Long.

Forbidden:

- “OFT escalates Object/Spatial failures” as a general claim from 1/16.
- Changing seed / collecting top-ups / relaxing requirements to make split READY.
- Training a selector on W10 or claiming learned routing.
- Conflating collection all-failure (80/80) with unconditional task impossibility
  outside this perturbation/policy/horizon setting.

## Next decisions

Do **not** immediately open a new method branch. Decide between:

1. **Claim contraction:** keep positive recoverability claims centered on
   Goal/Long; present Object/Spatial W10 as a regime where both policies fail.
2. **New preregistered coverage extension:** different t0 window, perturbation
   family, or policy pair — only if seeking Object/Spatial positives under a
   fresh freeze. Not a rerun of seed `20260731`.

Optional CPU follow-up that does not change the gate: run
`scripts/analyze_selector_benchmark.py` on the frozen 32-row merge for
descriptive tables, explicitly labeled non-confirmatory because split is
`NOT_READY`.

## Artifact index

- `runs/ngc_w10_manual_gate_manifest.json`
- `runs/ngc_w10_collection_request_schedule.json`
- `runs/ngc_w10_object_spatial_collect80.json`
- `runs/ngc_w10_object_spatial_inventory.json`
- `runs/ngc_w10_object_spatial_state_keys.json`
- `runs/ngc_w10_direct_smol_object_spatial16/summary.json`
- `runs/ngc_w10_direct_oft_object_object_spatial16/summary.json`
- `runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json`
- `runs/ngc_w10_object_spatial_failure_action_dataset.jsonl`
- `runs/ngc_w10_w9c_object_spatial_clean_action_dataset.jsonl`
- `runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl`
- `runs/ngc_w10_object_spatial_heldout_merge_manifest.json`
- `runs/ngc_w10_cross_source_identity_audit.json`
- `runs/ngc_w10_object_spatial_episode_splits.json`
- `runs/ngc_w10_pipeline_logs/pipeline_20260731_190512.log`

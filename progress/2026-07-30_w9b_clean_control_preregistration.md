# W9B clean-control preregistration

## Status

**Code/test/live-smoke complete. Formal collection not started.**

This record freezes the proposed W9B implementation. It does not authorize
running `scripts/run_w9b_clean_selector_pipeline.sh`.

## Root cause

W9A created a fresh LeRobot `LiberoEnv` for every episode with
`episode_index=0`. LeRobot initializes `init_state_id = episode_index`; the
first reset therefore repeatedly selected official `init_states[0]`. Policy
seed changes did not select a different initial state. The W9A pool remains
preserved and marked diagnostic/invalid-for-control.

## Frozen W9B paths

- pool: `pool/ngc_w9b_clean_controls`
- runs: `runs/ngc_w9b_*`
- collection config: `configs/collect_w9b_clean_controls.json`
- selection config: `configs/ngc_w9b_clean_controls.yaml`
- schedule: `configs/w9b_clean_control_schedule.json`
- entry point: `scripts/run_w9b_clean_selector_pipeline.sh`
- lock: `runs/ngc_w9b_clean_selector_pipeline.lock`

No W9A output path, run name, or lock is reused.

## Schedule algorithm

Schema/protocol:

- `rase-w9b-clean-control-schedule/v1`
- `W9B-clean-control/v1`

The complete 140-row schedule is generated once from seed `20260730`.
Each batch is independently suite-balanced:

- batch 1: 60 (15 per suite)
- batch 2: 40 (10 per suite)
- batch 3: 40 (10 per suite)

Mappings use separate SHA-256 salts:

- suite order: `suite-order/v1`
- task: `task-id/v1`
- init permutation: `init-order/v1`
- policy seed: `policy-seed/v1`

For each `(suite, task_id)`, init IDs are consumed from a salted permutation of
`0..49`. A second permutation cycle is permitted only after all 50 IDs are
consumed. Allocation state spans all three batches and never resets at a batch
boundary. Task selection, init ordering, and policy seed are therefore
deterministic but not modulo-coupled.

Every row records global index, batch/local request index, suite, clean task ID,
init ID, policy seed, clean/none/L0 semantics, and stable episode ID.

Schedule SHA-256:

`71e61d3cd4d36469652735293b7c8e23b93fb22aa450487c58d21e085e8e1943`

The schedule uses canonical byte-stable JSON and a `.sha256` sidecar. The
pipeline regenerates the expected schedule in memory and verifies both file
SHA and semantic equality before tests or simulator work. Missing rows,
duplicate episode IDs, invalid batch sizes, non-contiguous indices, or
out-of-range init IDs fail closed.

## Metadata and provenance

`StateMetadata.init_state_id: int | None = None` is backward compatible. It is
not part of state-key identity, so old state keys remain unchanged. New W9B
bundles write the value to `meta.json` and manifest entries.

Collection summaries include the scheduled episode rows plus:

- protocol version
- schedule path/SHA-256
- selected batch ID
- repository `git rev-parse HEAD`

At runtime, the adapter verifies scheduled suite/task/init against the actual
episode result. The factory and adapter pass explicit init IDs to LeRobot as
`episode_index`; out-of-range IDs fail before reset.

## Retention and collection cap

W9B fixes `successful_snapshot_retention = 1.0`: every eligible successful
snapshot is retained. Generic validation accepts `[0,1]`, preserving legacy
W9A `0.20` semantics.

The collection cap remains exactly `60 + 40 + 40 = 140`. Coverage failure at
140 stops the protocol. Clean selection remains 32 states:

- 4 suites
- 4 early + 4 mid per suite
- success only
- distinct episode groups

No adaptive top-up or threshold reduction is allowed.

## Tests

Environment: `smolvla`.

- focused: **69 passed**
- full: **181 passed, 5 skipped**
- `bash -n`: pass
- `py_compile`: pass
- frozen schedule regeneration/check: pass
- IDE lints: no new diagnostics

Coverage includes deterministic/seed-sensitive schedule generation, independent
salt mappings, init uniqueness through 50, cross-batch continuation,
adapter→episode-index plumbing, bounds/missing-init rejection, metadata
backward compatibility, retention `1.0` and legacy `0.20`, path isolation,
SHA mismatch rejection, and byte-stable resume.

## Live smoke

Artifact: `runs/ngc_w9b_clean_control_smoke.json`

Task: `libero_spatial_000005`.

Init 0/1/2 pairwise step-0 differences:

| Pair | L2 | max abs |
|---|---:|---:|
| 0–1 | 0.0581857979 | 0.0258676171 |
| 0–2 | 0.0980279662 | 0.0489377698 |
| 1–2 | 0.1021082088 | 0.0389167015 |

These are substantive differences, not `1e-14` numerical noise.

- Same init/seed repeated: sim-state, observations, fingerprint all byte/hash
  identical.
- Policy seed 101→202 with init fixed: step-0 sim-state unchanged.
- Adapter construction path vs official LeRobot eval path: sim-state,
  observations, and fingerprint all identical.
- Forced terminal: `terminated=true`, `truncated=false`, `final_info` present,
  and `success_from_info=true`.
- Resume smoke: first pass wrote 60 synthetic smoke entries; second pass wrote
  0, skipped the same 60 episode IDs, and left manifest bytes unchanged.

Smoke-only storage:

- `runs/ngc_w9b_clean_control_smoke.json`
- `runs/ngc_w9b_resume_smoke_pool/`

No state was written to the formal W9B pool.

## Diff and repository version

Pre-documentation W9B diff: 20 files, **1623 insertions / 37 deletions**.

Base commit before W9B: `ea7ad403c002302234cf7aa81476bb869e86b586`.
The commit containing this file is reported after commit; embedding its own
SHA in the committed bytes is cryptographically self-referential.

## Frozen decision

Formal W9B collection: **not started**.

After this implementation commit, stop and await explicit collection approval.
Do not start clean collection, selector/ridge, DQN, MLP, or RL.

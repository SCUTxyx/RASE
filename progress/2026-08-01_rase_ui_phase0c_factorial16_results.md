# RASE Unified Intervention Phase 0C factorial calibration result

Date: 2026-08-01 15:35 CST  
Status: **COMPLETED — opportunity gate NOT_READY; selector/world-model training remains closed**

## Research question

Can a suite-balanced calibration over clean and weak perturbation cells break the
fixed-OFT success ceiling and expose task-supported complementarity among strict
CONTINUE, REPLAN, and SWITCH_OFT?

## Frozen design

- Config: `configs/collect_rase_ui_phase0c_factorial16.json`
- Four suites: Spatial, Object, Goal, Long.
- Per suite: two clean L0 episodes, one camera L1 episode, one robot L1 episode.
- Sixteen unique tasks and source episodes; four decision states per episode at
  steps 0/2/4/6; 64 same-state trials per operator.
- One continuation seed per arm; strict CONTINUE requires decision-context v2
  active-action suffix provenance.
- Source outcomes: 2 success / 14 failure episodes.
- Base repository commit: `454f76384e5195a750584dd9753c29b0701bb6af`;
  the working tree also contains the tracked and untracked research changes
  documented below.

The factorial pool audit passed exactly: 16/16 expected episodes, 16 unique
tasks, no duplicates, 64/64 strict-CONTINUE states, and 16 states at every
frozen step.

## Main result

| Arm | Success | Rate |
|---|---:|---:|
| strict CONTINUE | 19/64 | 0.296875 |
| REPLAN Smol | 14/64 | 0.218750 |
| SWITCH_OFT | 38/64 | 0.593750 |
| same-state success oracle | 39/64 | 0.609375 |

The best fixed arm is SWITCH_OFT. The raw success oracle gain is only 1/64 =
0.015625. Episode-cluster bootstrap (16 source episodes, 10,000 replicates)
gives a 95% percentile interval `[0.0, 0.046875]`, with 0.644 of replicates
strictly positive. The preregistered minimum gap was 0.05.

Success vectors in registry order `(CONTINUE, REPLAN, OFT)` are:

- `000=25`: no arm succeeds;
- `001=20`: OFT alone succeeds;
- `101=5`: CONTINUE and OFT succeed;
- `110=1`: CONTINUE and REPLAN succeed while OFT fails;
- `111=13`: all arms succeed.

There is exactly one raw-success counterexample to OFT coverage, but it is a
CONTINUE/REPLAN tie. Only OFT has raw unique wins: 20 states across 12 tasks.
REPLAN is sample-wise weakly dominated by CONTINUE (CONTINUE-only 5,
REPLAN-only 0, ties 59; exact McNemar `p=0.0625`). OFT versus CONTINUE has
OFT-only 20 and CONTINUE-only 1 (`p=2.09808349609375e-05`).

The success-only opportunity audit is `not_ready` for two frozen reasons:

1. oracle gap 0.015625 < 0.05;
2. task-supported unique-winning operators 1 < 3.

## Where the single gap occurs

- Clean L0: C/R/O = 19/14/25 of 32, oracle 26/32, gap 1/32.
- Camera L1: C/R/O = 0/0/8 of 16, oracle gap 0.
- Robot L1: C/R/O = 0/0/5 of 16, oracle gap 0.
- Only Spatial has a suite gap: 1/16.
- Only snapshot step 2 has a time-stratum gap: 1/16.

The current weak perturbations do not form a recoverable Smol boundary: every
L1 CONTINUE and REPLAN rollout failed. The positive signal is concentrated in
one clean Spatial step-2 state and is not yet stable enough to scale directly.

## Cost and timing interpretation

Whole-rollout mean elapsed seconds were CONTINUE 11.29, REPLAN 11.87, and OFT
6.57, but these values are confounded by success-dependent early termination
and must not be interpreted as model inference cost.

New OFT instrumentation measured action-prediction RPC transfer plus server
inference separately: 1,651 calls, 213.631 seconds total, 129.395 ms/call, and
64/64 state coverage. Environment stepping is excluded.

The post-hoc cost sweep now separates recovery-supported winners from cheaper
choices on all-failure states. At an exploratory OFT penalty of 0.10, the raw
utility oracle gap is 0.084375, but success-supported unique winners are only
CONTINUE 19, REPLAN 0, OFT 20; another 25 CONTINUE winners are merely the
lowest-cost choice on `000` states. Therefore the cost sweep does not rescue a
three-family complementarity claim.

## Code completed in this phase

- Deterministic weighted factorial sampling and exact suite-by-cell balance.
- Pool audit for design counts, unique tasks, strict continuation, source
  outcomes, and snapshot steps.
- Suite/dimension/level/step stratified matrix analysis with episode-cluster
  bootstrap.
- OFT RPC inference timing, plus Smol action-selection timing for future runs.
- Generic matrix metadata; removed stale hard-coded “8-state/2-task” text.
- Cost-sweep separation of success-supported and failure-only cost winners.
- Resumable end-to-end Phase 0C runner with explicit opportunity gate.

## Decision and next work

Do not train a selector or world model, and do not run a 96-state replication of
the unchanged three arms. First:

1. Freeze this calibration as diagnostic; do not change its gate post hoc.
2. Add an executable, public-information-only third intervention whose semantics
   are not equivalent to discarding the Smol action queue. Preferred candidates
   are LOCAL_CORRECT or REWIND after simulator feasibility/parity tests; retire
   current REPLAN from the core set if an independent screen again finds zero
   REPLAN-only success.
3. Use the new Smol and OFT instrumentation to calibrate deployment cost on a
   separate timing cohort. Normalize per successful episode or per action
   decision, and pre-register the scalarization before outcome confirmation.
4. Run a small operator-semantics screen concentrated on clean/near-boundary
   Spatial and adjacent tasks, with task/episode IDs disjoint from this run.
5. Only after at least two tasks support each retained operator and the
   success-supported utility oracle gap is at least 0.05 should a balanced
   96-state held-out confirmation and selector/world-model split be frozen.

## Artifacts

- Pool: `runs/rase_ui_phase0c_factorial16_pool/`
- Pool audit: `runs/rase_ui_phase0c_factorial16_pool_audit.json`
- Smol run: `runs/rase_ui_phase0c_factorial16_smol_factorial_v1/`
- OFT runs: `runs/rase_ui_phase0c_factorial16_oft_{spatial,object,goal,10}_factorial_v1/`
- Matrix: `runs/rase_ui_phase0c_factorial16_matrix_factorial_v1/`
- Log: `runs/rase_ui_phase0c_factorial16_factorial_v1.log`

Key SHA-256 values:

- config: `a5e5ffbd2bf0171765c994b0e90a24c35f1bc7914c0c55b42a4d9397ca9ea4ab`
- state keys: `f0c7f002ee7530d92d806b28effba85c8c74608355c64b050c1e10e0303cc651`
- pool audit: `e0c65da69f2c5c4ccb6c979a2bb101041a773c4586a0c1f84195ff880b15e2a5`
- matrix summary: `aaeca9aeed068c7befb45b4830e38f6e9b412f2292c194046b0b756aa75f4fdf`
- matrix analysis: `40dc96f3f3fd3acf338fe0e18da43c499be4b11fe3d4ee0ae1fe3e34de5a6f0d`
- opportunity audit: `d0e9ba3bbffc4c6fd0918b79f3f9bb689e33366de6553373d1b755870305dcb4`

## Reproduction command

The completed run was launched in tmux 0 from the repository root:

```bash
FRESH_RUN=1 TAG=factorial_v1 ./scripts/run_rase_ui_phase0c_factorial16.sh
```

The session is now idle and the GPU has no compute process.

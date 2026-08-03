# RASE-PRE PRE-A2 closed-loop recovery duration

## Status and question

**Status: duration-structure signal; replication required.** PRE-A0 showed that
same-policy resampling does not create behavioral diversity, while PRE-A1 showed
that an OFT prefix of at most eight environment steps cannot repair Smol. PRE-A2
asks how long a stronger, closed-loop policy must remain in control before
handing back to the base policy.

This is a 12-state development mechanism audit, not confirmatory evidence and not
a top-conference claim by itself.

## Frozen protocol

- Cohort: frozen PRE-A0 12-state set, three cells per LIBERO suite (clean,
  camera L1, robot L1), one episode per state.
- Base/handback policy: the frozen SmolVLA checkpoint/profile and seed from
  PRE-A0/PRE-A1. Recovery policy: the frozen OFT RPC policy from Phase 1A.
- Recovery durations: `h={0,8,16,32,64}` environment actions. Each OFT action
  sequence was captured under persistent closed-loop OFT from the exact snapshot,
  then its deterministic prefix was replayed before same-seed Smol continuation.
- Persistent reference: existing Phase 1A OFT-only outcome at the same state.
- Config: `configs/pre_a0_strict_resample12.yaml` and
  `runs/rase_pre_a0_strict_resample12_keys_v1.json`.
- Scale: 12 captured trajectories plus 60 paired handback rollouts.

Replay preserves a deterministic counterfactual, but does not query OFT live
during prefix replay. Confirmation must remove this limitation or show parity.

## Reproducibility identity

- Code base before PRE-A2: `9b0504d768a75ad2e86da38b9e7a885028370255`.
- Python 3.12.13 at `/root/autodl-tmp/envs/smolvla/bin/python`.
- `env.lock.md` SHA-256:
  `b3e18d916dee8941105ed26ad10a2523fc2a82c02e8835478023682b6501adbb`.
- Entry point: `scripts/run_recovery_duration12.sh`.
- No trainable state and no world model, critic, selector, replanner, or
  termination model.

## Results

| Recovery duration | Success | Base successes harmed |
|---:|---:|---:|
| 0 | 3/12 | -- |
| 8 | 3/12 | 0 |
| 16 | 3/12 | 0 |
| 32 | 2/12 | 1 |
| 64 | 5/12 | 0 |

- Best fixed finite duration: 64 steps, **5/12**, versus base **3/12**.
- Finite-duration oracle: **5/12**, with **2 rescues across two suites/tasks**.
- Both rescues are camera L1 (Spatial and Goal), and both require 64 steps.
- Persistent OFT: **11/12**. Six base failures remain direct-OFT-only after every
  finite duration, so finite handback does not explain most of OFT's advantage.
- Goal clean is non-monotonic: success at `h={0,8,16,64}` but failure at `h=32`.
- Per suite at `h=64`: Spatial 2/3, Object 1/3, Goal 2/3, Long 0/3.

The positive result is narrow but meaningful: recovery has temporal structure and
premature handback can cause failure. The negative result is equally important:
a fixed duration can cause harm, and most failures still need persistent OFT.

## Decision and claim boundary

The pilot gate is `duration_structure_signal`: at least two rescues and at least
+8 percentage points over base under the finite-duration oracle. It passes
(2 rescues, +16.67 pp), so task-disjoint replication is justified. The critic/
termination gate remains **replication required** and the world-model gate remains
**closed**.

The viable claim is not “escalate to a stronger policy.” It is: **learn or certify
the minimum sufficient recovery duration and a safe handback condition under
recovery cost and false-handback harm constraints.** Always-OFT, fixed-64, and
persistent OFT remain mandatory baselines.

## Next stage

1. Freeze PRE-A3 before outcomes: 96--120 task/episode-disjoint states, balanced
   by suite and perturbation; reserve a hidden confirmation split.
2. Repeat `h={0,8,16,32,64}`, add `h={96,128}`, and use live closed-loop prefix
   execution. Report paired rescue/harm with episode/task bootstrap intervals.
3. Require replicated cross-task rescues, duration heterogeneity, and low
   false-handback harm before training anything.
4. Only after passing, fit a calibrated competence/termination predictor for
   “base succeeds after handback,” conditioned on state, elapsed recovery, and
   policy identity. Do not train an action replanner or generative world model.
5. Compare success, selective risk, recovery cost, and harm against base,
   always-OFT, fixed durations, oracle duration, and recovery baselines.

## Artifacts

- `runs/rase_pre_a2_oft_recovery_trajectories12_v1/*.npz`
- `runs/rase_pre_a2_recovery_duration12_v1/summary.json`
- `runs/rase_pre_a2_recovery_duration_audit12_v1.json`
- `runs/rase_pre_a2_recovery_duration12_v1.log`

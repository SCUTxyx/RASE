# RASE-UI Phase 0D timing16 calibration result

Date: 2026-08-01 16:45 CST  
Status: **COMPLETED — timing coverage PASS; no-world-model and no-training freeze remains active**

## Question and cohort

Can physically measured action-acquisition latency justify a higher utility cost
for SWITCH_OFT than for Smol CONTINUE/REPLAN?

The frozen cohort contains exactly one step-2 state from every Phase 0C source
episode: 16 states, 16 episodes, 16 tasks, four states per suite, with cells
clean L0 = 8, camera L1 = 4, and robot L1 = 4. The cohort is for resource
calibration only and is not an opportunity-confirmation dataset.

## Coverage

- Smol strict CONTINUE: 16/16 timed trials.
- Smol REPLAN: 16/16 timed trials.
- Existing same-state OFT: 16/16 timed trials.
- All selected keys have strict decision-context v2 continuation support.
- No missing or duplicate state keys.

## Outcome consistency

The step-2 subset reproduces the corresponding Phase 0C slice:

- CONTINUE: 6/16 success;
- REPLAN: 4/16 success;
- OFT: 10/16 success;
- paired Smol: both success 4, CONTINUE-only 2, REPLAN-only 0,
  both failure 10; exact McNemar `p=0.5`.

These outcomes are diagnostic consistency checks, not a new independent success
experiment because the same Phase 0C snapshots and deterministic seeds are used.

## Measured policy/action-acquisition time

| Operator | Mean policy time/trial | Policy time/env step | Calls | Mean/call |
|---|---:|---:|---:|---:|
| strict CONTINUE | 7.319 s | 29.394 ms | 3,904 | 29.996 ms |
| REPLAN | 8.113 s | 30.089 ms | 4,314 | 30.089 ms |
| SWITCH_OFT | 3.513 s | 16.483 ms | 433 | 129.806 ms |

Measurement scopes differ by deployment API:

- Smol measures wall time inside `select_env_action`, including cached action
  queue access and model forward passes, excluding environment stepping.
- OFT measures action-chunk prediction RPC transfer plus server inference,
  excluding environment stepping and client rollout control.

OFT has a slower individual RPC, but each RPC returns an action chunk. After
chunk amortization, its measured policy time per environment step is 0.561x
CONTINUE, and its mean policy time per trial is 0.480x CONTINUE. REPLAN is
1.024x CONTINUE per environment step and 1.108x per trial. Whole-rollout time is
also reported but remains outcome/horizon dependent.

## Scientific conclusion

The current data do **not** support assigning OFT a larger scalar cost on the
basis of deployment latency. In this software/hardware configuration, OFT has
higher per-RPC latency but lower action-chunk-amortized policy time. The previous
post-hoc `switch_oft=0.10` penalty remains an exploratory sensitivity value and
cannot be converted into a physical latency claim.

If OFT is to receive a higher resource cost, it must be supported by separately
measured quantities such as GPU peak memory, energy, model residency, network
bandwidth, or deployment availability. Those quantities must be measured on a
separate calibration cohort and scalarized before held-out outcomes are opened.

This result further strengthens the decision not to train a world model now:
the success benchmark is still nearly solved by always-OFT, and the hypothesized
latency tradeoff does not currently create additional decision headroom.

## Code completed

- `export_decision_context_keys.py`: deterministic `--step`,
  `--one-per-episode`, and exact `--expected-states` selection with suite/cell
  metadata.
- `analyze_intervention_timing.py`: exact Smol/OFT state-key join, coverage
  enforcement, per-trial/per-call/per-env-step metrics, and explicit
  cross-API comparability warnings.
- `run_rase_ui_phase0d_timing16.sh`: preflight, frozen-key export, resumable Smol
  timing, OFT metric join, and safe `--help` handling.
- Unit tests for selection, exact coverage, missing-state rejection, scope
  separation, and action-chunk normalization.

## Deviation record

A read-only preflight probe passed `--help` before the runner implemented help
handling, which started a partial `v1` run outside tmux. It was terminated
precisely, its partial artifacts were retained and excluded, and `--help` was
fixed and tested. The official run is `v2`, executed in tmux 0. It reused the
already deterministically frozen 16-key artifact and produced a fresh Smol run
and analysis.

## Next experiment priority

Do not run the unchanged 96-state matrix and do not train a model. The next
operator-semantics screen should compare:

1. strict CONTINUE;
2. immediate SWITCH_OFT;
3. active-suffix-then-SWITCH_OFT (deferred switch).

The deferred-switch profile is preferred over inventing a privileged local
controller because it has an executable public-state contract and directly
tests whether preserving a valid active suffix avoids the known harm from
immediate queue discard. Existing prefix-ablation code must first be audited to
ensure it replays the frozen decision-context suffix rather than candidate or
privileged actions.

Run 24–32 new, task/episode-disjoint near-boundary states only after that parity
audit. Enter a 96-state confirmation only if retained operators obtain unique
success-supported wins on at least two tasks each and a preregistered oracle gap
of at least 0.05.

## Artifacts and identity

- Keys: `runs/rase_ui_phase0d_timing16_keys.json`
- Official Smol run: `runs/rase_ui_phase0d_timing16_smol_v2/`
- Official analysis: `runs/rase_ui_phase0d_timing16_analysis_v2.json`
- Official log: `runs/rase_ui_phase0d_timing16_v2.log`
- Excluded partial run: `runs/rase_ui_phase0d_timing16_smol_v1/`
- Base commit: `454f76384e5195a750584dd9753c29b0701bb6af`

SHA-256:

- keys: `c262a77f50d8dd2989b537a1a677269b22f4a5e9f4db3dc1c277cf07a507dc71`
- Smol summary: `df8ae5d68d8c264432bf1f6bf555d0a6f3db07c6d0c8743a07fde85c6e34bc70`
- timing analysis: `4b89c0f7a217364bee53dd166bb7144dce5f1ba8c8650f37303ea03db541a86f`

Official command in tmux 0:

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=0 TAG=v2 ./scripts/run_rase_ui_phase0d_timing16.sh
```

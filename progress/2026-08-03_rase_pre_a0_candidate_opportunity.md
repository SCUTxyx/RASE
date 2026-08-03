# RASE-PRE candidate-opportunity progress — 2026-08-03

## Decision

The project is transitioning from a timing-first intervention selector to a
candidate-first escalation system. The immediate question is not whether a
world model can be trained. It is whether same-state alternatives contain
repeatable, task-distributed rescue opportunity beyond a preregistered base
proposal.

World-model and learned-critic training are frozen until the opportunity gates
pass. A world model, if later justified, will be candidate-conditioned evidence
for ranking and escalation rather than a general-purpose video-prediction goal.

## Evidence already established

- W5: changing SmolVLA sampling temperature increased diversity but produced
  0/576 candidate hits across 72 state-runs.
- W6/W7: strict same-policy candidate generation remained 0/256 candidate hits;
  OFT continuation succeeded on 10/64 W6 and 8/24 held-out W7 states.
- W7 attribution: 0/2 apparent prefix rescues were candidate-specific; one was
  continuation-sufficient and one was passive-zero-prefix sufficient.
- Phase 1A reset replacement audit: source-only 10/48, direct OFT 42/48, and
  source-to-OFT 37/48. This proves a strong fallback-policy opportunity, not a
  diverse candidate-family result.

Therefore the old same-policy candidate claim is closed. RASE-PRE remains
feasible only if an explicitly heterogeneous generator portfolio provides
complementary successes.

## PRE-A0 frozen pilot

- Development-only: 12 outcome-independent reset states.
- Coverage: 4 suites × {clean:L0, camera:L1, robot:L1}; one unique task/episode
  per cell.
- Candidate family tested now: K=4 fresh, reset-isolated SmolVLA samples at
  temperature 0.7, with one rollout per candidate.
- Preregistered base: candidate index 0.
- Existing heterogeneous comparison: same-state direct OFT result from Phase 1A.
- Primary outputs: strict oracle@4 headroom, heterogeneous oracle headroom,
  base-failure rescue fraction, rescue task concentration, and strict-only vs
  fallback-only coverage.

This is a minimal opportunity screen, not the full proposal. It does not yet
include the live current suffix, fresh replanning semantics, fixed local
corrections, abstention, or state-changing recovery. OFT fallback compute is not
matched to a 10-step strict prefix.

## Frozen pilot gates

The pilot signal requires all four development conditions:

1. heterogeneous oracle headroom ≥ 8 percentage points;
2. at least 20% of base failures rescued;
3. rescues span at least two tasks;
4. both strict-resample and fallback families have at least one unique success.

Held-out confirmation is deliberately false in PRE-A0. Even if all four pilot
conditions pass, the next status is only `pilot_signal_requires_scaled_heldout`.
Critic and world-model gates remain closed.

## Next-stage decision tree

- If strict-only and fallback-only successes both exist: implement PRE-A1 with
  protocol-matched current suffix, fresh replan, OFT fallback, and two fixed
  local corrections on 100–200 independently frozen snapshots.
- If only OFT succeeds: do not train a candidate critic or world model. First
  add genuinely different generator families; otherwise reformulate the method
  as conservative policy escalation/replacement.
- If strict sampling alone has headroom but no unique coverage: retain it as a
  low-cost diversity source, but the top-level novelty still requires
  heterogeneous generation.
- If neither family rescues failures: stop the PRE route on this benchmark and
  redesign the state distribution or base/fallback policy pair before scaling.

## Reproducible command

```bash
cd /root/autodl-tmp/RASE
tmux new-session -d -s 0 \
  'cd /root/autodl-tmp/RASE && FRESH_RUN=1 ./scripts/run_pre_a0_strict_resample12.sh'
tmux attach -t 0
```

## PRE-A0 completed result

The frozen run completed successfully in `tmux 0` on 2026-08-03. All 12
candidate artifacts had shape `(4, 10, 7)` and all 48 one-shot rollouts
completed. Mean pairwise endpoint L2 was 1.0616 and mean pairwise chunk L2 was
0.5237: the samples were numerically different.

The behavioral result was nevertheless negative for strict resampling:

- preregistered first sample: 3/12 = 25.0%;
- strict oracle@4: 3/12 = 25.0%;
- strict oracle headroom: +0.0 percentage points, task-bootstrap 95% CI [0, 0];
- strict rescues: 0/9 base failures;
- mixed-outcome states: 0/12. Every state was either 4/4 success or 0/4
  success.

All three strict-success states were clean:L0. Strict sampling was 0/4 on
camera:L1 and 0/4 on robot:L1. Therefore numerical action diversity did not
translate into outcome-level candidate opportunity on this pilot.

The already-completed direct OFT fallback produced:

- OFT: 11/12 = 91.7%;
- heterogeneous oracle (strict ∪ OFT): 11/12 = 91.7%;
- heterogeneous headroom over first sample: +66.7 percentage points,
  task-bootstrap 95% CI [+41.7, +91.7];
- 8/9 base failures rescued across 8 unique tasks;
- portfolio strict-only/fallback-only/both/neither = 0/8/3/1;
- strict-vs-fallback exact McNemar p = 0.0078125.

The headline is not “candidate generation works.” The entire heterogeneous
gain is attributable to the stronger fallback policy. The frozen status is
`not_ready`: the headroom, rescue-rate, and task-spread conditions pass, but
the required two generator families with unique successes fails; held-out
confirmation is also intentionally pending. World-model and critic-training
gates remain closed.

Artifacts:

- `runs/rase_pre_a0_strict_resample12_keys_v1.json`
- `runs/rase_pre_a0_strict_resample12_candidates_v1/summary.json`
- `runs/rase_pre_a0_strict_resample12_screen_v1/summary.json`
- `runs/rase_pre_a0_candidate_opportunity_v1.json`
- `runs/rase_pre_a0_candidate_opportunity_v1.md`
- `runs/rase_pre_a0_strict_resample12_v1.log`

## Revised next stage: PRE-A1 generator assay

Do not scale the current same-profile K=4 recipe and do not train a ranker yet.
The next experiment must first create genuinely different candidate semantics:

1. freeze 96–120 mid-episode snapshots without using downstream candidate
   outcomes, balanced across suites, perturbation cells, and source progress;
2. preserve the actual queued source suffix as the base candidate instead of
   using a reset first sample as a proxy;
3. compare one reset-isolated fresh Smol replan, a chunk-level OFT fallback,
   and two preregistered bounded local corrections around the base chunk;
4. execute every candidate for the same prefix length and under the same
   continuation/horizon, with family IDs and generation seeds in provenance;
5. report base vs oracle, per-family unique successes, task concentration,
   compute/latency, and held-out direction before fitting any critic.

OFT-only replacement is a strong engineering baseline and must remain in every
table. It is not by itself sufficient evidence for the RASE-PRE candidate
selection claim. If PRE-A1 still has zero non-OFT unique successes, the honest
project direction is conservative policy escalation/replacement rather than a
candidate-ranking method.

Only after PRE-A1 shows complementary candidate-family opportunity should
PRE-B fit a candidate-conditioned critic. A world model is considered later
only for residual cases where the non-world-model critic has measurable
ranking or safety failures; it is not a prerequisite and is not scheduled now.

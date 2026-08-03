# W8 direct OFT escalation on W7 held-out failure states

## Status

**Complete.** On 2026-07-29, W7 held-out paired evaluation and the W8 direct
OFT-from-snapshot arm both completed. The frozen evaluation unit is one state
from each of 24 distinct episode groups. No temperature, candidate count,
horizon, or state key was changed after outcomes were observed.

## Frozen protocol

- W7 config: `configs/ngc_w7_heldout24_screen.yaml`
- Frozen state keys: `runs/ngc_w7_heldout24_state_keys.json`
- Cohort: 24 SmolVLA failure-conditioned states, 24 episode groups
- Suite allocation: Spatial 4, Object 2, Goal 8, Long 10
- W7 diagnostic arm: any-of-8 Smol prefixes followed by OFT
- W8 deployable arm: direct OFT from the same snapshot, one rollout per state
- Smol checkpoint/candidate protocol: frozen W6/W7 checkpoint, `K=8`, `T=0.7`
- W8 runner: `scripts/run_w8_direct_escalation.sh`

Exact environment, checkpoint, pool, and candidate hashes remain in the
per-run manifests beside each summary. They must be copied into the paper
artifact table; they are not reconstructed from terminal output here.

## Results

### W7 paired matrix

- SmolVLA portfolio: **0/24 states (0%)**, Wilson 95% CI `[0, 0.1380]`
- Prefix+OFT portfolio: **8/24 states (33.3%)**, Wilson 95% CI
  `[0.1797, 0.5329]`
- State pairs: both hit 0, Smol only 0, OFT only 8, both miss 16
- Exact two-sided state-level McNemar: **`p=0.0078125`**

This is a significant policy-relative recoverability gap on the frozen
failure-conditioned cohort. It does not estimate unconditional LIBERO success.

### W8 direct escalation

- Direct OFT overall: **9/24 (37.5%)**, Wilson 95% CI `[0.2116, 0.5729]`
- Long: **5/10 (50%)**, Wilson `[0.2366, 0.7634]`
- Goal: **4/8 (50%)**, Wilson `[0.2152, 0.7848]`
- Object: **0/2**, Wilson upper bound `0.6576`
- Spatial: **0/4**, Wilson upper bound `0.4899`

Because every W7 Smol candidate failed, direct OFT versus Smol has nine
OFT-only discordant state pairs and exact two-sided McNemar
**`p=0.00390625`**, conditional on exact state-key alignment in the completed
summaries. The code now recomputes this from artifacts rather than trusting
terminal marginals.

The frozen state overlap is now complete: both-success **7**, prefix-only **1**
(Spatial), direct-only **2** (Goal 1, Long 1), and both-fail **14**. Exact
McNemar for prefix portfolio versus direct OFT is **`p=1.0`** and the marginal
direct-minus-prefix difference is +4.17pp. Direct OFT recovers 7/8 (87.5%) of
oracle-prefix hits using one deployable action rather than eight counterfactual
branches. The supported claim is simplicity/deployability and high overlap,
not a significantly better direct-OFT success probability.

## Scientific update

The strongest supported mechanism is now direct policy escalation:

1. SmolVLA candidate continuation remains zero on a held-out episode-disjoint
   cohort.
2. A stronger continuation policy recovers a statistically non-zero paired
   subset.
3. Direct OFT overlaps 7/8 prefix-portfolio successes and adds two direct-only
   successes, so candidate reranking is not needed to explain the primary gain.
4. Recovery is concentrated in Goal/Long on this small cohort; Object/Spatial
   zeros are underpowered and must not be presented as proof of impossibility.

The paper method is therefore frozen as a tiny three-action cost-sensitive
router: `CONTINUE_SMOL`, `ESCALATE_OFT`, `ABSTAIN`. Candidate portfolios remain
diagnostic/oracle baselines, not selector labels.

## Selector readiness

Artifact `runs/ngc_w8_direct_escalation_failure.jsonl` is a deployable direct-arm
dataset, but all 24 rows come from SmolVLA-failure episodes. It can support
escalate-versus-abstain diagnosis but cannot identify a safe `CONTINUE_SMOL`
region. Training a three-action selector on it would collapse toward escalation
and yield unmeasured clean regret.

`scripts/run_w8_posthoc_analysis.sh` now:

1. computes the exact W7/W8 state overlap;
2. freezes episode-grouped splits;
3. inventories success-control support across every server state pool; and
4. runs the selector readiness gate, which must return `NOT_READY` on the
   failure-only file.

This expected rejection is a scientific result, not a failed experiment.

## Artifacts

- `runs/ngc_w7_heldout24_policy_matrix.json`
- `runs/ngc_w8_direct_oft_{spatial,object,goal,10}_heldout24/summary.json`
- `runs/ngc_w8_direct_escalation_failure.jsonl`
- `runs/ngc_w8_direct_escalation_pairing.{json,md}`
- `runs/ngc_w8_failure_selector_splits.json`
- `runs/ngc_w8_failure_selector_audit/readiness_audit.json`
- `runs/ngc_w8_selector_pool_support.json`

## Next gate (W9)

1. **Done:** freeze state overlap (7 both, 1 prefix-only, 2 direct-only, 14 neither).
2. **Done:** pool audit found zero reusable clean-success episode groups.
3. Collect a dedicated success-retained control pool. Do not use
   multiple snapshots from one successful episode as independent evidence.
4. Roll out true empty-prefix direct Smol and direct OFT on controls and W7
   challenge states; candidate-0 is no longer the continuation label.
5. Combine challenge and controls, create both episode-heldout and
   task-heldout splits, and train ridge only after readiness passes.
6. Compare learned routing with always-Smol, always-OFT, matched-random trigger,
   abstain, and oracle. Do not add MLP/RL unless ridge beats matched random on
   held-out tasks.

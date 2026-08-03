# RASE escalation dataset schema v1

`rase-escalation-dataset/v1` has one row per decision state. It is separate
from the candidate-centric recovery export: the statistical and deployment
unit is a state, and every arm records the outcome of one executable decision.

## Frozen first-stage actions

1. `continue_smol`: invoke SmolVLA directly from the restored snapshot with an
   empty action prefix. Candidate-0 is legacy diagnostic data, not this action.
2. `escalate_oft`: invoke OFT directly from the restored snapshot, without a
   Smol candidate prefix or a time-matched zero prefix.
3. `abstain`: terminate under an explicitly declared abstention cost.

Candidate indices 0..K-1 are diagnostic arms and oracle/reranker baselines, not
K selector classes. An `any_of_K` portfolio hit is not a deployable action
outcome. Matrix-derived rows must set `proxy=true`; the default trainer rejects
every proxy outcome.

## Required fields

Each row contains `state_key`, `task_id`, `episode_id`, suite/perturbation
metadata, `episode_outcome`, `cohort`, a `features` mapping containing only
signals available at deployment, and an `arms` mapping. Every arm stores:

- `success`: binary task outcome;
- `cost`: preregistered scalar cost on the same scale as success reward;
- `observed`: whether the arm was actually evaluated;
- `proxy`: whether it is only a portfolio/surrogate result;
- `outcome_semantics`: exact policy/prefix protocol.

State-fork outcomes, future termination information, `episode_outcome`, oracle
labels, perturbation dimension/subtype, and ground-truth perturbation level are
targets/audit metadata and must never enter `features`. W9 uses current RGB
summary statistics, proprioception, and `t0` as its deployment baseline.

Frozen initial utility costs are `continue_smol=0.02`, `escalate_oft=0.10`, and
`abstain=0.0` on a unit-success-reward scale. This makes abstention preferable
to paying for a continuation known to fail. Paper results require cost
sensitivity; these constants are not physical latency claims.

## Training gate

The readiness audit blocks training if any of the following holds:

- state or `(task_id, episode_id)` leakage across splits;
- proxy or missing arm outcomes in train;
- no success/failure support for either continuation policy;
- no clean-success control or no failure-challenge states;
- fewer than two optimal actions in train;
- missing or non-finite deployable features.
- any forbidden ground-truth annotation inside `features`.

The primary split is episode-disjoint; a second frozen task-disjoint split is
required for the paper. The first baseline is multi-output ridge utility
regression and reports parameter count, success, cost, strong-policy usage,
abstention, clean regret, always-act baselines, matched-random escalation, and
the full-information oracle upper bound.

# R5-B24 paired probability result and R6 direction

Date: 2026-08-09  
Status: **R5 safe-handback method NO-GO; R6 opportunity qualification only**

## Executive decision

The frozen B24 collection completed all 24 states, 24 true train tasks and four
suites. The data protocol is valid after correcting one audit-only terminal-boundary
off-by-one rule. The pre-registered probability opportunity gate is **NOT READY**.
Consequently:

- do not train the B24 Beta-binomial handback model;
- do not run the five-seed B24 OOF sweep;
- do not unlock a second VLA, world-model features, validation or test;
- move safe handback out of the main method for this policy pair;
- retain real-time source-risk prediction plus corrective OFT takeover as a gated
  replacement hypothesis, not as an established result.

The current scientific interpretation is not that risk-aware multi-VLA control is
universally impossible. It is that **SmolVLA -> persistent OFT does not provide enough
reproducible safe-handback support to train or evaluate the proposed stopping model**.

## Frozen B24 protocol and artifacts

- Cohort: 24 outcome-enriched train-development states, one per true task, six per suite.
- Boundaries: `h={0,16,32,64,96,128}` when reachable.
- Continuations: `K=5` per exact saved boundary, with common random-number seeds across
  boundaries within each state.
- Result: 122 reachable boundaries and 610 continuation trials.
- Persistent replay parity: 24/24.
- Repeat completeness: 100%; duplicate repeat seeds: 0.
- Paired-seed protocol: READY.
- Final protocol gate: **READY**.

Server artifacts:

- `runs/pre_c0_r5/boundary_probability_b24_v1/boundary_transitions.jsonl`
- `runs/pre_c0_r5/boundary_probability_b24_v1/report.json`
- `runs/pre_c0_r5/boundary_probability_b24_v1/probabilistic_summary.json`
- `runs/pre_c0_r5/boundary_probability_b24_v1/decision.json`

Key SHA-256:

- dataset: `01c748d21cc424182983135bdbdecc254905e19f2b7b5b048fdfda62f5535e13`
- collection report: `0438d624624ae53780c6f4b0500a15cbcea410120b6d83f8c447fe4eacf770b3`
- final summary: `18de510df5851414cd36929a2fc6a6642905b458bb429f43e38494d8bf5f9e81`
- decision: `b61df6ddf2f26f084538ea67724546599023409711fffeff7fe14355e9602291`

Environment: repository commit `8d8212643893935368152efce4bc571bb83b59db`,
Python 3.12.13, NVIDIA RTX 5090, `env.lock.md` SHA-256
`b3e18d916dee8941105ed26ad10a2523fc2a82c02e8835478023682b6501adbb`.

## Label and opportunity results

| Metric | B24 result |
|---|---:|
| Persistent success | 21/24 = 87.5% |
| Mean empirical handback success probability | 33.44% |
| Mean Bernoulli entropy | 0.342 bits |
| Non-degenerate boundaries | 46/122 = 37.70% |
| All-success / all-failure boundaries | 15 / 61 |
| Historical binary vs new all-K label agreement | 74/122 = 60.66% |
| Live finite-safe states | 4/24 |
| Recovery-created all-K-safe states | **1/24** |
| Populated positive stopping bins with >=3 states | **0** |
| Conservative cost-oracle savings | 32.63% (3800 -> 2560 steps) |
| Best fixed empirical boundary | h128, 46.67% |
| Empirical probability oracle | 53.33% |
| Oracle minus best fixed | +6.67pp |

The 4 finite-safe states do not constitute four recovery examples: three were already
all-K safe at h0. Only `sp1_6e0e470b638555140e3b63428fb9560e` changed from unsafe at
h0 to all-K success at a later boundary. The high cost-oracle saving is therefore
concentrated in too few states to support a learned stopping controller.

The opportunity gate failed because:

- 4 live finite-safe states < 20;
- 0 populated positive stopping bins < 2.

Savings alone passed, but a cost saving concentrated in one genuine recovery state is
not a learnable or task-general opportunity.

## Entropy and non-monotonicity

- 11/24 states (45.83%) had at least one sampled downward transition.
- 13 downward transitions had an empirical drop of at least 0.4.
- 2 drops were separated by the one-sided Wilson intervals.
- Long had the highest mean entropy (0.646 bits) and 26/36 non-degenerate boundaries.
- Spatial had the lowest mean success probability (11.67%).

These results reject a monotone “more OFT always makes handback safer” assumption.
They do not establish a population rate because B24 is outcome-enriched development
data.

With one-sided 95% Wilson bounds and zero observed failures, at least 11, 25 and 52
repeats are required to certify lower bounds of 0.8, 0.9 and 0.95 respectively. Thus
K=5 is an opportunity screen, not a state-level 95%-safety certificate.

## Audit-only code correction

The initial summary marked one missing h128 boundary as a coverage failure even though
the persistent episode terminated successfully exactly at step 128. There is no
post-action state at which a handback decision can be made. The audit rule was corrected
from `boundary <= executed_steps` to `boundary < executed_steps`; collection data,
manifest, seeds and outcomes were unchanged. A regression test now covers equality at
the terminal step. The complete related R4/R5 risk, leakage, teacher-dependency and
safe-handback regression set passed 16/16 without warnings.

Training now also has a fail-closed `--require-opportunity-ready` interlock. It rejects a
closed opportunity gate, absent summary, mismatched dataset path or changed dataset hash
before fitting any fold.

## R6 source-risk opportunity audit

The QC71 deterministic matrix was used to evaluate an undeployable privileged policy:
use Student exactly on states where Student succeeds, otherwise use persistent OFT.

| Metric | Privileged R6 upper bound |
|---|---:|
| Student success | 18/71 |
| Persistent OFT success | 65/71 |
| Privileged trigger success | 65/71 |
| Persistent teacher steps | 11049 |
| Privileged trigger teacher steps | 8630 |
| Teacher-step savings | **21.89%** |

The old 20% opportunity threshold is barely passed, but a learned controller would need
to capture 91.35% of all privileged savings to reach it. The pre-registered R6 method
margin is therefore 30%; the current pair is **NOT READY** for source-risk model training.
Artifact: `runs/pre_c0_r6/source_risk_opportunity_qc71.json`.

## R6 execution plan

### R6-A: model-free policy-pair opportunity atlas

Before any new risk model, freeze at least two source-VLA/corrective-policy pairs and an
independent development manifest. For each pair collect Student-only and persistent
corrective outcomes plus actual teacher steps. No world-model feature is used.

Advance a pair only if all are true:

1. privileged trigger success equals persistent success;
2. privileged teacher-step savings >=30%;
3. at least 24 source-safe states, covering all four suites and at least 12 true tasks;
4. no suite contributes more than 50% of the available savings;
5. the result repeats on a second frozen collection seed.

The current SmolVLA/OFT pair fails item 2 and remains a diagnostic baseline. The server
currently contains only SmolVLA and suite-specific OFT checkpoints, so a second source
VLA must be provisioned and frozen before R6-A outcomes are observed.

### R6-B: lightweight takeover-only risk model

Only on opportunity-qualified pairs, train a shared lightweight model with:

- `P(source fails within horizon)`;
- `P(persistent corrective policy succeeds)`;
- conservative intervention-advantage head;
- source policy ID, canonical action chunk, proprioception and short visual/history input.

The deployment controller has only `CONTINUE_SOURCE` and `ENTER_PERSISTENT_CORRECTION`.
It does not hand back during the episode. Use task-bootstrap ensembles, one-sided LCB and
two-boundary dwell. Keep adapters VLA-specific but keep the risk core shared.

### R6-C: five-seed OOF gates

Require at least 4/5 seeds to satisfy simultaneously:

- success gap versus persistent >= -5pp;
- false `CONTINUE_SOURCE` on persistent-rescuable states <=5%;
- teacher-step savings >=20%;
- task-cluster bootstrap lower bound on savings >0;
- no suite failure and no single-suite shortcut.

### R6-D: multi-VLA and world-model ablations

After R6-C only, compare shared vs per-VLA, zero-shot and leave-one-VLA-out. A world model
is limited to the pre-registered multi-step residual/disagreement auxiliary feature. It
enters the main model only if it improves the state-level success-cost Pareto on held-out
tasks and both VLA pairs; AUC-only gains are insufficient.

### R6-E: independent validation and test

Collect a new validation cohort after all thresholds are frozen. Test and 100+ paired
closed-loop episodes remain sealed until validation passes. Safe handback may return only
as a separately powered future study with at least 20 recovery-created states and enough
repeats for its claimed safety level.

## Paper-position consequence

The strongest defensible paper today is a benchmark/diagnostic result about stochastic,
non-monotone recovery boundaries and the lack of learnable safe-handback opportunity for
the current pair. A multi-VLA real-time risk-control method remains a hypothesis, not a
claim, until R6-A finds two policy pairs with adequate model-free margin.

# R5 probabilistic smoke4 and RASE idea rewrite

Date: 2026-08-09  
Status: **PROBABILISTIC COLLECTION PROTOCOL READY; MODEL TRAINING REMAINS CLOSED**

## Main-line idea replacement

The repository-root idea document was replaced in place with the new project definition:

`RASE_PRE_candidate_generation_world_model_escalation_idea_2026-08-03.md`

The superseded 2,578-line candidate-generation-centered version is preserved at:

`archive/idea/RASE_PRE_candidate_generation_world_model_escalation_idea_2026-08-03_superseded_2026-08-09.md`

The new main line is a policy-conditioned, probabilistic controller for multiple frozen VLAs:

1. predict risk of the proposed source-VLA action chunk;
2. reject the chunk and enter a corrective policy only when intervention advantage is supported;
3. estimate recoverability and handback success during correction;
4. hand control back using a lower confidence bound plus dwell/hysteresis;
5. use a world model as an offline teacher/evidence source, while deploying a lightweight model.

This keeps the original real-time multi-VLA risk/correction idea, but removes unsupported
claims about deterministic labels, direct V-JEPA-delta benefit, physical rollback, and the
existing 0.6M checkpoint being deployment-ready.

## Code completed

### Probabilistic collector

`scripts/collect_r4_boundary_transitions_v3.py` now supports K Student continuations from the
same saved simulator boundary. It records independent policy seeds, outcomes, continuation
lengths, stop reasons, empirical probability and one-sided-95% Wilson bounds. Its conservative
binary compatibility field means `all_repeats_succeeded`.

`scripts/run_pre_c0_r5_probabilistic_collect.sh` exposes:

- `HANDBACK_REPEATS`;
- `BOUNDARIES`;
- suite/state filters;
- auditable eval-only opportunity-gate bypass.

### Probability losses

`rase/risk/probabilistic_losses.py` adds:

- binomial NLL from logits;
- Beta-binomial NLL for overdispersion;
- Beta posterior mean helper.

Server tests: **3/3 passed**.

### Audit and opportunity summary

`scripts/summarize_r5_probabilistic_boundaries.py` checks repeat completeness, seed uniqueness,
persistent parity, suite coverage, label entropy, Wilson LCB, continuation costs, fixed-boundary
probability and a descriptive probability oracle. It explicitly treats repeats as dependent
measurements, not independent states.

## Four-suite same-snapshot K=5 pilot

Protocol:

- split: current development-validation cohort;
- one pre-frozen state per LIBERO suite;
- boundaries h={0,64};
- five Student continuations per exact saved boundary;
- eight boundary records and 40 continuation rollouts total;
- frozen test untouched.

### Protocol QC

- persistent replay parity: **4/4**;
- valid repeat rows: **8/8**;
- repeat field completeness: **100%**;
- duplicate-seed rows: **0**;
- suite coverage: **4/4**;
- protocol gate: **READY**.

### Outcomes

| Suite | h0 | h64 | Interpretation |
|---|---:|---:|---|
| Spatial | 4/5 | 0/5 | strongly non-monotonic; more OFT can reduce handback success |
| Goal | 3/5 | 4/5 | probabilistic improvement, but still uncertified |
| Object | 0/5 | 0/5 | stable persistent-only candidate |
| Long | 0/5 | 0/5 | stable persistent-only candidate |

Aggregate:

- nondegenerate boundaries: **3/8 = 37.5%**;
- mean empirical handback probability: **0.275**;
- maximum one-sided-95% Wilson LCB: **0.435**;
- boundaries with LCB >=0.5: **0/8**;
- successful continuation length: mean 118.1, range 82--164;
- failed continuation length: mean 321.8, range 206--510;
- best fixed empirical boundary: h0, mean p=0.35;
- descriptive per-state probability oracle: p=0.40;
- oracle minus best fixed: +5pp over only four states; no inferential claim;
- oracle uses 64 total OFT steps because only Goal benefits from h64;
- Spatial is the observed non-monotonic state.

The separately executed Goal K=5 run produced h64=5/5, while the four-suite run produced
h64=4/5. Since h64 is reached through a fresh OFT prefix, this exposes prefix-trajectory
sensitivity in addition to same-snapshot Student stochasticity.

## Interpretation

1. The revised idea is supported as a research question: fixed duration is not sufficient,
   because Spatial and Goal require opposite decisions at h64.
2. K-repeat labels are necessary: three of eight boundaries have probabilities strictly
   between zero and one.
3. K=5 is insufficient for safety certification: even 4/5 yields LCB 0.435.
4. The negative Object/Long results prevent a broad claim that early handback is generally
   available.
5. World/state evidence remains potentially useful because elapsed duration is non-monotonic,
   but no world-model benefit is claimed until it beats the history-only controller.
6. The current 24-state `val` cohort has now influenced method design. It is development/
   calibration data, not final independent validation. The original test split remains frozen;
   final validation must come from newly collected tasks/states.

## Next-stage execution order

### R5-A16: label-entropy expansion

- 16 development states: four per suite;
- balance true tasks and historical finite-safe/persistent-only states where possible;
- h={0,16,64,128}, K=5;
- record same-snapshot repeats and a separate prefix-repeat ID;
- analyze label entropy, duration non-monotonicity, suite effects and compute cost.

Do not train the new controller before A16 completes. Eight boundary rows cannot support a
probability model.

The cohort is now frozen in `runs/pre_c0_r5/probability_pilot16_manifest_v1.json`: 16 fresh
states excluded from smoke4, four suites, eight true tasks, two states per task, and 12/16
historically finite-safe states. `scripts/run_r5_probability_pilot16.sh` is the versioned
launcher. It has not been used to open frozen test data.

### R5-B: probability model

- Beta-binomial handback head;
- persistent-success head;
- source failure-risk head;
- remaining-cost quantiles;
- task-bootstrap ensemble;
- lower-confidence-bound decision plus two-boundary dwell;
- five-seed nested task-held-out OOF.

Required gates: success gap >=-5pp, conditional false handback <=5%, savings >=20%, at least
4/5 seeds pass, and task-cluster intervals support the constraints.

### R5-C/R6: scale and multi-VLA

- at least 300 development states;
- at least 100 independent persistent-rescuable calibration states;
- new independent validation cohort;
- second source VLA through `CanonicalActionChunk`;
- shared vs per-VLA heads and leave-one-VLA-out;
- only then test preregistered world-model residual/disagreement features.

Frozen test and large paired closed-loop evaluation remain blocked until validation gates pass.

## Artifacts

- `runs/pre_c0_r5/boundary_prob_smoke4_k5_v1/`
- `runs/pre_c0_r5/boundary_prob_smoke4_k5_v1/probabilistic_summary.json`
- `scripts/collect_r4_boundary_transitions_v3.py`
- `scripts/run_pre_c0_r5_probabilistic_collect.sh`
- `scripts/summarize_r5_probabilistic_boundaries.py`
- `rase/risk/probabilistic_losses.py`
- `tests/test_probabilistic_losses.py`
- `scripts/freeze_r5_probability_pilot16.py`
- `scripts/run_r5_probability_pilot16.sh`
- `runs/pre_c0_r5/probability_pilot16_manifest_v1.json`

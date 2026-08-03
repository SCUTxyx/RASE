# RASE idea evolution and next questions

**Date:** 2026-07-31  
**Status:** current scientific narrative after W9C + W10; raw artifacts remain authoritative

## 1. Current outcome

W9C repaired a data-validity bug, not a weak numerical result. W9B used
LIBERO-Plus indices 0–9 while labeling them as clean-10; those indices are layout
variants rather than the official clean tasks. Exact-name vanilla BDDL/init loading
restored alignment: probe mean SR was 0.6125 (Spatial 0.45, Object 0.85, Goal 0.80,
Long 0.35), clean32 coverage completed, and episode/task readiness both passed.
This unblocked the preregistered ridge test; it did not validate the method.

Ridge matched, rather than beat, the action-matched random baseline on both held-out
splits (Δutility=0; confidence intervals cross zero). Therefore the preregistered
outcome is `kill_method_branch`: retain the benchmark/diagnosis contribution, do
not promote a learned selector, and do not escalate to MLP or RL.

W10 then tested whether Object/Spatial L1/L2 failures show the same policy-relative
recovery seen on Goal/Long. They mostly do not: collection was 80/80 failure, direct
Smol 0/16, direct OFT 1/16 (Spatial 0/8), and escalate-oracle support was only one
state, so the held-out split is `NOT_READY`. The diagnosis track remains, but
recoverability claims must be suite-scoped rather than treated as universal across
LIBERO suites.

## 2. Evidence-driven idea evolution

1. **Unified learned fallback space.** The initial idea was to sample candidate
   actions and learn which fallback to execute. This remained an engineering
   hypothesis, not an established novelty claim.
2. **Candidate recoverability.** W3/W4 repeatedly found Smol continuation near zero,
   while OFT recovered matched states (W4: Smol 0/1536 candidate outcomes; OFT
   portfolio 17/32 states, 65/256 candidates). The scientific object changed to
   policy-relative recoverability.
3. **Temperature/diversity explanation.** W5 varied proposal temperature across
   0.3/0.7/1.0; Smol remained 0/576 despite increasing endpoint diversity. This
   closed the “sample more diversely” explanation in the tested regime.
4. **Candidate-prefix mechanism.** W6/W7 direct, zero-prefix and candidate-prefix
   controls found candidate-specific rescue 0/2 in discovery and 0/8 held-out.
   Strong-policy continuation or passive elapsed time explained the positives;
   candidate content did not.
5. **Deployable direct escalation.** W7 failure-held-out Smol was 0/24 and
   prefix+OFT any-of-8 was 8/24; W8 direct OFT was 9/24. State pairing was
   both 7 / prefix-only 1 / direct-only 2 / neither 14, with prefix-vs-direct
   McNemar p=1.0. Direct escalation is deployable, but is not proven superior to
   the portfolio.
6. **Readiness before learning.** W9A/W9B correctly stopped on invalid controls.
   W9C fixed identity and passed probe/coverage/readiness, showing that readiness
   is a data-validity gate rather than a method result.
7. **Minimal selector falsification.** Ridge failed the preregistered held-out
   matched-random comparison. The method branch ends here; added model capacity
   cannot repair an unsupported generalization claim.
8. **Suite heterogeneity.** W10 Object/Spatial failure coverage found almost no
   OFT-only recovery (1/16). Recoverability is not only policy-relative; under the
   tested mild camera/robot perturbations it is also strongly suite-dependent.

## 3. Hypotheses falsified or not supported

- Smol proposal temperature alone explains zero recovery: falsified in the tested
  W5 frontier (0/576).
- Candidate-specific Smol prefixes cause the observed OFT rescue: falsified in the
  tested discovery and held-out ablations (0/2 and 0/8).
- Any-of-K portfolio labels are valid deployable selector actions: false by action
  semantics; they remain diagnostic/oracle upper bounds.
- Failure-only data are sufficient for a cost-aware escalation selector: rejected
  by readiness because clean regret and collapsed routing cannot be assessed.
- The W9B ~7.2% control result reflects official clean-10 competence: invalidated by
  wrong task identity.
- Passing probe/coverage/readiness implies a useful learned selector: falsified by
  the downstream ridge gate.
- A more expressive MLP/RL model is warranted after ridge failure: not supported;
  explicitly prohibited by the preregistered kill criterion.
- Object/Spatial L1/L2 failures are broadly recoverable by direct OFT: not
  supported (1/16; Spatial 0/8); split escalate-oracle support is insufficient.

These statements are bounded to the tested policies, tasks, perturbations, horizons
and cohorts; they are not universal impossibility claims.

## 4. Allowed claims

- Recoverability is policy-relative in the evaluated failure-conditioned cohorts:
  weak-policy continuation fails while strong-policy continuation succeeds on a
  subset of matched snapshots.
- W7/W8 provide episode-disjoint paired evidence for prefix+OFT and direct-OFT
  outcomes, with direct escalation a real deployable arm.
- Candidate-specific rescue was absent in the tested controlled ablations; most
  observed rescue is consistent with stronger continuation policy or passive time.
- W9C repaired clean task identity and passed the preregistered alignment,
  coverage, and readiness gates.
- The preregistered ridge method gate failed; reporting this negative result is part
  of the benchmark/diagnosis contribution.

## 5. Forbidden or unsupported claims

- “RASE learned when to escalate,” “the selector generalizes to unseen tasks,” or
  “ridge improves utility.”
- Direct OFT is statistically better than prefix+OFT based on 9/24 vs 8/24; their
  paired comparison is p=1.0.
- Any-of-K is a deployable action or valid training proxy.
- Candidate actions causally rescue failures, or zero rescue proves candidates can
  never help outside this regime.
- Failure-conditioned recovery is unconditional task success, clean performance,
  or evidence across all suites/phases/levels/backbones.
- Readiness PASS is method PASS, or identity repair retroactively validates W9A/B.
- MLP/RL is the next justified method step.

## 6. Exact task-held-out limitation

`runs/ngc_w9c_selector_task_splits.json` groups by `task_id` without leakage, but
its test partition is narrow:

- **8 states total, all clean-control**: Long 3, Object 2, Spatial 3;
- **0 failure-challenge states** and no Goal test states;
- learned actions are **7 continue, 0 escalate, 1 abstain**;
- learned utility equals action-matched random utility: Δ=0, 95% bootstrap CI
  [-0.0075, 0.0075], n=8.

Consequently this split tests neither failure detection nor escalation on unseen
tasks. It supports only the preregistered decision that the method did not clear
its task-held-out gate. “8 clean, no failure, 0 escalation” must accompany any
mention of this result.

## 7. Benchmark questions for the next phase

W10 answered the Object/Spatial L1/L2 direct-escalation coverage question mostly
negatively. Remaining questions:

1. **Claim contraction vs new coverage:** Should positive recoverability claims stay
   Goal/Long-centered, or is a new preregistered Object/Spatial regime (different
   t0 / perturbation / policy pair) justified?
2. **Mechanism accounting:** What fraction of prefix positives are also direct or
   zero-prefix positives, with paired uncertainty rather than marginal rates?
3. **Coverage sensitivity:** Which claims survive when restricted to balanced cells,
   and which are driven by Goal/Long sampling?
4. **External validity:** Does any recoverable asymmetry reproduce on a second
   policy pair? No such result is currently claimed.
5. **Benchmark validity:** Keep exact task identity, episode grouping, provenance and
   action semantics invariant under re-export/reanalysis.

Future work should prioritize analyses that sharpen where evidence is present versus
absent. Any new learned method requires a new preregistration and new data design;
it is not a continuation of the killed W9C method branch, and W10 does not reopen it.

## 8. Evidence index

- `runs/ngc_w9c_clean_probe_combined_audit.json`
- `runs/ngc_w9c_clean_control_state_keys.json`
- `runs/ngc_w9c_selector_gate_summary.{json,md}`
- `runs/ngc_w9c_selector_episode_splits.json`
- `runs/ngc_w9c_selector_task_splits.json`
- `progress/2026-07-31_w9c_clean_task_identity_fix.md`
- `progress/2026-07-31_w9c_selector_gate_result.md`
- `progress/2026-07-31_paper_claim_freeze.md`
- `progress/2026-07-31_w10_object_spatial_benchmark.md`
- `runs/ngc_w10_object_spatial_episode_splits.json`
- `runs/ngc_w10_object_spatial_heldout_action_dataset.jsonl`

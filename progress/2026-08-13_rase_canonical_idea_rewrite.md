# RASE canonical idea rewrite after R10-B

Date: 2026-08-13

The repository's canonical idea has been rewritten to incorporate every
formal result through R10-B.  The core project goal is retained: lightweight
real-time risk estimation for multiple frozen VLAs, followed by conservative
source/fallback policy switching, with optional world-model dynamics evidence.

The new definition changes the statistical problem rather than abandoning the
idea:

- source risk is policy-, action- and history-conditioned;
- fallback recoverability is a stochastic probability distribution rather
  than a deterministic boundary label;
- opportunity-window loss is modeled as a censored survival/hazard process;
- source-risk, fallback-success, intervention-advantage, cost and abstention
  are distinct outputs;
- pre-entry inference does not require a free fallback action query; any such
  query is an explicitly costed baseline;
- the main multi-VLA claim is a shared core plus outcome-free behavior
  descriptor and small calibration, not one zero-shot threshold;
- risk generalization and selector admissibility are separate claims;
- world-model features are limited to preregistered multi-step residual and
  disagreement evidence and remain optional until state-level cross-VLA Pareto
  gains are established.

The document explicitly records the negative evidence from R3, PRE-A1/A2,
R6-C, R7-A, R9-C and R10-B.  It forbids deterministic small-K labels,
case-control prevalence claims, deleting unstable states, pooled policy priors
as universal risk, physical rollback claims, and world-model escalation after
an information-gate failure.

Canonical local artifact:
`outputs/RASE_CANONICAL_IDEA_stochastic_multi_vla_risk_control_2026-08-13.md`.

SHA256:
`3a39df2760fa559ef0f68f87a792627a8e9860c616d2b6edca4cd4c707ed9acf`.

On the server, this revision replaces the historical root canonical path
`RASE_PRE_candidate_generation_world_model_escalation_idea_2026-08-03.md`.
The previous file is preserved under `archive/idea/` before replacement.  A
descriptively named copy is also kept at
`outputs/RASE_CANONICAL_IDEA_stochastic_multi_vla_risk_control_2026-08-13.md`.

This idea rewrite does not unlock model training, selector, world-model,
validation or test.  The current R10-B full OFT action-trace root-cause
diagnostic remains the only running experiment.

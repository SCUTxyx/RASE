# CVPR 2027 canonical document refresh and execution overlay

Date: 2026-08-13

## Change

The current project guidance is now:

- `outputs/RASE_IDEA_DETAILED_CVPR2027.md` — canonical idea;
- `plan/RASE_NEXT_STEPS_EXECUTION_PLAN_CVPR2027.md` — canonical execution plan;
- `plan/RASE_RL_AND_PUBLIC_DATASETS_DETAILED_PLAN.md` — RL/public-data plan;
- `plan/RASE_CURRENT_EXECUTION_OVERLAY_2026-08-13.md` — mapping from the
  long-form plans to the live experimental state and locks.

The old canonical output, root idea/next-step files, RL selector guide and
indexes were copied before replacement to:

`archive/superseded_docs/2026-08-13_cvpr2027_refresh/`.

Legacy paths remain populated with the new corresponding content so existing
references do not silently lead to superseded guidance.

## Evidence alignment

This refresh does not override any experimental verdict. In particular:

- R7-A final-failure prediction remains FAIL 0/5;
- R8-B local hazard remains FAIL 0/5;
- R9-C information support remains FAIL;
- R10-B deterministic case/control reproducibility remains FAIL;
- the R10-B post-failure probability diagnostic remains insufficient to
  unlock a probabilistic risk model;
- selector, RL, world model, validation and test remain locked.

## Current execution

The completed full-action trace diagnostic reports
`CLOSED_LOOP_TRACE_DIVERGENCE` and selects `AUDIT_CHUNK_INPUT_DIVERGENCE`.
Hash-only per-chunk input/output instrumentation passed seven focused tests and
a real one-group/K3 smoke. The full frozen 18-group/K3 diagnostic is running in
tmux `r10b_chunk_full` and will automatically write
`runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json`.

The result determines whether the next root-cause branch is restore/observable
state, later closed-loop observation amplification, OFT inference variance, or
no reproduced chunk divergence. None of these branches automatically unlocks
model training; a new target/data protocol and model-free operator-opportunity
gate are required first.


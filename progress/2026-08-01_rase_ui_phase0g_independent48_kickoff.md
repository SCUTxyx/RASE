# RASE-UI Phase 0G independent48 timing-opportunity kickoff

Date: 2026-08-01 19:00 CST  
Status: **PREREGISTERED — collection and outcomes not yet opened**

## Objective

Test whether the immediate-versus-deferred OFT timing complementarity observed
in Phase 0E transfers to tasks and episodes absent from Phases 0C–0F. This is a
screening experiment, not independent confirmation and not selector training.

## Frozen cohort design

- 48 episodes and 48 selected decision states.
- Four LIBERO suites.
- Three cells per suite: clean L0, camera L1, robot L1.
- Four tasks per suite/cell.
- Exactly one selected step-2 state per task and episode.
- Expected active suffix length: five actions.
- Collection seed: `2026081807`, selected using task/catalog metadata only.
- All 48 planned task ids are unique and have zero overlap with the frozen
  Phase 0C–0F 16-task cohort.
- Selection does not use source or intervention outcomes.

The exact planned task/episode/cell table must be written before collection and
matched exactly after key export. Any task reuse, cell drift, key checksum
failure, wrong decision step, or wrong suffix length stops the run.

## Operators

Primary timing comparison:

1. immediate OFT;
2. exact active-suffix-then-OFT.

Strict CONTINUE is retained for the exact three-operator coverage join. REPLAN
is excluded because it has no unique success support in prior opportunity
cohorts and is not part of the timing-opportunity gate.

## Preregistered gate

The screen is `ready` for a separate 96-state confirmation only if all hold:

- immediate-only success on at least two distinct tasks;
- deferred-only success on at least two distinct tasks;
- same-state timing oracle minus best fixed timing is at least `0.05`;
- exact state/task/episode coverage and prefix parity pass.

Report the task/episode bootstrap 95% interval for the oracle gap, but do not
change the gate after observing it. A failed gate stops selector and world-model
training. A passed gate authorizes protocol freezing and new data collection,
not model training on this screening cohort.

## Frozen command

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=1 TAG=v1 ./scripts/run_rase_ui_phase0g_independent48.sh
```

# W6 L1–L2 collection and coverage gate

## Status

Collection and failure-challenge coverage passed. Outcome-balanced Plus
coverage did not pass and is not pursued by brute-force top-up.

## Frozen evidence

- Config: `configs/collect_w5_l1_l2_camera_robot.json`
- Pool: `pool/ngc_w5_l1_l2_camera_robot`
- Episodes: 40, seed 20260727
- Retained states: 762
- Episode outcomes: 0 success / 40 failure
- Eligible distinct failure episodes after `min_remaining_steps=100`:
  - camera L1: 11
  - camera L2: 9
  - robot L1: 8
  - robot L2: 12
- Audit exclusions, missing metadata, missing episode groups, and outside-bin
  records: all zero
- Frozen pilot: 8 states, two globally distinct episodes per `dim×level` cell
- State-key SHA-256: `6cc29a7d5613d05a57bfb2570afbf7ea5059e3c80ea84246efdc6d4c1ef8a085`

## Decision

The full Plus collapse map previously measured camera L1/L2 at 0/558 and robot
L1/L2 at 5/537. Requiring two SmolVLA-success episodes in every Plus cell would
therefore be an inefficient and potentially impossible gate, especially for
camera. W6 is frozen as a failure-conditioned challenge cohort for a paired
Smol→Smol versus Smol→OFT diagnostic. It cannot support unconditional NGC-rate
or clean-regret claims.

Clean-success controls will be collected and reported as a separate cohort
before selector training/evaluation. This avoids mixing task distributions and
keeps the failure-challenge denominator explicit.

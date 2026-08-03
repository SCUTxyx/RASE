# W6 L1–L2 paired policy matrix

## Status

**Complete diagnostic pilot.** The matched Smol→Smol and Smol→OFT one-shot
matrix is complete for all 8 frozen states and 64 candidate prefixes per arm.
OFT recovered 2/8 states; Smol recovered 0/8. This is directional mechanism
evidence, not a powered superiority claim.

## Frozen inputs

- Cohort: failure-conditioned L1–L2 challenge set
- State-key artifact: `runs/ngc_w6_l1_l2_state_keys.json`
- States: 8; two distinct episodes in each camera/robot × L1/L2 cell
- State-key SHA-256:
  `6cc29a7d5613d05a57bfb2570afbf7ea5059e3c80ea84246efdc6d4c1ef8a085`
- Candidate artifacts: `runs/ngc_w6_l1_l2_candidates_t07`
- Candidate-directory SHA-256:
  `2cfa1f27c66d83860b84c40c0b5fc80cda0cb41b5415478e1eba25b7079b5013`
- SmolVLA checkpoint SHA-256:
  `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`
- Repo Git SHA recorded by the runner:
  `ea7ad403c002302234cf7aa81476bb869e86b586`
- `env.lock.md` SHA-256:
  `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2`

## Candidate generation audit

- 8/8 artifacts written; no skips
- Shape per state: `[8, 10, 7]`
- Temperature: 0.7
- Mean pairwise endpoint L2: 1.7568
- Mean pairwise chunk L2: 1.7832
- Minimum pairwise endpoint L2: 0.3357

The candidate artifacts are non-collapsed and suitable for the paired screen.

## Smol→Smol result

- Summary: `runs/ngc_w6_l1_l2_screen_t07/summary.json`
- Completed: 64/64 one-shot candidate rollouts
- Candidate hits: **0/64**
- Portfolio state hits: **0/8**
- All four dim×level cells: 0/2 portfolio state hits
- Retries: 0
- Mean rollout time: 12.26 s; median 10.80 s; p90 23.66 s
- Wall time: 1072.8 s

This is valid negative screen evidence. The eight `uncertain` diagnostic labels
are expected because one-shot screening cannot assign Wilson Set A/B/C labels.
Since the hit union is empty, no Smol formal confirmation run should be launched.

## Completed paired result

- Smol candidate hits: **0/64**
- OFT candidate hits: **10/64** (descriptive; candidates are not independent)
- Smol portfolio state hits: **0/8**, Wilson 95% `[0.000, 0.324]`
- OFT portfolio state hits: **2/8**, Wilson 95% `[0.071, 0.591]`
- State pairs: both-hit=0, Smol-only=0, OFT-only=2, both-miss=6
- Exact two-sided state-level McNemar: `p=0.5`

The ten OFT candidate hits are concentrated in two states. Long/camera-L1
`sp1_9224df57d14685d1a38858b3fa311f17` has 8/8 OFT hits, consistent with an
OFT-sufficient state whose success may not depend on the candidate prefix.
Goal/robot-L2 `sp1_f0c5d3a950eb6035ce142f78aca29ddd` has 2/8 OFT hits (`c4,c5`),
which is the pilot's candidate-dependent rescue signal. Six states remain
both-miss. The failure-conditioned cohort does not estimate unconditional NGC.

The earlier terminal-rollout fingerprint bug was fixed by using a fresh
environment for every candidate. All resumed records passed provenance checks.

## Decision

1. Freeze W6 unchanged; do not rerun or regenerate it.
2. Run W7 prefix attribution on the two OFT-only states with direct OFT and a
   time-matched zero prefix as controls. Only `direct=false`, `zero=false`, and
   candidate success qualifies as candidate-specific rescue.
3. Freeze a held-out 24-state cohort (6 per dim×level cell) that excludes every
   episode group represented in W6, not only its exact snapshots. Run the same
   paired matrix without tuning on W7 outcomes.
4. Treat state as the inferential unit. Candidate counts remain descriptive.
5. Build the escalation selector only after attribution and held-out validation;
   clean-success controls remain a separate cohort for regret/cost measurement.

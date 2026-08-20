# PRE-C0 protocol freeze — 2026-08-04

## PRE-A3 boundary

- finite handback method gate: **NOGO**
- blocking result: val adaptive finite-duration headroom = 0.0pp
- PRE-A3 hidden test24: **sealed / not unblinded**
- PRE-B, termination selector, candidate critic, and world model: **closed**

## PRE-C0 pivot

PRE-C0 tests early-deviation same-policy correction, not OFT handback timing.
Runtime control remains the frozen SmolVLA checkpoint.

- protocol: `protocol/pre_c0_same_policy_corrective_v1.md`
- design: `runs/rase_pre_c0_design24_v1.json`
- design SHA256:
  `981d9599622d5823ab619cb3225abe133a8dd53a20266c3b031752790df13efe`
- source: 24 PRE-A3 train tasks only, six per suite
- cells: 8 clean:L0 / 8 camera:L1 / 8 robot:L1
- outcomes used for selection: no
- main audit stages: paired T1 first-deviation and T3 failure-in-progress

The older `runs/rase_pre_a3_s_opportunity_spec_v1.json` remains unchanged and
must not be merged with PRE-C0 gates.

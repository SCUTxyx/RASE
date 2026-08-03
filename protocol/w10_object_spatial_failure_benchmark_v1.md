# W10 Object/Spatial failure benchmark protocol v1

This preregistration extends benchmark coverage and diagnosis only. It does not
train, tune, or evaluate a selector. The two primary episode groups are eight
`Spatial` failures and eight `Object` failures, sampled before rollout outcomes
are observed. State-level direct SmolVLA and direct OFT outcomes are paired.

The frozen selection inputs are in
`configs/w10_object_spatial_failure_schedule.v1.json`; its SHA-256 is pinned by
the W10 config and sidecar. Any change to seed, strata, support threshold,
identity rule, exclusion inputs, policy hash, or rollout counts requires a new
protocol version. The selected-key hash is intentionally unset while the gate
is blocked; it must be filled and frozen before the first rollout, never in
response to benchmark outcomes.

Hard gates:

1. exactly 8 episode-distinct failure states per suite and 16 total;
2. every selected state has non-empty `task_id`, `libero_flavor`, and
   non-negative `init_state_id`;
3. no selected task/init identity appears in any declared prior pool and no
   selected episode group appears in a declared prior state-key artifact;
4. state-key artifact hash equals the preregistered hash;
5. frozen K=8, temperature=0.7, one direct Smol rollout and one direct OFT
   rollout per state; no outcome-adaptive replacement or stopping.

Current preregistration status is **BLOCKED**. The legacy W5 target pool lacks
`init_state_id`, and post-exclusion inventory is below 8 groups per suite.
Collection of new Object/Spatial failures needs a future versioned collector
schedule with task/init identity and cross-pool exclusion enforced before GPU
collection. Existing `collect_state_pool.py` only gives schedule semantics to
W9B/W9C clean controls, so no executable W10 collection command is claimed.

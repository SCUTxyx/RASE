# PRE-C1.2 R0 Decision

- branch: `revised_short_horizon_training`
- rationale: Default revised path: early recoverable student states + short-horizon objective; legacy E3/E4 remains paused.
- legacy E3/E4 allowed: `False`
- capacity ladder allowed: `False`
- coverage ok: `True` (9/9 anchors)

## Teacher-forced

- original adapted/base loss: `0.0280` / `1.0027` good=`True`
- R1 query adapted/base loss: `0.0598` / `1.1238` good=`True`

## Recoverability

- R(OFT,0)=1.000
- R(base,1)=0.889 R(adapted,1)=0.889
- R(base,4)=0.556 R(adapted,4)=0.667
- decay_fast=`False`
- P(OFT success|student query)=`0.276`

## Next actions

- `build_early_query_dataset`
- `residual_or_prefix_weighted_short_horizon_train`
- `gate_on_R_k_before_terminal_8pp`

## Curves

```json
{
  "adapted": {
    "0": 1.0,
    "1": 0.8888888888888888,
    "16": 0.4444444444444444,
    "2": 0.6666666666666666,
    "4": 0.6666666666666666,
    "8": 0.5555555555555556
  },
  "base": {
    "0": 1.0,
    "1": 0.8888888888888888,
    "16": 0.4444444444444444,
    "2": 0.7777777777777778,
    "4": 0.5555555555555556,
    "8": 0.4444444444444444
  },
  "oft_k0": 1.0,
  "oft_replan_k1": 0.7777777777777778
}
```

## Teacher-forced buckets

```json
{
  "original_c1_1": {
    "adapted_loss_full": 0.028020873085404057,
    "adapted_loss_prefix_2": 0.11794449108302968,
    "adapted_loss_prefix_4": 0.10747945534553308,
    "adapted_loss_tail": 0.01457239368721023,
    "adapted_minus_base": -0.9746984220678031,
    "adapted_prefix_gripper_error": 0.250792862801111,
    "adapted_prefix_rotation_error": 0.04697751511043187,
    "adapted_prefix_translation_error": 0.12021025418889883,
    "base_loss_full": 1.0027192951532071,
    "base_loss_prefix_2": 0.616111965217231,
    "base_loss_prefix_4": 0.5715913333034401,
    "base_loss_tail": 1.0951537853507904,
    "base_prefix_gripper_error": 0.7711990384429411,
    "base_prefix_rotation_error": 0.4573648427345714,
    "base_prefix_translation_error": 0.6192818966707544,
    "n": 418
  },
  "r1_student_query": {
    "adapted_loss_full": 0.05983641630612037,
    "adapted_loss_prefix_2": 0.2749056888745197,
    "adapted_loss_prefix_4": 0.2570930002974,
    "adapted_loss_tail": 0.01344585581826939,
    "adapted_minus_base": -1.0639338197084605,
    "adapted_prefix_gripper_error": 0.7122505716386603,
    "adapted_prefix_rotation_error": 0.051012576537010496,
    "adapted_prefix_translation_error": 0.3114542151937908,
    "base_loss_full": 1.123770236014581,
    "base_loss_prefix_2": 0.7757710471503005,
    "base_loss_prefix_4": 0.6877374293605897,
    "base_loss_tail": 1.2249094285378772,
    "base_prefix_gripper_error": 1.6558127682297663,
    "base_prefix_rotation_error": 0.5183224881852954,
    "base_prefix_translation_error": 0.5344605503665391,
    "n": 484
  }
}
```

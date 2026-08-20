# R6-A Policy-Pair Opportunity Lock

Date: 2026-08-09

## Why R6-A precedes another risk model

The prior 71-state handback cohort had a privileged teacher-step ceiling of
21.893%, below the new 30% method-margin requirement. The 24-task boundary
protocol was stable but exposed only four finite-safe states and one genuinely
recovery-created opportunity. These results reject further model tuning on the
same policy pair: model capacity cannot manufacture a missing model-free
opportunity.

R6-A therefore changes the independent variable from model architecture to the
source/corrective policy pair. No learned gate is trained in this stage.

## Frozen cohort and corrective policy

- 48 exact reset snapshots, 48 true tasks, 12 tasks per suite;
- 16 clean, 16 camera-L1, and 16 robot-L1 states;
- every snapshot is at policy step zero, before any source-policy action;
- suite-specific deterministic OpenVLA-OFT is reused as the persistent
  corrective policy;
- the cohort remains development-only and is excluded from validation/test.

Frozen input identities are recorded in
`configs/r6a_policy_pair_manifest_v1.json`.

## Frozen source policies

1. `smolvla_libero`: diagnostic baseline;
2. `pi0fast_libero`: `lerobot/pi0fast-libero`, pinned revision
   `840f4b503f4c09110421c33c810a85b6684fd658`;
3. `pi05_libero`: `lerobot/pi05_libero_finetuned_v044`, pinned revision
   `8e174154ef5f6c60a8da12ae99c303d8963138c1`.

Both new policies use relative 7-DoF LIBERO actions and are evaluated through
the same LeRobot environment preprocessing/postprocessing path. X-VLA is not in
this lock because its official LIBERO recipe uses absolute actions, which is
not exchangeable with the current relative-control snapshot/OFT protocol.

## Opportunity gates

For each policy and each of two source-policy seeds:

- privileged trigger success is no worse than persistent OFT;
- privileged teacher-step savings are at least 30%;
- source-safe support contains at least 12 distinct `suite:task_id` tasks;
- all four suites contain source-safe support;
- no one suite accounts for more than half of saved teacher steps.

A policy pair passes only if every gate passes in both seeds. Shared multi-VLA
risk training remains locked until at least two policy pairs pass.

## Locked code identities

- `rase/collect/forked_rollout.py`: `ccc130d0a7ef914f78736ffc5dbc4068fbdfaa3b1ae49e1eb56e1e1166bc1ea2`
- `rase/backends/lerobot_libero_plus.py`: `fde41f9c63074e8238c7bde12655d7aed6464520a4b38b7ee789269f74793963`
- `scripts/rollout_lerobot_source_from_initial_states.py`: `eaf97b877b0a4008f1482d09cea5e28bd75434e44872f4010d151e0d9d51aafa`
- `scripts/audit_r6a_policy_pair_atlas.py`: `71d203b27ff6a5d56b679b7af96603a77dc7f962370448b198f03af234773d8a`
- `configs/r6a_policy_pair_manifest_v1.json`: `01dfb382f8679981416187050beaa37be95aade1bdd2ede089f4ecd32fd10680`

## Conditional downstream sequence

Only after the atlas gate is READY:

1. train a shared multi-VLA risk/handback model with five task-bootstrap seeds;
2. report shared versus per-VLA, zero-shot, and leave-one-VLA-out evaluation;
3. test only the preregistered multi-step residual/disagreement world-model
   ablation, retaining it only for state-level Pareto gain;
4. recollect independent validation, and only then unseal test and begin 100+
   paired closed-loop episodes.

## Dependency-closure amendment

Before any Pi0Fast or Pi0.5 source outcome was produced, inspection of their
checkpoint processors showed runtime references to a PaliGemma text tokenizer
and, for Pi0Fast, a FAST action tokenizer. They are now pinned locally. The
public PaliGemma replica is accepted only because its tokenizer blob IDs and
SHA-256 values are byte-identical to the canonical gated Google repository.
This amendment changes no cohort, policy weights, seed, gate, or analysis.

Amended code identities used for the two new VLAs:

- `rase/collect/forked_rollout.py`: `ede97f4e57e998269e83f467c3aa932d6c2c1145418a77d31fc008b3dc80ddea`
- `rase/backends/lerobot_libero_plus.py`: `b365edc6ef42a7e0c8ea09ba6a8f94ccfe8781a1559fe706535f573bb10eb1de`
- `scripts/rollout_lerobot_source_from_initial_states.py`: `a5994ba49b543c7895bfcc3d8c41b8bdfe6589f236f406da4bf0003119582e1d`
- `configs/r6a_policy_pair_manifest_v1.json`: `5ae5cd41cdf787d482cf91330a8787527bf74b7cb4a9ca9e2bca50389dd15e5d`

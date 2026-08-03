# RASE recovery dataset schema v1

`scripts/export_recovery_dataset.py` writes one JSONL row per
`(state_key, candidate_id)` using schema `rase-recovery-dataset/v1`.

Required identity and provenance fields:

- `state_key`, `candidate_id`, `candidate_seed`
- `candidate_artifact`, `candidate_shape`, `candidate_temperature`
- `candidate_policy_hash`
- `task_id`, `instruction`, `suite`, `perturb_dim`, `level`, `t0`
- `episode_id`, `perturb_sub`, `state_seed`
- `episode_outcome`

Dual-track labels:

- `set_label_smolvla`: Wilson A/B/C/uncertain under the stochastic SmolVLA
  continuation protocol.
- `dual_track_label`: cross-oracle split such as `consensus_recoverable`,
  `smol_only`, `oft_only`, `both_fail`, or `uncertain`.
- `recoverable_smolvla`, `successes_smolvla`, `trials_smolvla`
- `recoverable_oft`, `successes_oft`, `trials_oft`

The OFT fields represent deterministic one-shot candidate verification. They
must not be interpreted as a Wilson-certified probability above `tau`.

Optional QC fields:

- `trace_manifests`: portable JPEG trace manifests
- `videos`: MP4 files rendered from those traces

The companion `*.splits.json` groups unique state keys by dual-track label.
The companion `*.benchmark-splits.json` is the train/validation/test manifest.
It assigns all snapshots from the same `(task_id, episode_id)` together and
greedily balances suite, perturbation dimension, level, and dual-track label.
Its `audit` section records exact group and stratum counts. Training and
evaluation must use this episode-grouped manifest, never candidate-row or
state-level random splitting, to avoid temporal leakage.

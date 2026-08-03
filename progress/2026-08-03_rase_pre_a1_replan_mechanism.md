# RASE-PRE PRE-A1 replan mechanism audit — 2026-08-03

## Status and question

Status: `persistent_fallback_required` on a development-only 12-state audit.

Question: does resampling fail merely because SmolVLA proposes a poor first
chunk, such that one short expert replan can move the environment into a state
where frozen SmolVLA succeeds? This distinguishes a trainable recovery-prefix
problem from a persistent closed-loop policy-capability problem.

No model was trained. SmolVLA and all four suite-specific OFT checkpoints were
frozen.

## Mechanism and controls

The exact 12 outcome-independent PRE-A0 states were reused: four suites ×
`clean:L0 / camera:L1 / robot:L1`, with one unique task/episode per cell.

For every state, its suite-specific OFT policy generated one 8-action chunk.
The environment was then reset to the identical snapshot for four paired arms:

- zero OFT steps, then SmolVLA;
- one OFT step, then SmolVLA;
- four OFT steps, then SmolVLA;
- all eight OFT steps, then SmolVLA.

All four arms used the same per-state SmolVLA continuation seed. OFT and Smol
were loaded sequentially rather than simultaneously. Candidate files record
shape and suite-specific oracle provenance. Full direct OFT outcomes came from
the already-completed Phase 1A exact state join.

## Results

- Smol baseline (`h=0`): 3/12.
- `h=1`: 3/12.
- `h=4`: 3/12.
- `h=8`: 3/12.
- short-prefix oracle over `h∈{1,4,8}`: 3/12.
- short-prefix rescues: 0/9 base failures, across 0 tasks.
- direct, persistent OFT: 11/12.
- direct-only rescues relative to both base and short-prefix oracle: 8.

Every state had identical success/failure across all four prefix lengths. The
three successes were the same clean states; camera:L1 and robot:L1 had no
short-prefix rescue.

## Causal interpretation

The negative result is stronger than another same-policy resampling failure.
Even the first chunk from a policy that succeeds under persistent closed-loop
execution cannot rescue frozen SmolVLA after handback. Therefore the dominant
failure mechanism is not local initial-action noise. SmolVLA repeatedly returns
to a systematic failure policy under the perturbed visual/robot distribution.

PRE-A0 already showed that stochastic Smol candidates were numerically diverse
but behaviorally uniform. PRE-A1 now shows that replacing that chunk with an
expert chunk is also insufficient. The causal chain supported by current data
is:

`perturbation/domain shift → persistent Smol closed-loop capability error →`
`short candidate differences are washed out after handback`.

This does not prove that every longer prefix fails; it proves that an entire
native OFT chunk (8 actions) is insufficient on this cohort.

## Best next method

Treat recovery as a temporally extended option, not a one-step intervention:

1. base option: frozen SmolVLA;
2. recovery option: persistent OFT or a distilled perturbation-aware fallback;
3. termination policy: remain in recovery with hysteresis until base competence
   is re-established; default to episode-long fallback when uncertain;
4. abstention: stop when neither policy has adequate support.

The next mechanism experiment should sweep *closed-loop OFT duration*, not
resampling count: 8, 16, 32, 64 steps and episode-long OFT, followed by a
same-seed Smol handback. Replan at each OFT chunk boundary so this tests a real
recovery option rather than an open-loop action sequence. The primary quantity
is the minimum sufficient escalation duration per state.

If compute requires a smaller fallback, train a full recovery policy, not a
short replan head:

- collect OFT trajectories on balanced camera/robot perturbations;
- initialize from SmolVLA and fit a LoRA/adapter using OFT action chunks;
- use iterative student-state collection (DAgger-style relabeling) to reduce
  offline covariate shift;
- keep task/episode-disjoint validation and compare against direct OFT;
- only claim success if the student preserves recovery coverage while reducing
  latency/memory.

A candidate critic or world model remains unjustified. A world model cannot
repair a base policy that systematically chooses poor actions after every
handback; at most it could predict that failure. The next learnable component,
after duration opportunity is established, is an option-termination/competence
model—not a video predictor.

## Artifacts and reproduction

- `runs/rase_pre_a1_oft_chunks12_v1/`
- `runs/rase_pre_a1_oft_prefix_to_smol12_v1/summary.json`
- `runs/rase_pre_a1_replan_mechanism12_v1.json`
- `runs/rase_pre_a1_replan_mechanism12_v1.log`

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=0 ./scripts/run_replan_mechanism12.sh
```

Known limitations: 12 development states; no task-held-out claim; OFT prefix
arms use portions of the initial native chunk; full OFT uses more inference
compute; no learned recovery policy was evaluated.

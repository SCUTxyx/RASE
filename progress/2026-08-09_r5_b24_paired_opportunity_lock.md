# R5-B24 paired probability opportunity lock

Date: 2026-08-09

Status: **FROZEN BEFORE COLLECTION**.

## Motivation

R5-A16 had a valid repeat protocol but a conservative opportunity ceiling of
only 11.39% OFT savings and the learned controller failed 0/5 seeds.  The next
stage therefore returns to model-free opportunity rather than tuning the model,
adding a second VLA, or testing world-model features.

The QC-filtered 71-state train audit is safe-handback ready after excluding the
single deterministic OFT-prefix violation
`sp1_a314f2452b808a78f168205e0feddedf`.  It contains 52 historical finite-safe
states across all 24 tasks and has a historical conservative oracle saving of
39.42%.  This motivates a smaller paired-repeat live replication.

## Frozen cohort

- 24 train states, exactly one per true train task;
- six tasks/states per suite;
- 21 lexicographically selected historical finite-safe states;
- one historical persistent-failure state in each suite with such support:
  Goal, Long, and Object (three total);
- selection is explicitly outcome-enriched development selection and supports
  no natural-prevalence or held-out claim;
- A16 outcomes are not used for state selection and the train tasks are disjoint
  from the A16 val tasks.

## Frozen collection

- boundaries `h={0,16,32,64,96,128}`;
- five Student continuations from every reachable exact boundary snapshot;
- within a state, all boundaries reuse the exact same five continuation seeds
  (common random numbers);
- two parallel workers, all four suites;
- maximum 144 boundaries and 720 continuations; terminal persistent trajectories
  may make late boundaries unreachable;
- simulator restore remains supervision-only.

## Protocol gate

- 24/24 manifest state coverage and persistent replay parity;
- all 24 tasks/four suites represented;
- K=5 fields complete, unique seeds within a boundary;
- common seed tuple identical across every boundary of a state;
- no unexplained missing reachable boundary.

## Opportunity gate

All conditions must hold before any new model training:

- at least 20 live conservative finite-safe states;
- at least three true tasks with finite-safe states;
- at least two finite stopping bins with at least three states each;
- conservative privileged oracle OFT savings at least 25%.

The 25% threshold provides margin above the learned-controller target of 20%.
If any condition fails, do not train, do not alter thresholds post hoc, and do
not open second-VLA/world-model/validation/test stages.

## Frozen identities

- manifest SHA256:
  `bd6675a635ec2885664597ce5c60fd11a0d5a27cfa7fcab8537f86435b2236ac`;
- QC audit SHA256:
  `9e4d306960d0ecfd1c6aff5449750545893055ccac5c6658e1ab3cdcc506adb4`;
- collector SHA256:
  `170980635a317a2e35488b69add1fa6b876d0a642d1706d1dec295f7865ec983`;
- collection runner SHA256:
  `9eb910dd24ba3fb4d6799903ddd7a4b79edb993c109be355837dd1e0aa69862e`;
- summarizer SHA256:
  `a9449c2c6a662b1cdad5babbd1639811b51309e4ac344e4b7c46166e7e00ee74`;
- freezer SHA256:
  `6deebe32265ad41a355e3b91dae074ff0cdd141664f6e81284b884d8d5785619`;
- launcher SHA256:
  `40adeaddf2872f3c7e09bb3ad18d03176e5bb45d5edc3315ee0384d5a1f48af0`.

Nine probability/controller/seed-pairing tests pass.  Test remains sealed.

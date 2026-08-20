# PRE-C1.1 collection hard-stop → persistent OFT fix

Date: 2026-08-04

## Fixed-h=128 attempt (failed hard-stop)

| Metric | Value |
|--------|------:|
| Attempted failure states | 42 |
| Successful OFT recoveries | **5** |
| Kept train chunks | **45** |
| Thresholds | ≥16 states, ≥200 chunks |

Hard-stop fired: `success_states=5<16 or chunks=45<200`.

## Fix (collection only; gate thresholds unchanged)

PRE-A3 showed persistent OFT oracle gap ≈50pp vs weaker fixed handback. Re-collect failed states with:

- `teacher_horizon_mode: persistent_episode`
- `teacher_horizon_steps: 0` (until success or episode end)
- Keep the 5 already-successful trajectories (`--resume`)
- Still `keep_only_successful_oft: true`

Gate: recovery ≥8pp / retention drop ≤2pp **unchanged**.

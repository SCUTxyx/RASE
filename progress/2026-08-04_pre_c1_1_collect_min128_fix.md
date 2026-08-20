# PRE-C1.1 collection fix #2: min-128 from fork

After naive persistent OFT hard-stop (9/42 successes, 138 chunks):

Late mid-episode forks hit `_max_episode_steps` too soon (e.g. fork@231 → only 49 remaining steps). Fixed h=128 previously *extended* past episode_max; naive persistent did not.

## New rule

`teacher_horizon_mode: persistent_min128_from_fork`

```
max_steps = max(128, episode_max - fork_t)
max_episode_steps = fork_t + max_steps
```

Keep existing successful teachers; re-collect failures only. Gate thresholds unchanged (8pp / 2pp).

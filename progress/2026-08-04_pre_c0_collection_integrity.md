# PRE-C0 collection integrity

- pass: `True`
- completed: `24/24`
- outcomes: `{'failure': 19, 'success': 5}`
- states_in_pool: `737`
- exit_marker_present: `False`
- note: `PRE_C0_COLLECT_EXIT string absent; completion accepted from 24 COLLECT_EPISODE_DONE + written summary`

## Checks

- `collect_done_count_matches_24`: `True`
- `collection_exit_marker_present`: `False`
- `completion_accepted`: `True`
- `decision_context_sample_ok`: `True`
- `expected_episodes_24`: `True`
- `no_duplicate_design_tasks`: `True`
- `no_duplicate_seeds_in_sample`: `True`
- `no_unexpected_pool_episodes`: `True`
- `pool_covers_all_design_episodes`: `True`
- `summary_file_present`: `True`
- `summary_has_24_metrics`: `True`
- `summary_matches_design_episodes`: `True`

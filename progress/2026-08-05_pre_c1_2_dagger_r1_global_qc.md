# PRE-C1.2 DAgger Round 1 Global QC

- anchors covered: `9/9`
- OFT queries: `1755`
- successful teacher queries: `484`
- P(OFT success | student query): `0.2758`
- accepted rows: `1452`
- median teacher recovery length: `110.0`
- meets per-anchor Round1 minimum (all): `False`
- failed teacher JSON count (not in BC): `1271`

## Source / offset

- sources: `{"student_query_state": 484, "teacher_suffix_after_student_query": 968}`
- offsets: `{"0": 484, "1": 484, "2": 484}`

## Successful queries by trigger

- `anchor_start`: successful_queries=45 accepted_rows=135
- `periodic`: successful_queries=223 accepted_rows=669
- `progress_stall`: successful_queries=216 accepted_rows=648

## Per-anchor minimums

- `sp1_0b79453cd8e5…`: seeds=5 unique_q=62 success_relabel=62 near_chunks=186 ok=True triggers={'anchor_start': 5, 'periodic': 27, 'progress_stall': 30}
- `sp1_39cb47f5b2b2…`: seeds=5 unique_q=55 success_relabel=55 near_chunks=165 ok=True triggers={'anchor_start': 5, 'periodic': 26, 'progress_stall': 24}
- `sp1_3d04a36d3f68…`: seeds=5 unique_q=88 success_relabel=88 near_chunks=264 ok=True triggers={'anchor_start': 5, 'periodic': 41, 'progress_stall': 42}
- `sp1_91bab71a5427…`: seeds=5 unique_q=46 success_relabel=46 near_chunks=138 ok=True triggers={'anchor_start': 5, 'periodic': 20, 'progress_stall': 21}
- `sp1_a858a183b4a9…`: seeds=5 unique_q=21 success_relabel=21 near_chunks=63 ok=True triggers={'anchor_start': 5, 'periodic': 10, 'progress_stall': 6}
- `sp1_c793922aa327…`: seeds=5 unique_q=7 success_relabel=7 near_chunks=21 ok=False triggers={'anchor_start': 5, 'periodic': 1, 'progress_stall': 1}
- `sp1_ca72ec230b81…`: seeds=5 unique_q=5 success_relabel=5 near_chunks=15 ok=False triggers={'anchor_start': 5}
- `sp1_d9692ec51653…`: seeds=5 unique_q=29 success_relabel=29 near_chunks=87 ok=True triggers={'anchor_start': 5, 'periodic': 15, 'progress_stall': 9}
- `sp1_def8dd2b1b28…`: seeds=5 unique_q=171 success_relabel=171 near_chunks=513 ok=True triggers={'anchor_start': 5, 'periodic': 83, 'progress_stall': 83}

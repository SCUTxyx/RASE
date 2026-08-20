# PRE-C1.1 collection fix #3: expand T0/T2/T4 teachers

T1/T3 current_suffix failures only yielded **9/42** successful OFT recoveries (138 chunks) — below hard-stop (≥16 states, ≥200 chunks).

## Fix

Expand offline OFT teacher collection to the same PRE-C0 pool’s **T0/T2/T4** stage keys (+72 states), still:

- `keep_only_successful_oft: true`
- `persistent_min128_from_fork`
- gate thresholds **unchanged** (8pp / 2pp)

Dataset builder now indexes **all** successful OFT teachers (not only pilot48 failures); clean retention still from PRE-C0 `clean:L0` successes.

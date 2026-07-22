# Patch: LIBERO-plus NumPy 2 `np.float_` removal

- Upstream: LIBERO-plus @ `4976dc3`
- File: `libero/libero/envs/env_wrapper.py` (`plasma_fractal`)
- Change: `dtype=np.float_` → `dtype=np.float64`
- Reason: NumPy 2.0 removed `np.float_`; fog/noise corruptions crash otherwise.
- Applied: local checkout `/data/data2/yuxuan/LIBERO-plus` (2026-07-17)

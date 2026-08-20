#!/usr/bin/env python3
"""Create a cross-domain patched copy of run_libero_eval.py.

Adds:
  1. cross-domain fallback of unnorm_key (model's own suite stats)
  2. --cross_norm_stats_path: inject the target suite's dataset statistics so
     cross-domain unnormalization is exactly correct (controlled experiment to
     separate model capability from unnorm mismatch).
"""
import ast
from pathlib import Path

SRC = Path("/root/autodl-tmp/src/openvla-oft/experiments/robot/libero/run_libero_eval.py")
OUT = Path("/root/autodl-tmp/RASE/scripts/run_libero_eval_cross.py")

# 1) add the dataclass field after load_in_4bit
old_field = """    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization"""
new_field = """    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    cross_norm_stats_path: str = ""                  # (RASE) JSON of the target suite's dataset_statistics to inject for cross-domain unnorm"""

# 2) patch check_unnorm_key
old_check = """    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!\""""

new_check = """    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    # RASE cross-domain patch: when the task suite is absent from this model's
    # norm_stats (e.g. oft_spatial rolled out on libero_object):
    #   - if --cross_norm_stats_path is given, inject the target suite's
    #     statistics so unnormalization is exactly correct (controlled test);
    #   - otherwise fall back to the model's own suite statistics.
    if unnorm_key not in model.norm_stats:
        if cfg.cross_norm_stats_path:
            import json as _json
            stats = _json.load(open(cfg.cross_norm_stats_path))
            for k, v in stats.items():
                model.norm_stats[k] = v
            unnorm_key = list(stats.keys())[0]
            print(f"[rase-cross] injected norm_stats from {cfg.cross_norm_stats_path}; "
                  f"unnorm_key={unnorm_key}")
        else:
            fallback = list(model.norm_stats)
            assert fallback, f"no usable norm_stats keys: {list(model.norm_stats)}"
            unnorm_key = fallback[0]
            print(f"[rase-cross] suite {cfg.task_suite_name} not in norm_stats; "
                  f"falling back to unnorm_key={unnorm_key}")

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!\""""

src = SRC.read_text()
for label, old, new in [
    ("dataclass field", old_field, new_field),
    ("check_unnorm_key", old_check, new_check),
]:
    assert old in src, f"{label} block not found in source"
    src = src.replace(old, new, 1)
OUT.write_text(src)
ast.parse(src)
print(f"patched script written: {OUT}")

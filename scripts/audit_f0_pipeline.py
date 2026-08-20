#!/usr/bin/env python3
"""PRE-C0-R0 Step 1.2: Pipeline implementation audit.

Rule out normalization/frame/index bugs in the F0 correction pipeline.
Computes E[delta_target] (teacher-student residual), per-dim stats,
per-task means, per-suite means, temporal decomposition, and
compares F0's constant vector c with the empirical residual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_all_r0_episodes(data_dir):
    episodes = []
    r0_dir = data_dir / "R0"
    if not r0_dir.is_dir():
        return episodes
    for ep_path in sorted(r0_dir.glob("*.json")):
        episodes.append(json.loads(ep_path.read_text(encoding="utf-8")))
    return episodes


def extract_delta_targets(episodes):
    records = []
    for ep in episodes:
        for step in ep.get("teacher_recovery", []):
            dt = step.get("delta_target", None)
            if dt is None:
                continue
            dt_arr = np.asarray(dt, dtype=np.float32).flatten()[:7]
            if dt_arr.size != 7:
                continue

            a_s = np.asarray(step.get("action", np.zeros(7)), dtype=np.float32).flatten()[:7]
            if len(a_s) != 7 and a_s.size >= 7:
                a_s = a_s[:7]
            a_t = np.asarray(step.get("teacher_action", np.zeros(7)), dtype=np.float32).flatten()[:7]
            if len(a_t) != 7 and a_t.size >= 7:
                a_t = a_t[:7]

            dt_direct = a_t - a_s

            records.append(dict(
                delta_target=dt_arr,
                delta_direct=dt_direct,
                student_action=a_s,
                teacher_action=a_t,
                step_index=step.get("t", step.get("step_index", 0)),
                episode_id=ep.get("task_id", ""),
                suite=ep.get("suite", "unknown"),
                student_success=ep.get("student_success", None),
                teacher_success=ep.get("teacher_success", None),
                boundary_step=ep.get("boundary_step", 0),
            ))
    return records


def per_dim_stats(deltas):
    per_dim = {}
    for d in range(7):
        vals = deltas[:, d]
        per_dim["dim_{}_{}".format(d, DIM_NAMES[d])] = dict(
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            std=float(np.std(vals)),
            min_val=float(np.min(vals)),
            max_val=float(np.max(vals)),
            sign_ratio_positive=float(np.mean(vals > 0)),
            sign_ratio_negative=float(np.mean(vals < 0)),
            sign_ratio_zero=float(np.mean(np.abs(vals) < 1e-8)),
            pct_in_clip=float(np.mean(np.abs(vals) <= 0.5)),
            pct_near_boundary=float(np.mean(np.abs(vals) > 0.45)),
        )
    return per_dim


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir-spatial", type=Path,
                        default=ROOT / "runs/route_c_r0_scaled")
    parser.add_argument("--data-dir-object", type=Path,
                        default=ROOT / "runs/route_c_r0_object")
    parser.add_argument("--data-dir-goal", type=Path,
                        default=ROOT / "runs/route_c_r0_goal")
    parser.add_argument("--data-dir-10", type=Path,
                        default=ROOT / "runs/route_c_r0_10")
    parser.add_argument("--f0-vector", type=Path,
                        default=ROOT / "runs/pre_c0_r0/f0_constant_vector.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/pre_c0_r0")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    f0_c = None
    if args.f0_vector and args.f0_vector.is_file():
        f0_data = json.loads(args.f0_vector.read_text(encoding="utf-8"))
        f0_c = np.array(f0_data["f0_constant_vector_c"], dtype=np.float32)
        print("Loaded F0 vector c: [{}]".format(
            ", ".join("{:+.6f}".format(x) for x in f0_c)))

    suite_dirs = dict(
        libero_spatial=args.data_dir_spatial,
        libero_object=args.data_dir_object,
        libero_goal=args.data_dir_goal,
        libero_10=args.data_dir_10,
    )

    all_records = []
    suite_records = {}

    for suite_name, data_dir in suite_dirs.items():
        if not data_dir.is_dir():
            print("WARNING: {} data dir not found: {}".format(suite_name, data_dir))
            continue
        episodes = load_all_r0_episodes(data_dir)
        records = extract_delta_targets(episodes)
        if records:
            suite_records[suite_name] = records
            all_records.extend(records)
            print("  {}: {} episodes, {} records".format(
                suite_name, len(episodes), len(records)))
        else:
            print("  {}: {} episodes, NO valid records".format(
                suite_name, len(episodes)))

    if not all_records:
        print("FATAL: No valid records found across all suites.")
        return 1

    print("\nTotal records: {}".format(len(all_records)))

    all_deltas = np.stack([r["delta_target"] for r in all_records], axis=0)
    all_direct = np.stack([r["delta_direct"] for r in all_records], axis=0)

    delta_diff = np.abs(all_deltas - all_direct).mean(axis=0)
    delta_consistency = float(np.abs(all_deltas - all_direct).mean())

    E_delta = all_deltas.mean(axis=0)
    E_direct = all_direct.mean(axis=0)

    print("E[delta_target] = [{}]".format(
        ", ".join("{:+.6f}".format(x) for x in E_delta)))
    print("E[delta_direct] (aT-aS) = [{}]".format(
        ", ".join("{:+.6f}".format(x) for x in E_direct)))
    print("|delta_target - (aT-aS)| mean = {:.8f}".format(delta_consistency))
    print("Per-dim difference: [{}]".format(
        ", ".join("{:.2e}".format(x) for x in delta_diff)))

    global_stats = dict(
        E_delta_target=[float(x) for x in E_delta],
        E_delta_direct=[float(x) for x in E_direct],
        norm_E_delta=float(np.linalg.norm(E_delta)),
        delta_consistency_err=delta_consistency,
        per_dim_diff_target_vs_direct=[float(x) for x in delta_diff],
        n_total_records=len(all_records),
        global_per_dim=per_dim_stats(all_deltas),
    )

    if f0_c is not None:
        cos_target = cosine(f0_c, E_delta)
        cos_direct = cosine(f0_c, E_direct)
        global_stats["cos_f0_vs_E_delta_target"] = cos_target
        global_stats["cos_f0_vs_E_delta_direct"] = cos_direct
        print("\ncos(c_F0, E[delta_target]) = {:.6f}".format(cos_target))
        print("cos(c_F0, E[delta_direct]) = {:.6f}".format(cos_direct))
        if cos_target > 0.95:
            print(">>> F0 is essentially learning the MEAN residual (systematic bias)")
        elif cos_target > 0.8:
            print(">>> F0 is strongly aligned with mean residual, but not exactly")
        else:
            print(">>> F0 vector is NOT simply the mean residual")

    suite_analysis = {}
    suite_E_deltas = {}
    for suite_name, records in suite_records.items():
        if len(records) < 5:
            suite_analysis[suite_name] = dict(n_records=len(records), note="too few records")
            continue
        deltas = np.stack([r["delta_target"] for r in records], axis=0)
        E_suite = deltas.mean(axis=0)
        suite_E_deltas[suite_name] = E_suite
        entry = dict(n_records=len(records),
                     E_delta=[float(x) for x in E_suite],
                     norm_E_delta=float(np.linalg.norm(E_suite)),
                     per_dim=per_dim_stats(deltas))
        if f0_c is not None:
            entry["cos_f0_vs_E_suite"] = cosine(f0_c, E_suite)
        suite_analysis[suite_name] = entry
        print("\n--- {} (N={}) ---".format(suite_name, len(records)))
        print("  E[delta] = [{}]".format(
            ", ".join("{:+.6f}".format(x) for x in E_suite)))
        print("  |E[delta]| = {:.6f}".format(entry["norm_E_delta"]))

    suite_cosines = {}
    suite_names = list(suite_E_deltas.keys())
    for i, sa in enumerate(suite_names):
        for j, sb in enumerate(suite_names):
            if i < j:
                key = "{}_vs_{}".format(sa, sb)
                cos_val = cosine(suite_E_deltas[sa], suite_E_deltas[sb])
                suite_cosines[key] = cos_val
                print("  cos(E_{}, E_{}) = {:.4f}".format(sa, sb, cos_val))

    success_deltas = []
    fail_deltas = []
    for r in all_records:
        if r["student_success"] is True:
            success_deltas.append(r["delta_target"])
        elif r["student_success"] is False:
            fail_deltas.append(r["delta_target"])

    outcome_analysis = {}
    E_success = None
    E_fail = None
    if success_deltas:
        s_arr = np.stack(success_deltas, axis=0)
        E_success = s_arr.mean(axis=0)
        outcome_analysis["student_success"] = dict(
            n=len(success_deltas),
            E_delta=[float(x) for x in E_success],
            norm=float(np.linalg.norm(E_success)),
            per_dim=per_dim_stats(s_arr))
        print("\nE[delta | student_success] (N={}) = [{}]".format(
            len(success_deltas),
            ", ".join("{:+.6f}".format(x) for x in E_success)))
    if fail_deltas:
        f_arr = np.stack(fail_deltas, axis=0)
        E_fail = f_arr.mean(axis=0)
        outcome_analysis["student_fail"] = dict(
            n=len(fail_deltas),
            E_delta=[float(x) for x in E_fail],
            norm=float(np.linalg.norm(E_fail)),
            per_dim=per_dim_stats(f_arr))
        print("E[delta | student_fail] (N={}) = [{}]".format(
            len(fail_deltas),
            ", ".join("{:+.6f}".format(x) for x in E_fail)))
    if E_success is not None and E_fail is not None:
        cos_sf = cosine(E_success, E_fail)
        outcome_analysis["cos_success_vs_fail"] = cos_sf
        print("cos(E_success, E_fail) = {:.4f}".format(cos_sf))

    early, mid, late = [], [], []
    for r in all_records:
        t = r["step_index"]
        rel_t = t - r["boundary_step"] if r["boundary_step"] > 0 else t
        if rel_t < 20:
            early.append(r["delta_target"])
        elif rel_t < 100:
            mid.append(r["delta_target"])
        else:
            late.append(r["delta_target"])

    temporal = {}
    for label, arr_list in [("early_t<20", early), ("mid_20-100", mid), ("late_>100", late)]:
        if arr_list:
            arr = np.stack(arr_list, axis=0)
            E_t = arr.mean(axis=0)
            temporal[label] = dict(n=len(arr_list),
                                    E_delta=[float(x) for x in E_t],
                                    norm=float(np.linalg.norm(E_t)))
            print("E[delta | {}] (N={}) = [{}]".format(
                label, len(arr_list),
                ", ".join("{:+.6f}".format(x) for x in E_t)))

    spatial_records = suite_records.get("libero_spatial", [])
    task_groups = {}
    for r in spatial_records:
        tid = r["episode_id"]
        task_groups.setdefault(tid, []).append(r["delta_target"])

    task_analysis = {}
    for tid, deltas_list in task_groups.items():
        arr = np.stack(deltas_list, axis=0)
        E_task = arr.mean(axis=0)
        task_analysis[tid] = dict(n=len(deltas_list),
                                   E_delta=[float(x) for x in E_task],
                                   norm=float(np.linalg.norm(E_task)))
        if len(task_groups) <= 12:
            print("Task {}: N={} E=[{}]".format(
                tid, len(deltas_list),
                ", ".join("{:+.4f}".format(x) for x in E_task)))

    task_names = list(task_analysis.keys())
    task_cosines = {}
    for i, ta in enumerate(task_names):
        for j, tb in enumerate(task_names):
            if i < j:
                a_vec = np.array(task_analysis[ta]["E_delta"])
                b_vec = np.array(task_analysis[tb]["E_delta"])
                task_cosines["{}_vs_{}".format(ta, tb)] = cosine(a_vec, b_vec)

    all_aS = np.stack([r["student_action"] for r in all_records], axis=0)
    all_aT = np.stack([r["teacher_action"] for r in all_records], axis=0)

    saturation = dict(
        student_pct_near_plus1=float(np.mean(all_aS > 0.99)),
        student_pct_near_minus1=float(np.mean(all_aS < -0.99)),
        teacher_pct_near_plus1=float(np.mean(all_aT > 0.99)),
        teacher_pct_near_minus1=float(np.mean(all_aT < -0.99)),
        student_per_dim_near_boundary={},
        teacher_per_dim_near_boundary={},
    )
    for d in range(7):
        saturation["student_per_dim_near_boundary"][
            "dim_{}_{}".format(d, DIM_NAMES[d])] = \
            float(np.mean(np.abs(all_aS[:, d]) > 0.98))
        saturation["teacher_per_dim_near_boundary"][
            "dim_{}_{}".format(d, DIM_NAMES[d])] = \
            float(np.mean(np.abs(all_aT[:, d]) > 0.98))

    print("\nAction saturation: student near +1={:.4f} near -1={:.4f}".format(
        saturation["student_pct_near_plus1"], saturation["student_pct_near_minus1"]))
    print("                   teacher near +1={:.4f} near -1={:.4f}".format(
        saturation["teacher_pct_near_plus1"], saturation["teacher_pct_near_minus1"]))

    bugs = []

    if delta_consistency > 1e-3:
        bugs.append("delta_target != teacher-student diff (mean_err={:.2e})".format(
            delta_consistency))

    if global_stats["global_per_dim"]["dim_6_gripper"]["mean"] != 0:
        bugs.append("gripper dimension has non-zero correction: mean={:.6f}".format(
            global_stats["global_per_dim"]["dim_6_gripper"]["mean"]))

    saturated_dims = []
    for d in range(7):
        pct = saturation["student_per_dim_near_boundary"][
            "dim_{}_{}".format(d, DIM_NAMES[d])]
        if pct > 0.5:
            saturated_dims.append("{}={:.2f}".format(DIM_NAMES[d], pct))
    if saturated_dims:
        bugs.append("student_action dimension saturation: {}".format(
            ", ".join(saturated_dims)))

    if "libero_object" in suite_E_deltas and "libero_spatial" in suite_E_deltas:
        cos_so = cosine(suite_E_deltas["libero_spatial"],
                        suite_E_deltas["libero_object"])
        if cos_so < 0:
            bugs.append("spatial vs object correction OPPOSITE (cos={:.4f})".format(cos_so))

    for d in range(7):
        info = global_stats["global_per_dim"]["dim_{}_{}".format(d, DIM_NAMES[d])]
        ratio = max(info["sign_ratio_positive"], info["sign_ratio_negative"])
        if ratio > 0.9:
            direction = "+" if info["sign_ratio_positive"] > 0.9 else "-"
            bugs.append("{} has {:.0%} {} bias -- systematic offset".format(
                DIM_NAMES[d], ratio, direction))

    if bugs:
        print("\n!!! BUG FINDINGS ({}) !!!".format(len(bugs)))
        for b in bugs:
            print("  - {}".format(b))
    else:
        print("\nNo pipeline bugs detected.")

    report = dict(
        global_stats=global_stats,
        suite_analysis=suite_analysis,
        suite_cross_cosines=suite_cosines,
        outcome_analysis=outcome_analysis,
        temporal_stage_analysis=temporal,
        task_analysis=task_analysis,
        task_cross_cosines=task_cosines,
        action_saturation=saturation,
        n_suites_loaded=len(suite_records),
        n_total_records=len(all_records),
        bug_findings=bugs,
    )

    out_path = output_dir / "f0_pipeline_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("\nSaved to: {}".format(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

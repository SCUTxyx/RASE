#!/usr/bin/env python3
"""P0 audit: same-root provenance + offline/deploy feature equivalence.

Checks (per RASE CRR execution plan §2):
  1. root consistency     : s_t_proprio identical within root; chunk 8x7;
                            future_steps == 64.
  2. candidate freeze     : stored chunk stats == recomputed from chunk_raw.
  3. feature equivalence  : offline path (train_same_root_risk inline layout:
                            s_t_proprio + canonical + bigram) == deploy path
                            (rase_selector_loop feats_of layout: observation
                            state + canonical + bigram); ||x_off - x_dep||inf
                            < 1e-6; risk scores from saved npz consistent.
  4. vocab consistency    : rebuilt vocab == saved vocab (keys + order).
  5. normalizer           : npz mean/scale/weights/intercept == refit on full
                            data (median-binarized consequence_label).
  6. label semantics      : consequence_label == displacement ==
                            ||s_th[:3] - s_t[:3]||; median split boundary.
  7. distribution gap     : same-root s_t vs dp_collect (branch-trajectory)
                            proprio per-dim Wasserstein (quantifies the
                            closed-loop-v2 train/deploy gap).
  8. B1 replication       : full-data ridge AUROC ~0.98 on this data.

Usage (server, oft env):
  python audit_same_root_provenance.py \
    --data runs/oft_opportunity/same_root_w1.jsonl \
    --risk runs/oft_opportunity/same_root_risk.npz \
    --vocab runs/oft_opportunity/same_root_risk.vocab.json \
    --dp-dir runs/oft_opportunity \
    --output runs/oft_opportunity/p0_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase_common import canonical_chunk_features, build_bigram_vocab, \
    bigram_features, fit_ridge, ridge_predict, auroc, wasserstein_1d


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def deploy_feats(proprio: np.ndarray, chunk_raw, task: str, vocab: dict) -> np.ndarray:
    """Faithful replication of rase_selector_loop.feats_of (deployment)."""
    arr = np.asarray(chunk_raw, dtype=np.float64).reshape(-1, 7)
    parts = [np.asarray(proprio, dtype=np.float64)]
    parts.append(canonical_chunk_features(arr))
    t = task.lower()
    x = np.zeros(len(vocab))
    for i in range(len(t) - 1):
        idx = vocab.get(t[i:i + 2])
        if idx is not None:
            x[idx] += 1.0
    parts.append(x)
    return np.concatenate(parts)


def offline_feats(proprio: np.ndarray, chunk_raw, task: str, vocab: dict) -> np.ndarray:
    """Faithful replication of train_same_root_risk.main feature layout."""
    arr = np.asarray(chunk_raw, dtype=np.float64).reshape(-1, 7)
    X_state = np.asarray(proprio, dtype=np.float64)
    X_chunk = canonical_chunk_features(arr)
    X_bigram = bigram_features(task, vocab)
    return np.concatenate([X_state, X_chunk, X_bigram])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--risk", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--dp-dir", type=Path, default=None,
                    help="dir with dp_collect_{spatial,object}.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.data)
    report: dict = {"n_rows": len(rows), "schema": "rase-p0-audit/v1"}

    # ---- 1. root consistency ----
    roots: dict[tuple, list[dict]] = {}
    for r in rows:
        roots.setdefault((r["task"], r["episode_idx"], r["decision_idx"]), []).append(r)
    max_sdiff, bad_roots, bad_steps = 0.0, 0, 0
    chunk_lens: dict[int, int] = {}
    for rk, rs in roots.items():
        sp = [np.asarray(r["s_t_proprio"], dtype=np.float64) for r in rs]
        d = max(float(np.abs(sp[0] - s).max()) for s in sp[1:])
        max_sdiff = max(max_sdiff, d)
        if d > 1e-9:
            bad_roots += 1
        arr = np.asarray(rs[0]["chunk_raw"]).reshape(-1, 7)
        chunk_lens[len(arr)] = chunk_lens.get(len(arr), 0) + 1
        if any(r["future_steps"] != 64 for r in rs):
            bad_steps += 1
    report["root_consistency"] = {
        "roots": len(roots),
        "max_s_t_diff_within_root": max_sdiff,
        "roots_with_diff": bad_roots,
        "chunk_len_counts": chunk_lens,
        "roots_with_non64_future": bad_steps,
        "pass": max_sdiff == 0.0 and bad_roots == 0 and bad_steps == 0
                and list(chunk_lens) == [8],
    }

    # ---- 2. candidate freeze: stored stats == recomputed ----
    from crr_common import canonical_of
    stats_keys = ["chunk_mean_pos", "chunk_mean_rot", "chunk_std_pos",
                  "chunk_std_rot", "chunk_gripper_mean", "chunk_gripper_std",
                  "chunk_total_disp", "chunk_norm_mean"]
    stat_bad = 0
    for r in rows:
        arr = np.asarray(r["chunk_raw"], dtype=np.float64).reshape(-1, 7)
        pos = arr[:, :3]
        recomputed = {
            "chunk_mean_pos": arr[:, :3].mean(0),
            "chunk_mean_rot": arr[:, 3:6].mean(0),
            "chunk_std_pos": arr[:, :3].std(0),
            "chunk_std_rot": arr[:, 3:6].std(0),
            "chunk_gripper_mean": arr[:, 6].mean(),
            "chunk_gripper_std": arr[:, 6].std(),
            "chunk_total_disp": float(np.abs(np.diff(pos, axis=0)).sum()),
            "chunk_norm_mean": float(np.linalg.norm(pos, axis=1).mean()),
        }
        for k in stats_keys:
            got = np.asarray(r[k], dtype=np.float64)
            exp = np.asarray(recomputed[k], dtype=np.float64)
            if got.shape != exp.shape or float(np.abs(got - exp).max()) > 1e-8:
                stat_bad += 1
                break
    report["candidate_freeze"] = {
        "rows_checked": len(rows), "rows_with_stats_mismatch": stat_bad,
        "pass": stat_bad == 0,
    }

    # ---- 3. feature equivalence offline vs deploy ----
    vocab_rebuilt = build_bigram_vocab([r["task"] for r in rows])
    vocab_saved = json.loads(args.vocab.read_text())
    max_feat_diff = 0.0
    for r in rows:
        xo = offline_feats(r["s_t_proprio"], r["chunk_raw"], r["task"], vocab_saved)
        xd = deploy_feats(r["s_t_proprio"], r["chunk_raw"], r["task"], vocab_saved)
        max_feat_diff = max(max_feat_diff, float(np.abs(xo - xd).max()))
    report["feature_equivalence"] = {
        "feat_dim": int(offline_feats(
            rows[0]["s_t_proprio"], rows[0]["chunk_raw"],
            rows[0]["task"], vocab_saved).shape[0]),
        "max_abs_diff_offline_vs_deploy": max_feat_diff,
        "pass": max_feat_diff < 1e-6,
    }

    # ---- 4. vocab consistency ----
    vocab_match = (list(vocab_rebuilt) == list(vocab_saved))
    report["vocab"] = {
        "rebuilt_size": len(vocab_rebuilt), "saved_size": len(vocab_saved),
        "identical_keys_and_order": vocab_match, "pass": vocab_match,
    }

    # ---- 5. normalizer: refit == saved npz ----
    X = np.stack([offline_feats(r["s_t_proprio"], r["chunk_raw"], r["task"],
                                vocab_saved) for r in rows])
    labels = np.array([float(r["consequence_label"]) for r in rows])
    yb = (labels > np.median(labels)).astype(float)
    mean, scale, beta, y_mean = fit_ridge(X, yb, alpha=1.0)
    with np.load(args.risk, allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    norm_pass = (float(np.abs(mean - npz["mean"]).max()) < 1e-6
                 and float(np.abs(scale - npz["scale"]).max()) < 1e-6
                 and float(np.abs(beta[1:] - npz["weights"]).max()) < 1e-6
                 and abs(float(y_mean + beta[0] - npz["intercept"])) < 1e-6)
    report["normalizer"] = {
        "max_abs_diff_mean": float(np.abs(mean - npz["mean"]).max()),
        "max_abs_diff_scale": float(np.abs(scale - npz["scale"]).max()),
        "max_abs_diff_weights": float(np.abs(beta[1:] - npz["weights"]).max()),
        "abs_diff_intercept": float(abs(y_mean + beta[0] - npz["intercept"])),
        "pass": norm_pass,
    }

    # ---- 6. label semantics ----
    lab_bad = 0
    for r in rows:
        st = np.asarray(r["s_t_proprio"], dtype=np.float64)
        sh = np.asarray(r["s_th_proprio"], dtype=np.float64)
        d = float(np.linalg.norm(sh[:3] - st[:3]))
        if abs(d - r["consequence_label"]) > 1e-6 \
                or abs(d - r["displacement"]) > 1e-6:
            lab_bad += 1
    med = float(np.median(labels))
    report["label_semantics"] = {
        "rows_with_label_mismatch": lab_bad,
        "consequence_label_min": float(labels.min()),
        "consequence_label_max": float(labels.max()),
        "median_split_boundary": med,
        "pos_rate": float(yb.mean()),
        "note": "consequence_label = ||eef_pos(s_{t+H}) - eef_pos(s_t)|| (64 steps)",
        "pass": lab_bad == 0,
    }

    # ---- 7. distribution gap: same-root s_t vs dp_collect states ----
    gap: dict[str, dict] = {}
    if args.dp_dir is not None:
        for suite in ("spatial", "object"):
            dp_path = args.dp_dir / f"dp_collect_{suite}.jsonl"
            if not dp_path.is_file():
                continue
            dp_rows = load_rows(dp_path)
            dp = np.stack([np.asarray(r["proprio"], dtype=np.float64)
                           for r in dp_rows])
            sr = np.stack([np.asarray(r["s_t_proprio"], dtype=np.float64)
                           for r in rows if r["suite"] == f"libero_{suite}"])
            w = np.mean([wasserstein_1d(dp[:, k], sr[:, k]) for k in range(8)])
            gap[suite] = {
                "dp_rows": len(dp), "same_root_rows": len(sr),
                "mean_per_dim_wasserstein": float(w),
                "per_dim_mean_abs_diff": [
                    float(np.abs(dp[:, k].mean() - sr[:, k].mean())) for k in range(8)],
                "per_dim_std_abs_diff": [
                    float(np.abs(dp[:, k].std() - sr[:, k].std())) for k in range(8)],
            }
    report["distribution_gap_vs_dp_collect"] = gap

    # ---- 8. B1 replication ----
    p_full = ridge_predict(X, mean, scale, beta, y_mean)
    rep = {"full_data_auroc": auroc(p_full, yb)}
    vlas = sorted({r["model"] for r in rows})
    rep["lovo"] = {}
    for held in vlas:
        tr = [i for i, r in enumerate(rows) if r["model"] != held]
        te = [i for i, r in enumerate(rows) if r["model"] == held]
        m_, s_, b_, ym_ = fit_ridge(X[tr], yb[tr], alpha=1.0)
        p_te = ridge_predict(X[te], m_, s_, b_, ym_)
        rep["lovo"][held] = round(auroc(p_te, yb[te]), 4)
    report["b1_replication"] = rep

    passed = all(
        report[k]["pass"] for k in ("root_consistency", "candidate_freeze",
                                    "feature_equivalence", "vocab",
                                    "normalizer", "label_semantics"))
    report["gate_p0"] = {"verdict": "PASS" if passed else "FAIL",
                         "checks": ["root_consistency", "candidate_freeze",
                                    "feature_equivalence", "vocab",
                                    "normalizer", "label_semantics"]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_p0": report["gate_p0"], "n_rows": len(rows),
                      "feat_dim": report["feature_equivalence"]["feat_dim"],
                      "max_s_t_diff": report["root_consistency"]["max_s_t_diff_within_root"],
                      "max_feat_diff": report["feature_equivalence"]["max_abs_diff_offline_vs_deploy"],
                      "b1_full_auroc": rep["full_data_auroc"],
                      "lovo": rep["lovo"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

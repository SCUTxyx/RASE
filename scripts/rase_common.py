#!/usr/bin/env python3
"""RASE common module: canonical features, metrics, risk-model loading,
identity probes.  Pure numpy — runs locally and on the server.

Feature variants (Stage C ablation):
  C0: state + raw chunk stats            (no language, no prior)
  C1: state + canonical chunk features   (no language, no prior)
  C2: state + canonical + bigram         (no prior)
  C3: state + canonical + bigram + visual placeholder
  C4: C3 + candidate prior stats
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# canonical action features (platform-agnostic intent, computed from a chunk)
# ---------------------------------------------------------------------------

CHUNK_KEYS_RAW = [
    "chunk_mean_pos", "chunk_mean_rot", "chunk_std_pos", "chunk_std_rot",
    "chunk_gripper_mean", "chunk_gripper_std", "chunk_total_disp",
    "chunk_norm_mean",
]


def canonical_chunk_features(arr: np.ndarray) -> np.ndarray:
    """arr: (H, D) native action chunk (D=7 for LIBERO: dpos3+drot3+gripper).

    Returns translation/rotation/velocity/acceleration/smoothness/direction
    consistency — independent of the platform's raw action convention where
    possible (gripper is assumed last dim).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D chunk, got {arr.shape}")
    H, D = arr.shape
    pos = arr[:, :3]
    rot = arr[:, 3:6] if D >= 6 else np.zeros((H, 0))
    gripper = arr[:, 6] if D >= 7 else np.zeros(H)
    dv = np.diff(pos, axis=0)          # (H-1, 3)
    da = np.diff(dv, axis=0) if len(dv) > 1 else np.zeros((0, 3))
    feats: list[float] = []
    # translation
    feats += list(pos.mean(axis=0))
    feats += list(pos.std(axis=0))
    feats.append(float(np.linalg.norm(dv, axis=1).mean()))       # mean velocity
    feats.append(float(np.linalg.norm(dv, axis=1).max()))        # max velocity
    feats.append(float(np.linalg.norm(pos[-1] - pos[0])))        # net displacement
    feats.append(float(np.abs(pos[-1] - pos[0]).sum()))          # manhattan net
    if len(da):
        feats.append(float(np.linalg.norm(da, axis=1).mean()))   # mean accel
        feats.append(float(np.linalg.norm(da, axis=1).max()))
    else:
        feats += [0.0, 0.0]
    # direction consistency: mean cosine between consecutive displacements
    if len(dv) > 1:
        norms = np.linalg.norm(dv, axis=1)
        cos = np.sum(dv[:-1] * dv[1:], axis=1) / (
            norms[:-1] * norms[1:] + 1e-9)
        feats.append(float(np.mean(cos)))
        feats.append(float(np.std(cos)))
    else:
        feats += [0.0, 0.0]
    # rotation
    if rot.shape[1]:
        feats += list(rot.mean(axis=0))
        feats += list(rot.std(axis=0))
    # gripper intent
    feats.append(float(np.mean(gripper)))
    feats.append(float(np.std(gripper)))
    feats.append(float(np.mean(gripper > 0)))
    # smoothness of speed profile
    speed = np.linalg.norm(dv, axis=1)
    feats.append(float(speed.std()) if len(speed) else 0.0)
    return np.asarray(feats, dtype=np.float64)


def raw_chunk_stats_feats(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    pos = arr[:, :3]
    out = np.concatenate([
        arr[:, :3].mean(axis=0), arr[:, 3:6].mean(axis=0),
        arr[:, :3].std(axis=0), arr[:, 3:6].std(axis=0),
        [float(arr[:, 6].mean()), float(arr[:, 6].std()),
         float(np.abs(np.diff(pos, axis=0)).sum()),
         float(np.linalg.norm(pos, axis=1).mean())],
    ])
    return out


# ---------------------------------------------------------------------------
# language features
# ---------------------------------------------------------------------------


def build_bigram_vocab(texts: Iterable[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in texts:
        text = text.lower()
        for i in range(len(text) - 1):
            vocab.setdefault(text[i:i + 2], len(vocab))
    return vocab


def bigram_features(text: str, vocab: dict[str, int]) -> np.ndarray:
    text = text.lower()
    x = np.zeros(len(vocab), dtype=np.float64)
    for i in range(len(text) - 1):
        idx = vocab.get(text[i:i + 2])
        if idx is not None:
            x[idx] += 1.0
    return x


# ---------------------------------------------------------------------------
# candidate prior (dataset statistics) — identity leakage risk! ablation-only
# ---------------------------------------------------------------------------

_prior_cache: dict[str, np.ndarray] = {}


def candidate_prior(model_name: str, ckpts_root: str = "/root/autodl-tmp/RASE/ckpts") -> np.ndarray:
    """Candidate identity via its own action prior statistics.  For LeRobot
    policies (no OpenVLA dataset_statistics.json) we try the LeRobot
    policy_preprocessor normalizer; if unavailable, return zeros (honest
    missing-prior signal) — the frozen v3 model then scores with prior=0."""
    if model_name not in _prior_cache:
        path = Path(f"{ckpts_root}/{model_name}/dataset_statistics.json")
        if path.is_file():
            stats = json.load(open(path))
            key = list(stats.keys())[0]
            act = stats[key]["action"]
            mean = np.asarray(act["mean"], dtype=np.float64)[:7]
            span = np.asarray(act["q99"], dtype=np.float64)[:7] - np.asarray(
                act["q01"], dtype=np.float64)[:7]
            _prior_cache[model_name] = np.concatenate([mean, span])
        else:
            # LeRobot-style normalizer fallback (policy_preprocessor.json)
            lr = Path(f"{ckpts_root}/{model_name}/policy_preprocessor.json")
            prior = None
            if lr.is_file():
                try:
                    pp = json.load(open(lr))
                    for step in pp.get("steps", []):
                        cfg = step.get("config", {})
                        feats = cfg.get("features", {})
                        act_key = [k for k in feats if "action" in k]
                        if act_key and "mean" in cfg:
                            mean = np.asarray(cfg["mean"], dtype=np.float64)
                            std = np.asarray(cfg["std"], dtype=np.float64)
                            if mean.size >= 7:
                                prior = np.concatenate(
                                    [mean[:7], 2.0 * std[:7]])
                                break
                except Exception:
                    prior = None
            if prior is None:
                prior = np.zeros(14, dtype=np.float64)
            _prior_cache[model_name] = prior
    return _prior_cache[model_name]


# ---------------------------------------------------------------------------
# feature assembly for a collection row
# ---------------------------------------------------------------------------


def row_chunk_array(row: dict) -> np.ndarray:
    """Recover the 8x7 chunk from a collection row (raw or stats-only)."""
    if "chunk_raw" in row and row["chunk_raw"]:
        arr = np.asarray(row["chunk_raw"], dtype=np.float64)
        return arr.reshape(-1, 7) if arr.ndim == 1 else arr
    # fall back: reconstruct from stats (approximation; prefer chunk_raw!)
    raise KeyError("row has no chunk_raw; re-collect with chunk_raw enabled")


def build_row_features(
    row: dict,
    vocab: dict[str, int],
    *,
    action: str = "canonical",      # "raw" | "canonical"
    use_language: bool = False,
    use_prior: bool = False,
    ckpts_root: str = "/root/autodl-tmp/RASE/ckpts",
) -> np.ndarray:
    parts = [np.asarray(row["proprio"], dtype=np.float64)]
    arr = row_chunk_array(row)
    if action == "canonical":
        parts.append(canonical_chunk_features(arr))
    else:
        parts.append(raw_chunk_stats_feats(arr))
    if use_language:
        parts.append(bigram_features(row["task"], vocab))
    if use_prior:
        parts.append(candidate_prior(row["model"], ckpts_root))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# risk model (frozen v3 artifact)
# ---------------------------------------------------------------------------


class RiskModel:
    """Standardized ridge (logistic link) with mean/scale/weights/intercept."""

    def __init__(self, npz_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as z:
            self.mean = z["mean"]
            self.scale = z["scale"]
            self.weights = z["weights"]
            self.intercept = float(z["intercept"])
            self.alpha = float(z["alpha"])
            self.model_type = str(z["model_type"])
            self.feature_version = str(z["feature_version"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        logit = self.intercept + Xs @ self.weights
        return 1.0 / (1.0 + np.exp(-logit))

    def score_row(self, row: dict, vocab: dict[str, int], **kw) -> float:
        x = build_row_features(row, vocab, **kw)
        return float(self.predict(x[None, :])[0])


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = (X - mean) / scale
    design = np.column_stack((np.ones(len(Xs)), Xs))
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    y_mean = float(y.mean())
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ (y - y_mean))
    return mean, scale, beta, y_mean


def ridge_predict(X: np.ndarray, mean, scale, beta, y_mean) -> np.ndarray:
    Xs = (np.asarray(X, dtype=np.float64) - mean) / scale
    logit = y_mean + beta[0] + Xs @ beta[1:]
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# metrics (no sklearn dependency)
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")  # ascending rank (1 = lowest)
    y_s = y[order]
    n_pos = int(y_s.sum())
    n_neg = len(y_s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_s) + 1)
    pos_ranks = ranks[y_s == 1]
    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(scores: np.ndarray, y: np.ndarray) -> float:
    """Precision-recall AUC (average precision)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / max(1.0, tp[-1])
    # AP = sum over positive points of precision at that recall
    ap = 0.0
    prev_rec = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


def brier(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores) - np.asarray(y)) ** 2))


def ece(scores: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = y[m].mean()
        total += (m.sum() / len(scores)) * abs(conf - acc)
    return float(total)


def pairwise_ranking_accuracy(
    scores_a: np.ndarray, scores_b: np.ndarray,
    better_a: np.ndarray,
) -> float:
    """P(R(a_better) > R(a_worse)) over informative pairs."""
    s_a = np.asarray(scores_a, dtype=np.float64)
    s_b = np.asarray(scores_b, dtype=np.float64)
    better = np.asarray(better_a, dtype=bool)
    ok = np.where(
        better, s_a > s_b, s_b > s_a
    )
    return float(ok.mean()) if len(ok) else float("nan")


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


# ---------------------------------------------------------------------------
# identity probes
# ---------------------------------------------------------------------------


def probe_identity(
    X: dict[str, np.ndarray],
    alpha: float = 1.0,
    held_out: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """VLA-ID probe: one-vs-rest ridge classifiers on feature z -> VLA id.

    X: {vla_name: feature matrix}.  Reports:
      - in-domain multiclass accuracy (shortcut diagnostic: >random means
        the representation preserves policy fingerprint);
      - pairwise one-vs-one AUROC for every pair (separability);
      - held-out rows' predicted-class distribution (if held_out given).
    """
    names = sorted(X)
    n_classes = len(names)
    Xall = np.concatenate([X[n] for n in names])
    models = {}
    for name in names:
        y = np.array([1.0 if n == name else 0.0 for n in names
                      for _ in range(len(X[n]))])
        models[name] = fit_ridge(Xall, y, alpha=alpha)
    argmax_parts = []
    for name in names:
        ps = np.stack([
            ridge_predict(X[name], *models[k]) for k in names
        ], axis=1)
        argmax_parts.append(ps.argmax(axis=1))
    argmax = np.concatenate(argmax_parts)
    n_c = sum(int((argmax[i * len(X[n]):(i + 1) * len(X[n])] == i).sum())
              for i, n in enumerate(names))
    n_t = sum(len(v) for v in X.values())

    pair_auc: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s_ab = ridge_predict(np.concatenate([X[a], X[b]]),
                                 *models[a])
            y_ab = np.array([1.0] * len(X[a]) + [0.0] * len(X[b]))
            pair_auc[f"{a}_vs_{b}"] = round(auroc(s_ab, y_ab), 4)

    held_dist = {}
    if held_out:
        for name, rows in held_out.items():
            ps = np.stack([
                ridge_predict(rows, *models[k]) for k in names
            ], axis=1)
            cls = ps.argmax(axis=1)
            held_dist[name] = {
                "n": int(len(cls)),
                "predicted_class_counts": {
                    k: int((cls == i).sum()) for i, k in enumerate(names)
                },
            }
    return {
        "n_classes": n_classes,
        "rows_per_class": {k: len(v) for k, v in X.items()},
        "in_domain_multiclass_accuracy": round(n_c / max(1, n_t), 4),
        "pairwise_auroc": pair_auc,
        "held_out_prediction_distribution": held_dist,
        "random_baseline": round(1.0 / n_classes, 4),
    }

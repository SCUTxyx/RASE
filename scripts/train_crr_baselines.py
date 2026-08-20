#!/usr/bin/env python3
"""P1: CRR (Counterfactual Residual Risk) baselines on same-root pairs.  v2.

Models (pure numpy, weighted pairwise logistic loss):
  A-lin   : shared linear scorer r = w^T x + b, P(i>j) = sigmoid(r_i - r_j);
            x = [s_t, bigram, a]  (linear -> within-root only chunk diff counts)
  A-mlp   : shared 2x64 ReLU scorer, pairwise loss (can use state x chunk
            interactions; 3-seed ensemble)
  B-ridge : explicit delta model, x_ij = [s_t, bigram, a_i, a_j, da, |da|]
  B-noctx : B without s_t/bigram (pure chunk-pair signal)
  C-mlp   : antisymmetric g_ij = h(s,a_i,a_j) - h(s,a_j,a_i), h = 2x64 MLP
            (3-seed ensemble)
  C-noctx : C without s_t (chunk-pair only antisymmetric MLP)

Baselines:
  random     : constant 0.5
  majority   : always predict candidate i (label-skew reference)
  task_prior : predict the suite-favorite candidate (task-level memory upper
               reference; pairs without the favorite -> random)
  b1_abs     : frozen same_root_risk.npz absolute scores (closed-loop-v2
               model; IN-SAMPLE caveat: trained on all 648 rows)

Controls:
  perm_label : labels shuffled within root (chunk->q link broken, state/task
               priors kept) -> mechanism decomposition: if a model keeps high
               accuracy on permuted labels it exploits priors, not chunks.

Splits (by root): lovo_goal / random_80 / suite_objs.
Metrics: pairwise accuracy (vs majority), AUROC, margin correlation, switch
precision-coverage, estimated selector gain-coverage on the (spatial,object)
deployment arm over ALL pairs, root-stratified bootstrap CI.

Usage (server, oft env):
  python train_crr_baselines.py \
    --pairs runs/oft_opportunity/crr_pairs.jsonl \
    --risk runs/oft_opportunity/same_root_risk.npz \
    --output runs/oft_opportunity/crr_p1_results.json \
    --fig-dir runs/oft_opportunity/figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rase_common import auroc

RNG_SEED = 0
MLP_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def standardize(X, mean=None, scale=None):
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(0)
        scale = X.std(0)
        scale[scale < 1e-8] = 1.0
    return (X - mean) / scale, mean, scale


def relu(x):
    return np.maximum(x, 0.0)


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def fit_ridge_weighted(X, y, w, alpha=1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-8] = 1.0
    Xs = (X - mean) / scale
    design = np.column_stack((np.ones(len(Xs)), Xs))
    Wd = np.diag(w)
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    y_mean = float((w * y).sum() / w.sum())
    beta = np.linalg.solve(design.T @ Wd @ design + penalty,
                           design.T @ (w * (y - y_mean)))
    return mean, scale, beta, y_mean


def ridge_predict_w(X, mean, scale, beta, y_mean):
    Xs = (np.asarray(X, dtype=np.float64) - mean) / scale
    logit = y_mean + beta[0] + Xs @ beta[1:]
    return sigmoid(logit)


# ---------------------------------------------------------------------------
# models (pairwise views: Xa = features of candidate i, Xb = candidate j)
# ---------------------------------------------------------------------------

def train_shared_linear(Xa, Xb, y, w, iters=5000, lr=0.05, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    d = Xa.shape[1]
    wv = rng.normal(0, 0.01, d)
    b = 0.0
    for _ in range(iters):
        p = sigmoid((Xa @ wv + b) - (Xb @ wv + b))
        g = (p - y) * w
        wv -= lr * ((g[:, None] * (Xa - Xb)).mean(0) + l2 * wv)
        b -= lr * float(g.mean())
    return wv, b


def predict_shared_linear(Xa, Xb, wv, b):
    return sigmoid((Xa @ wv + b) - (Xb @ wv + b))


def train_shared_mlp(Xa, Xb, y, w, hid=64, iters=3000, lr=5e-3, l2=1e-4,
                     seed=0):
    rng = np.random.default_rng(seed)
    d = Xa.shape[1]
    W1 = rng.normal(0, 0.1, (d, hid))
    b1 = np.zeros(hid)
    W2 = rng.normal(0, 0.1, hid)
    n = len(y)
    for _ in range(iters):
        za = Xa @ W1 + b1
        ha = relu(za)
        zb = Xb @ W1 + b1
        hb = relu(zb)
        p = sigmoid((ha @ W2) - (hb @ W2))
        g = (p - y) * w
        dW2 = ((ha - hb) * g[:, None]).mean(0) + l2 * W2
        dha = g[:, None] * W2[None, :]
        dhb = -g[:, None] * W2[None, :]
        dza = dha * (za > 0)
        dzb = dhb * (zb > 0)
        dW1 = (Xa.T @ dza + Xb.T @ dzb) / n + l2 * W1
        db1 = (dza.sum(0) + dzb.sum(0)) / n
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
    return W1, b1, W2


def predict_shared_mlp(Xa, Xb, W1, b1, W2):
    return sigmoid((relu(Xa @ W1 + b1) @ W2) - (relu(Xb @ W1 + b1) @ W2))


def train_antisym_mlp(Xij, Xji, y, w, hid=64, iters=3000, lr=5e-3, l2=1e-4,
                      seed=0):
    """g_ij = h(x_ij) - h(x_ji), h = 2-layer ReLU, pairwise loss."""
    rng = np.random.default_rng(seed)
    d = Xij.shape[1]
    W1 = rng.normal(0, 0.1, (d, hid))
    b1 = np.zeros(hid)
    W2 = rng.normal(0, 0.1, hid)
    n = len(y)
    for _ in range(iters):
        z_f = Xij @ W1 + b1
        h_f = relu(z_f)
        z_r = Xji @ W1 + b1
        h_r = relu(z_r)
        p = sigmoid((h_f @ W2) - (h_r @ W2))
        g = (p - y) * w
        dW2 = ((h_f - h_r) * g[:, None]).mean(0) + l2 * W2
        dh_f = g[:, None] * W2[None, :]
        dh_r = -g[:, None] * W2[None, :]
        dz_f = dh_f * (z_f > 0)
        dz_r = dh_r * (z_r > 0)
        dW1 = (Xij.T @ dz_f + Xji.T @ dz_r) / n + l2 * W1
        db1 = (dz_f.sum(0) + dz_r.sum(0)) / n
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
    return W1, b1, W2


def predict_antisym_mlp(Xij, Xji, W1, b1, W2):
    return sigmoid((relu(Xij @ W1 + b1) @ W2) - (relu(Xji @ W1 + b1) @ W2))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def pair_features(pairs: list[dict]) -> dict:
    s_t = np.stack([np.asarray(p["s_t"], dtype=np.float64) for p in pairs])
    bg = np.stack([np.asarray(p["bigram"], dtype=np.float64) for p in pairs])
    ai = np.stack([np.asarray(p["a_i"], dtype=np.float64) for p in pairs])
    aj = np.stack([np.asarray(p["a_j"], dtype=np.float64) for p in pairs])
    da = ai - aj
    ada = np.abs(ai - aj)
    y = np.array([p["y"] for p in pairs])
    dq = np.array([p["dq"] for p in pairs])
    tie = np.array([p["tie"] for p in pairs])
    return {
        "y": y, "dq": dq, "tie": tie,
        "x_full_i": np.concatenate([s_t, bg, ai], axis=1),
        "x_full_j": np.concatenate([s_t, bg, aj], axis=1),
        "x_delta": np.concatenate([s_t, bg, ai, aj, da, ada], axis=1),
        "x_noctx": np.concatenate([ai, aj, da, ada], axis=1),
        "x_anti_f": np.concatenate([s_t, ai, aj], axis=1),
        "x_anti_r": np.concatenate([s_t, aj, ai], axis=1),
        "x_nc_f": np.concatenate([ai, aj], axis=1),
        "x_nc_r": np.concatenate([aj, ai], axis=1),
    }


def permute_labels(pairs: list[dict], seed: int):
    """Shuffle q among candidates within each root (breaks chunk->q link,
    keeps state/task priors).  Returns (y_perm, dq_perm, w_perm)."""
    roots: dict[tuple, list[int]] = {}
    for i, p in enumerate(pairs):
        roots.setdefault((p["task"], p["episode_idx"], p["decision_idx"]),
                         []).append(i)
    rng = np.random.default_rng(seed)
    y_perm = np.zeros(len(pairs))
    dq_perm = np.zeros(len(pairs))
    for idxs in roots.values():
        if len(idxs) < 2:
            continue
        # collect per-candidate q within this root
        cand_q: dict[str, float] = {}
        for i in idxs:
            p = pairs[i]
            cand_q[p["model_i"]] = p["q_i"]
            cand_q[p["model_j"]] = p["q_j"]
        keys = list(cand_q)
        vals = rng.permutation(list(cand_q.values()))
        shuffled = dict(zip(keys, vals))
        for i in idxs:
            p = pairs[i]
            qi = shuffled[p["model_i"]]
            qj = shuffled[p["model_j"]]
            dq = qi - qj
            dq_perm[i] = dq
            y_perm[i] = 1.0 if dq > 1e-12 else 0.0
    return y_perm, dq_perm


def b1_scores(pairs: list[dict], risk_path: Path) -> np.ndarray:
    with np.load(risk_path, allow_pickle=False) as z:
        mean = z["mean"]
        scale = z["scale"]
        weights = z["weights"]
        intercept = float(z["intercept"])
    out = []
    for p in pairs:
        def score(canon):
            x = np.concatenate([np.asarray(p["s_t"], dtype=np.float64),
                                np.asarray(canon, dtype=np.float64),
                                np.asarray(p["bigram"], dtype=np.float64)])
            logit = intercept + ((x - mean) / scale) @ weights
            return 1.0 / (1.0 + np.exp(-logit))
        pi = score(p["a_i"])
        pj = score(p["a_j"])
        out.append(pi / (pi + pj + 1e-12))
    return np.array(out)


def task_prior_pred(pairs: list[dict]) -> np.ndarray:
    out = []
    for p in pairs:
        fav = {"libero_spatial": "oft_spatial",
               "libero_object": "oft_object"}.get(p["suite"])
        if fav is None or fav not in (p["model_i"], p["model_j"]):
            out.append(0.5)
        elif fav == p["model_i"]:
            out.append(1.0)
        else:
            out.append(0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def roc_points(scores, y):
    order = np.argsort(-scores, kind="mergesort")
    y_s = y[order]
    tpr = np.cumsum(y_s) / max(1.0, y_s.sum())
    fpr = np.cumsum(1 - y_s) / max(1.0, (1 - y_s).sum())
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def eval_pairs(y, dq, tie, scores):
    nz = ~tie
    acc = float(((scores[nz] > 0.5) == (y[nz] > 0.5)).mean()) if nz.any() \
        else float("nan")
    auc = auroc(scores, y) if len(np.unique(y)) > 1 else float("nan")
    mcorr = spearman(2.0 * scores[nz] - 1.0, dq[nz]) if nz.sum() > 1 \
        else float("nan")
    margins = np.abs(scores - 0.5)
    taus = np.unique(margins)[::-1]
    precs, covs = [], []
    for tau in taus:
        m = margins > tau
        if m.sum() == 0:
            continue
        correct = (scores[m] > 0.5) == (y[m] > 0.5)
        precs.append(float(correct.mean()))
        covs.append(float(m.mean()))
    precs = np.array(precs)
    covs = np.array(covs)
    prec_at_cov5 = float("nan")
    cov_at_prec85 = float("nan")
    if len(covs):
        near = np.where(covs >= 0.05)[0]
        if len(near):
            k = near[np.argmin(np.abs(covs[near] - 0.05))]
            prec_at_cov5 = float(precs[k])
        hi = np.where(precs >= 0.85)[0]
        if len(hi):
            j = hi[np.argmax(covs[hi])]
            cov_at_prec85 = float(covs[j])
    return {"accuracy": acc, "auroc": auc, "margin_corr": mcorr,
            "prec_at_coverage5": prec_at_cov5, "cov_at_precision85": cov_at_prec85,
            "n_pairs": int(len(y)), "n_ties": int(tie.sum()),
            "pos_rate": float(y[nz].mean()) if nz.any() else float("nan"),
            "majority_acc": float(max(y[nz].mean(), 1.0 - y[nz].mean()))
            if nz.any() else float("nan")}


def p1_gate(ev):
    """Plan §8 gate for one held-out split/model."""
    basic = ev["accuracy"] >= 0.65 and ev["auroc"] >= 0.70
    deploy = ev["cov_at_precision85"] >= 0.05
    strong = ev["cov_at_precision85"] >= 0.10
    if basic and deploy:
        verdict = "STRONG_PASS" if strong else "PASS"
    elif basic:
        verdict = "PASS_A_ONLY"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "basic_learnable": bool(basic),
            "deployment_potential": bool(deploy), "strong": bool(basic and strong),
            "thresholds": {"accuracy": 0.65, "auroc": 0.70,
                           "precision": 0.85, "coverage": 0.05,
                           "strong_coverage": 0.10}}


def boot_ci(root_data: dict, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(root_data)
    stats = []
    for _ in range(n):
        sc, yy, tt = [], [], []
        for k in rng.choice(keys, size=len(keys), replace=True):
            s, yv, tv = root_data[k]
            sc.append(s)
            yy.append(yv)
            tt.append(tv)
        sc = np.concatenate(sc)
        yy = np.concatenate(yy)
        tt = np.concatenate(tt)
        v = fn(sc, yy, tt)
        if not np.isnan(v):
            stats.append(float(v))
    if not stats:
        return [float("nan"), float("nan")]
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def gain_curve(pairs, scores, tau_grid):
    """Signed selector gain over the suite-favorite default.

    A switch is counted only when the model is above the confidence threshold
    *and* prefers the non-default candidate.  Gain is q(selected)-q(default),
    so a wrong switch is negative.  ``gain`` is averaged over every eligible
    root (abstentions contribute zero), which is the quantity comparable with
    a deployment success-rate improvement.  ``gain_per_switch`` is diagnostic.
    """
    rows = []
    for p, s in zip(pairs, scores):
        if set((p["model_i"], p["model_j"])) != {"oft_spatial", "oft_object"}:
            continue
        fav = {"libero_spatial": "oft_spatial",
               "libero_object": "oft_object"}.get(p["suite"])
        q_by_model = {p["model_i"]: float(p["q_i"]),
                      p["model_j"]: float(p["q_j"])}
        w_ = max(q_by_model, key=q_by_model.get)
        pred_w = p["model_i"] if s > 0.5 else p["model_j"]
        rows.append({"q": q_by_model, "w": w_, "pred_w": pred_w,
                     "m": float(abs(s - 0.5)), "fav": fav})
    if not rows:
        return {"curve": [], "oracle_gain": float("nan"),
                "always_switch_gain": float("nan"), "best_fixed_gain": 0.0,
                "n_deployment_roots": 0}
    curve = []
    for tau in tau_grid:
        switched = [r for r in rows
                    if r["m"] > tau and r["pred_w"] != r["fav"]]
        if not switched:
            curve.append({"tau": float(tau), "gain": 0.0, "coverage": 0.0,
                          "gain_per_switch": float("nan"),
                          "precision": float("nan"), "n_switch": 0})
            continue
        signed = np.array([r["q"][r["pred_w"]] - r["q"][r["fav"]]
                           for r in switched])
        prec = float(np.mean(signed > 0.0))
        curve.append({"tau": float(tau),
                      "gain": float(signed.sum() / len(rows)),
                      "gain_per_switch": float(signed.mean()),
                      "coverage": float(len(switched) / len(rows)),
                      "precision": prec, "n_switch": len(switched)})
    always_signed = [r["q"][r["pred_w"]] - r["q"][r["fav"]]
                     if r["pred_w"] != r["fav"] else 0.0 for r in rows]
    always = float(np.mean(always_signed))
    oracle = float(np.mean([max(r["q"].values()) - r["q"][r["fav"]]
                            for r in rows]))
    return {"curve": curve, "oracle_gain": oracle,
            "always_switch_gain": always, "best_fixed_gain": 0.0,
            "n_deployment_roots": len(rows)}


# ---------------------------------------------------------------------------
# model fitting API (used by splits, permutation control and figures)
# ---------------------------------------------------------------------------

def fit_predict(mname: str, F: dict, tr: np.ndarray, te: np.ndarray,
                ytr: np.ndarray, wtr: np.ndarray, pairs: list[dict],
                risk_path: Path) -> np.ndarray:
    """Train on indices tr (with labels ytr/wtr), predict for te."""
    if mname == "A-lin":
        xi_s, m_, s_ = standardize(F["x_full_i"][tr])
        xj_s, _, _ = standardize(F["x_full_j"][tr], m_, s_)
        wv, b = train_shared_linear(xi_s, xj_s, ytr, wtr)
        xi_t, _, _ = standardize(F["x_full_i"][te], m_, s_)
        xj_t, _, _ = standardize(F["x_full_j"][te], m_, s_)
        return predict_shared_linear(xi_t, xj_t, wv, b)
    if mname == "A-mlp":
        xi_s, m_, s_ = standardize(F["x_full_i"][tr])
        xj_s, _, _ = standardize(F["x_full_j"][tr], m_, s_)
        accs = []
        for seed in MLP_SEEDS:
            W1, b1, W2 = train_shared_mlp(xi_s, xj_s, ytr, wtr, seed=seed)
            xi_t, _, _ = standardize(F["x_full_i"][te], m_, s_)
            xj_t, _, _ = standardize(F["x_full_j"][te], m_, s_)
            accs.append(predict_shared_mlp(xi_t, xj_t, W1, b1, W2))
        return np.mean(accs, axis=0)
    if mname == "B-ridge":
        mB, sB, betaB, ymB = fit_ridge_weighted(F["x_delta"][tr], ytr, wtr)
        return ridge_predict_w(F["x_delta"][te], mB, sB, betaB, ymB)
    if mname == "B-noctx":
        mN, sN, betaN, ymN = fit_ridge_weighted(F["x_noctx"][tr], ytr, wtr)
        return ridge_predict_w(F["x_noctx"][te], mN, sN, betaN, ymN)
    if mname == "C-mlp":
        xf_s, mC, sC = standardize(F["x_anti_f"][tr])
        xr_s, _, _ = standardize(F["x_anti_r"][tr], mC, sC)
        accs = []
        for seed in MLP_SEEDS:
            W1, b1, W2 = train_antisym_mlp(xf_s, xr_s, ytr, wtr, seed=seed)
            xf_t, _, _ = standardize(F["x_anti_f"][te], mC, sC)
            xr_t, _, _ = standardize(F["x_anti_r"][te], mC, sC)
            accs.append(predict_antisym_mlp(xf_t, xr_t, W1, b1, W2))
        return np.mean(accs, axis=0)
    if mname == "C-noctx":
        xf_s, mC, sC = standardize(F["x_nc_f"][tr])
        xr_s, _, _ = standardize(F["x_nc_r"][tr], mC, sC)
        accs = []
        for seed in MLP_SEEDS:
            W1, b1, W2 = train_antisym_mlp(xf_s, xr_s, ytr, wtr, seed=seed)
            xf_t, _, _ = standardize(F["x_nc_f"][te], mC, sC)
            xr_t, _, _ = standardize(F["x_nc_r"][te], mC, sC)
            accs.append(predict_antisym_mlp(xf_t, xr_t, W1, b1, W2))
        return np.mean(accs, axis=0)
    if mname == "task_prior":
        return task_prior_pred([pairs[i] for i in te])
    if mname == "b1_abs":
        return b1_scores([pairs[i] for i in te], risk_path)
    if mname == "random":
        return np.full(len(te), 0.5)
    raise ValueError(mname)


MODEL_ORDER = ["A-lin", "A-mlp", "B-ridge", "B-noctx", "C-mlp", "C-noctx"]
BASELINE_ORDER = ["task_prior", "b1_abs", "majority", "random"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--risk", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--margin-m", type=float, default=0.5)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    F = pair_features(pairs)
    y, dq, tie = F["y"], F["dq"], F["tie"]
    w = np.minimum(1.0, np.abs(dq) / args.margin_m)
    w[tie] = 0.0
    root_of = [f"{p['task']}|{p['episode_idx']}|{p['decision_idx']}"
               for p in pairs]
    json.loads(args.vocab.read_text())  # sanity: vocab exists

    # ---- splits ----
    def split_mask(name):
        if name == "lovo_goal":
            return np.array([p["model_i"] != "oft_goal"
                             and p["model_j"] != "oft_goal" for p in pairs])
        if name == "random_80":
            rng = np.random.default_rng(RNG_SEED)
            roots = sorted(set(root_of))
            tr_roots = set(rng.choice(roots, size=int(round(0.8 * len(roots))),
                                      replace=False))
            return np.array([r in tr_roots for r in root_of])
        if name == "suite_objs":
            return np.array([p["suite"] == "libero_spatial" for p in pairs])
        raise ValueError(name)

    split_names = ["lovo_goal", "random_80", "suite_objs"]
    results: dict = {"schema": "rase-p1-crr/v2",
                     "margin_m": args.margin_m,
                     "n_pairs": len(pairs), "n_roots": len(set(root_of)),
                     "n_ties": int(tie.sum()),
                     "mlp_seeds": MLP_SEEDS,
                     "splits": {}, "figures": {}}

    tau_grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
                0.35, 0.40, 0.45, 0.49]

    for name in split_names:
        tr_mask = split_mask(name)
        tr = np.where(tr_mask)[0]
        te = np.where(~tr_mask)[0]
        res: dict = {"n_train_pairs": int(len(tr)),
                     "n_test_pairs": int(len(te)),
                     "models": {}, "baselines": {}, "perm_label": {},
                     "gain_sim": {}}
        ytr, wtr = y[tr], w[tr]
        yte, dqte, tiete = y[te], dq[te], tie[te]

        scores: dict[str, np.ndarray] = {}
        for mname in MODEL_ORDER + ["task_prior", "b1_abs"]:
            scores[mname] = fit_predict(mname, F, tr, te, ytr, wtr, pairs,
                                        args.risk)
        scores["majority"] = np.full(len(te), 0.999999)  # always predict i
        scores["random"] = np.full(len(te), 0.5)

        for mname, sc in scores.items():
            ev = eval_pairs(yte, dqte, tiete, sc)
            ev["gate_p1"] = p1_gate(ev)
            bucket = "models" if mname in MODEL_ORDER else "baselines"
            res[bucket][mname] = ev

        # ---- permutation control: chunk->q link broken within root ----
        yp, dqp = permute_labels(pairs, seed=0)
        wtr_p = np.minimum(1.0, np.abs(dqp[tr]) / args.margin_m)
        wtr_p[tie[tr]] = 0.0
        for mname in ("B-ridge", "B-noctx", "C-mlp", "C-noctx"):
            sc_p = fit_predict(mname, F, tr, te, yp[tr], wtr_p, pairs,
                               args.risk)
            res["perm_label"][mname] = eval_pairs(yte, dqte, tiete, sc_p)

        # ---- bootstrap CI for B-ridge & C-mlp (acc + auroc) ----
        root_data: dict = {}
        for i, r in enumerate(te):
            root_data.setdefault(root_of[i], ([], [], []))
            root_data[root_of[i]][0].append(scores["B-ridge"][i])
            root_data[root_of[i]][1].append(yte[i])
            root_data[root_of[i]][2].append(tiete[i])
        root_data = {k: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
                     for k, v in root_data.items()}
        acc_fn = lambda sc_, yy_, tt_: float(
            (((sc_ > 0.5) == (yy_ > 0.5)) & ~tt_).mean())
        auc_fn = lambda sc_, yy_, tt_: auroc(sc_, yy_)
        for mname in ("B-ridge", "C-mlp"):
            rd = {}
            for i, r in enumerate(te):
                rd.setdefault(root_of[i], ([], [], []))
                rd[root_of[i]][0].append(scores[mname][i])
                rd[root_of[i]][1].append(yte[i])
                rd[root_of[i]][2].append(tiete[i])
            rd = {k: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
                  for k, v in rd.items()}
            res["models"][mname]["acc_ci95"] = boot_ci(rd, acc_fn,
                                                       args.bootstrap_n)
            res["models"][mname]["auroc_ci95"] = boot_ci(rd, auc_fn,
                                                         args.bootstrap_n)

        # ---- gain simulation on the deployment arm over ALL pairs ----
        for mname in ("B-ridge", "C-mlp", "A-mlp", "task_prior", "b1_abs",
                      "random"):
            sc_all = fit_predict(mname, F, tr, np.arange(len(pairs)), ytr,
                                 wtr, pairs, args.risk)
            res["gain_sim"][mname] = gain_curve(pairs, sc_all, tau_grid)

        results["splits"][name] = res

    # ---- figures ----
    if args.fig_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig_paths = {}
        te_idx = np.where(~split_mask("lovo_goal"))[0]
        yte_l = y[te_idx]
        tr_idx = np.where(split_mask("lovo_goal"))[0]

        def all_scores(mname):
            return fit_predict(mname, F, tr_idx, np.arange(len(pairs)), y[tr_idx],
                               w[tr_idx], pairs, args.risk)

        # fig 1: pairwise ROC (lovo_goal test)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="random")
        for mname in MODEL_ORDER:
            sc = all_scores(mname)[te_idx]
            fpr_, tpr_ = roc_points(sc, yte_l)
            plt.plot(fpr_, tpr_, lw=1.5,
                     label=f"{mname} (AUC={auroc(sc, yte_l):.3f})")
        for bname in ("task_prior", "b1_abs"):
            sc = all_scores(bname)[te_idx]
            fpr_, tpr_ = roc_points(sc, yte_l)
            plt.plot(fpr_, tpr_, "o", ms=4, label=bname)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("P1 pairwise ROC (lovo_goal test)")
        plt.legend(fontsize=7)
        p1 = args.fig_dir / "p1_pairwise_roc_lovo_goal.png"
        plt.savefig(p1, dpi=130, bbox_inches="tight")
        plt.close()
        fig_paths["pairwise_roc"] = str(p1)

        # fig 2: pairwise probability calibration (C-mlp & B-ridge)
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        for mname in ("B-ridge", "C-mlp"):
            sc = all_scores(mname)[te_idx]
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (sc >= lo) & (sc < hi)
                if m.sum() < 5:
                    continue
                xs.append(sc[m].mean())
                ys.append(yte_l[m].mean())
            plt.plot(xs, ys, "o-", lw=1.5, label=mname)
        plt.xlabel("predicted P(i>j)")
        plt.ylabel("empirical P(i > j)")
        plt.title("P1 pairwise probability calibration (lovo_goal)")
        plt.legend(fontsize=8)
        p2 = args.fig_dir / "p1_margin_calibration.png"
        plt.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close()
        fig_paths["margin_calibration"] = str(p2)

        # fig 3: precision-coverage (lovo_goal test)
        plt.figure(figsize=(6, 5))
        for mname in MODEL_ORDER + ["task_prior", "b1_abs"]:
            sc = all_scores(mname)[te_idx]
            margins = np.abs(sc - 0.5)
            taus = np.unique(margins)[::-1]
            precs, covs = [], []
            for tau in taus:
                m = margins > tau
                if m.sum() == 0:
                    continue
                precs.append(((sc[m] > 0.5) == (yte_l[m] > 0.5)).mean())
                covs.append(m.mean())
            plt.plot(covs, precs, lw=1.5, label=mname)
        plt.axhline(0.85, color="gray", ls=":", lw=1)
        plt.xlabel("coverage")
        plt.ylabel("switch precision")
        plt.title("P1 switch precision-coverage (lovo_goal)")
        plt.legend(fontsize=7)
        p3 = args.fig_dir / "p1_precision_coverage.png"
        plt.savefig(p3, dpi=130, bbox_inches="tight")
        plt.close()
        fig_paths["precision_coverage"] = str(p3)

        # fig 4: gain-coverage (deployment arm, lovo_goal-trained model)
        plt.figure(figsize=(6, 5))
        for mname in ("B-ridge", "C-mlp", "A-mlp", "task_prior", "b1_abs"):
            gs = results["splits"]["lovo_goal"]["gain_sim"][mname]
            covs = [c["coverage"] for c in gs["curve"]]
            gains = [c["gain"] for c in gs["curve"]]
            plt.plot(covs, gains, "o-", ms=3, lw=1.2, label=mname)
        plt.axhline(0, color="k", lw=0.8)
        plt.xlabel("coverage")
        plt.ylabel("estimated gain (z-units)")
        plt.title("P1 estimated selector gain-coverage (deployment arm)")
        plt.legend(fontsize=7)
        p4 = args.fig_dir / "p1_gain_coverage.png"
        plt.savefig(p4, dpi=130, bbox_inches="tight")
        plt.close()
        fig_paths["gain_coverage"] = str(p4)
        results["figures"] = fig_paths

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    # ---- console table ----
    print("=" * 96)
    print(f"P1 CRR v2 | pairs={len(pairs)} roots={len(set(root_of))} "
          f"ties={int(tie.sum())}")
    for name in split_names:
        r = results["splits"][name]
        print(f"\n--- split {name} (train {r['n_train_pairs']} / "
              f"test {r['n_test_pairs']}; test pos_rate "
              f"{r['models']['B-ridge']['pos_rate']:.3f}) ---")
        print(f"{'model':<11}{'acc':>7}{'auroc':>7}{'mcorr':>7}"
              f"{'prec@cov5':>10}{'cov@prec85':>10}  perm-acc")
        for mname in MODEL_ORDER + BASELINE_ORDER:
            ev = r["models"].get(mname) or r["baselines"].get(mname)
            pe = r["perm_label"].get(mname)
            pea = f"{pe['accuracy']:.3f}" if pe else "   -  "
            print(f"{mname:<11}{ev['accuracy']:>7.3f}{ev['auroc']:>7.3f}"
                  f"{ev['margin_corr']:>7.3f}{ev['prec_at_coverage5']:>10.3f}"
                  f"{ev['cov_at_precision85']:>10.3f}  {pea}")
        for mname in ("B-ridge", "C-mlp"):
            m = r["models"][mname]
            print(f"{mname} acc CI95 {m['acc_ci95']}  auroc CI95 "
                  f"{m['auroc_ci95']}")
        gs = r["gain_sim"]["C-mlp"]
        print(f"gain-sim(C-mlp): oracle={gs['oracle_gain']:.3f} "
              f"always-switch={gs['always_switch_gain']:.3f} "
              f"n_arm={gs['n_deployment_roots']}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

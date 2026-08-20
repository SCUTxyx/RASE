#!/usr/bin/env python3
"""CRR (Counterfactual Residual Risk) shared module: same-root pair
construction and consequence scores.  Pure numpy — runs locally and on the
server (conda env: oft).  Reuses rase_common canonical features.

q definition v1 (available from same_root_w1.jsonl):
  q_i = z(Progress_i) - w2 * z(Drift_i) + w3 * Recoverability_i
    Progress_i   = consequence_label = ||eef_pos(s_{t+H}) - eef_pos(s_t)||
    Drift_i      = mean over objects of ||xyz(end) - xyz(start)||
    Recoverability_i = 0 in v1 (needs reference rollouts; P3 collection)
  z = standardize across all rows.

Pair record fields:
  task, suite, root_key, model_i, model_j,
  s_t (8), bigram (V), a_i (24 canonical), a_j (24), da (24), ada (24),
  chunk_raw_i (56), chunk_raw_j (56),        # for B1-absolute baseline reuse
  q_i, q_j, dq (q_i - q_j), y (1 if q_i > q_j), tie (|dq| < 1e-12)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from rase_common import canonical_chunk_features, build_bigram_vocab, \
    bigram_features


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def root_key(row: dict) -> tuple:
    return (row["task"], row["episode_idx"], row["decision_idx"])


def group_roots(rows: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(root_key(r), []).append(r)
    return out


def object_drift(row: dict) -> float:
    """Mean per-object ||xyz(end) - xyz(start)|| over matched objects."""
    start = {p[0]: np.asarray(p[1], dtype=np.float64) for p in row.get("s_t_objects", [])}
    end = row.get("future_objects")
    if not start or not end:
        return 0.0
    last = {p[0]: np.asarray(p[1], dtype=np.float64) for p in end[-1]}
    ds = [float(np.linalg.norm(last[k] - start[k])) for k in start if k in last]
    return float(np.mean(ds)) if ds else 0.0


def q_scores(rows: list[dict], w2: float = 0.5, w3: float = 0.0) -> np.ndarray:
    """Return standardized q per row (same order as rows)."""
    prog = np.array([float(r["consequence_label"]) for r in rows])
    drift = np.array([object_drift(r) for r in rows])
    pz = (prog - prog.mean()) / (prog.std() + 1e-12)
    dz = (drift - drift.mean()) / (drift.std() + 1e-12)
    q = pz - w2 * dz + w3 * 0.0
    return q


def canonical_of(row: dict) -> np.ndarray:
    arr = np.asarray(row["chunk_raw"], dtype=np.float64).reshape(-1, 7)
    return canonical_chunk_features(arr)


def build_pairs(rows: list[dict], vocab: dict[str, int]) -> list[dict]:
    """All C(K,2) pairs per root; s_t/bigram are root-constant."""
    roots = group_roots(rows)
    q = q_scores(rows)
    by_key = {(root_key(r), r["model"]): i for i, r in enumerate(rows)}
    pairs: list[dict] = []
    for rk, rs in sorted(roots.items()):
        s_t = np.asarray(rs[0]["s_t_proprio"], dtype=np.float64)
        bg = bigram_features(rs[0]["task"], vocab)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ri, rj = rs[i], rs[j]
                ai = canonical_of(ri)
                aj = canonical_of(rj)
                qi = q[by_key[(root_key(ri), ri["model"])]]
                qj = q[by_key[(root_key(rj), rj["model"])]]
                dq = float(qi - qj)
                pairs.append({
                    "task": ri["task"], "suite": ri["suite"],
                    "episode_idx": ri["episode_idx"],
                    "decision_idx": ri["decision_idx"],
                    "model_i": ri["model"], "model_j": rj["model"],
                    "s_t": s_t.tolist(),
                    "bigram": bg.tolist(),
                    "a_i": ai.tolist(), "a_j": aj.tolist(),
                    "da": (ai - aj).tolist(), "ada": np.abs(ai - aj).tolist(),
                    "chunk_raw_i": [float(x) for x in np.asarray(
                        ri["chunk_raw"], dtype=np.float64).flatten()],
                    "chunk_raw_j": [float(x) for x in np.asarray(
                        rj["chunk_raw"], dtype=np.float64).flatten()],
                    "q_i": qi, "q_j": qj,
                    "dq": dq,
                    "y": 1.0 if dq > 1e-12 else 0.0,
                    "tie": bool(abs(dq) <= 1e-12),
                })
    return pairs


def saved_or_built_vocab(rows: list[dict], vocab_path: Path | None) -> dict[str, int]:
    if vocab_path is not None and vocab_path.is_file():
        return json.loads(vocab_path.read_text())
    return build_bigram_vocab([r["task"] for r in rows])

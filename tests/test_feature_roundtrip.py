#!/usr/bin/env python3
"""Phase 5.1: Feature round-trip test.

Verifies that the feature extraction pipeline is correctly wired:
1. SmolVLA latents are extracted (non-zero)
2. Feature dimensions are correct for all feature levels
3. Different observations produce different features
4. Both collection and evaluation code paths produce valid features
5. Pipeline version and SHA are consistent

Note: SmolVLA select_action uses flow-matching with internal noise
sources, so exact feature equality between two separate calls is not
guaranteed even with identical seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.smolvla_feature_extractor import (
    SmolVLAFeatureExtractor,
    build_feature_vector,
)
from rase.recovery.feature_pipeline import RecoveryFeaturePipeline


def _get_proprio(obs):
    pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
    quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
    return np.concatenate([pos, quat])[:7]


# All feature levels produce 144-dim obs_features:
#   latent(128) + proprio(7) + action(7) + stagnation(2) = 144
# F0/F1 fill the latent portion with zeros; F2 uses real SmolVLA features.
FEATURE_DIM = 128 + 7 + 7 + 2  # 144


def test_roundtrip():
    """Functional and structural correctness of the feature pipeline."""
    checkpoint = "/root/autodl-tmp/RASE/ckpts/smolvla_libero"
    tokenizer_path = (
        "/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/"
        "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    )
    bundle = load_smolvla_policy_bundle(
        checkpoint, device="cuda",
        tokenizer_path=tokenizer_path,
        observation_height=360, observation_width=360,
    )

    pipeline = RecoveryFeaturePipeline(bundle)
    eval_extractor = SmolVLAFeatureExtractor(bundle)
    task_id = "libero_spatial_000002"

    env_a = make_libero_env_for_task(task_id, init_state_id=0, seed=42, libero_flavor="clean")
    env_b = make_libero_env_for_task(task_id, init_state_id=1, seed=43, libero_flavor="clean")
    obs_a = observation_from_libero_env(env_a.vector_env.envs[0])
    obs_b = observation_from_libero_env(env_b.vector_env.envs[0])
    p_a = _get_proprio(obs_a)
    p_b = _get_proprio(obs_b)
    dummy_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    all_pass = True

    # ── 1. Feature dimension checks for all levels ──
    print("── Test 1: Feature dimensions ──")
    dim_checks = {
        "F0": (FEATURE_DIM, pipeline.extract_standalone(p_a, dummy_action, feature_level="F0")),
        "F1": (FEATURE_DIM, pipeline.extract_standalone(p_a, dummy_action, feature_level="F1")),
    }
    # F2 needs observation
    f2_f = pipeline.extract(obs_a, p_a, dummy_action, feature_level="F2")
    dim_checks["F2"] = (FEATURE_DIM, f2_f)

    for level, (expected, feats) in dim_checks.items():
        actual = feats.obs_features.shape[0]
        status = "OK" if actual == expected else f"FAIL (got {actual})"
        print(f"  {level}: {actual} dims {status}")
        if actual != expected:
            all_pass = False

    # ── 2. SmolVLA latent non-zero ──
    print("── Test 2: SmolVLA latent richness ──")
    for name, obs in [("obs_a", obs_a), ("obs_b", obs_b)]:
        latent = eval_extractor.extract(obs)
        nz = np.count_nonzero(latent)
        abs_mean = float(np.mean(np.abs(latent)))
        ok = nz > 0 and abs_mean > 1e-6
        status = f"OK ({nz}/128 nonzero, |mean|={abs_mean:.4f})" if ok else f"FAIL (all zero)"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    # ── 3. Different obs → different features ──
    print("── Test 3: Feature discriminability ──")
    cfa = pipeline.extract(obs_a, p_a, dummy_action, feature_level="F2")
    cfb = pipeline.extract(obs_b, p_b, dummy_action, feature_level="F2")
    feat_ab_diff = np.max(np.abs(cfa.obs_features - cfb.obs_features))
    latent_ab_diff = np.max(np.abs(cfa.smolvla_latent - cfb.smolvla_latent))
    print(f"  feat_obs max_abs_diff:  {feat_ab_diff:.6f}")
    print(f"  latent max_abs_diff:   {latent_ab_diff:.6f}")
    if feat_ab_diff < 1e-4:
        print("  FAIL: features nearly identical for different observations")
        all_pass = False
    else:
        print("  OK")

    # ── 4. Both code paths produce valid features ──
    print("── Test 4: Dual-path validity ──")
    # Collector path (uses pipeline.extract)
    col = pipeline.extract(obs_a, p_a, dummy_action, feature_level="F2")
    # Evaluator path (uses extractor + build_feature_vector)
    lat_eval = eval_extractor.extract(obs_a)
    feat_eval = build_feature_vector(lat_eval, p_a, dummy_action, feature_level="F2")
    col_ok = col.obs_features.shape == (FEATURE_DIM,) and np.isfinite(col.obs_features).all()
    eval_ok = feat_eval.shape == (FEATURE_DIM,) and np.isfinite(feat_eval).all()
    print(f"  collector path: {'OK' if col_ok else 'FAIL'}")
    print(f"  evaluator path: {'OK' if eval_ok else 'FAIL'}")
    if not col_ok or not eval_ok:
        all_pass = False

    # ── 5. Pipeline metadata ──
    print("── Test 5: Pipeline metadata ──")
    print(f"  version: {col.pipeline_version}")
    print(f"  sha:     {pipeline.extractor_sha}")
    ok = col.pipeline_version == "route-c-recovery/v2" and len(pipeline.extractor_sha) >= 8
    print(f"  {'OK' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    # ── 6. F0 standalone produces deterministic output (no latent involved) ──
    print("── Test 6: F0 standalone determinism ──")
    f0a = pipeline.extract_standalone(p_a, dummy_action, feature_level="F0")
    f0b = pipeline.extract_standalone(p_a, dummy_action, feature_level="F0")
    f0_diff = np.max(np.abs(f0a.obs_features - f0b.obs_features))
    print(f"  F0 same-input max_abs_diff: {f0_diff:.10f}")
    if f0_diff > 1e-8:
        print("  FAIL: F0 not deterministic")
        all_pass = False
    else:
        print("  OK")

    # ── Summary ──
    print()
    if all_pass:
        print("ROUND-TRIP TEST: PASS")
    else:
        print("ROUND-TRIP TEST: FAIL")

    env_a.close()
    env_b.close()
    return all_pass


if __name__ == "__main__":
    ok = test_roundtrip()
    raise SystemExit(0 if ok else 1)

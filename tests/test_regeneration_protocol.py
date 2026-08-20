from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_is_metadata_only_and_preserves_order() -> None:
    mod = _load("freeze_regeneration_keys", "scripts/freeze_regeneration_keys.py")
    records = [
        {"state_key": "a", "suite": "Goal", "perturbation_dimension": "clean", "step": 0, "task_id": "g0"},
        {"state_key": "b", "suite": "Long", "perturbation_dimension": "robot", "step": 0, "task_id": "l0"},
        {"state_key": "c", "suite": "Long", "perturbation_dimension": "clean", "step": 2, "task_id": "l0"},
    ]
    result = mod.freeze(
        {"artifact_version": "source/v1", "pool": "pool", "records": records, "state_keys": ["c", "a", "b"]},
        suites={"Goal", "Long"},
        dimensions={"clean"},
        steps={0, 2},
    )
    assert result["state_keys"] == ["c", "a"]
    assert result["selection_uses_outcomes"] is False
    assert result["n_tasks"] == 2


def test_common_root_seed_pairs_candidates() -> None:
    mod = _load("rollout_pool_candidates", "scripts/rollout_pool_candidates.py")

    def fake_seed(state: str, candidate: int, rollout: int) -> int:
        return hash((state, candidate, rollout)) & 0xFFFFFFFF

    a = mod._continuation_seed(fake_seed, "s", 0, 2, mode="common_root_rollout")
    b = mod._continuation_seed(fake_seed, "s", 3, 2, mode="common_root_rollout")
    assert a == b
    assert mod._continuation_seed(fake_seed, "s", 0, 2, mode="candidate_specific") != mod._continuation_seed(fake_seed, "s", 3, 2, mode="candidate_specific")
    with pytest.raises(ValueError):
        mod._continuation_seed(fake_seed, "s", 0, 0, mode="bad")


def test_analyzer_opens_both_gates_on_cross_policy_headroom() -> None:
    mod = _load("analyze_regeneration_opportunity", "scripts/analyze_regeneration_opportunity.py")
    records = [
        {"state_key": f"s{i}", "task_id": "t0" if i < 2 else "t1", "episode_id": f"e{i}", "suite": "Goal" if i < 2 else "Long", "step": i}
        for i in range(4)
    ]
    keys = {
        "artifact_version": "rase-regeneration-state-keys/v1",
        "selection_uses_outcomes": False,
        "state_keys": [f"s{i}" for i in range(4)],
        "state_keys_sha256": "sum",
        "records": records,
    }
    outcomes = [[False, True], [True, True], [False, True], [False, False]]
    resample = {
        "continuation_seed_mode": "common_root_rollout",
        "state_keys_provenance": {"selected_state_keys_sha256": "sum"},
        "per_state": [
            {"state_key": f"s{i}", "candidates": [{"trials": 1, "successes": int(v)} for v in values]}
            for i, values in enumerate(outcomes)
        ],
    }
    source = {"per_pair": [
        {"state_key": f"s{i}", "continue_smol_active_chunk": i == 1}
        for i in range(4)
    ]}
    fallback = {"per_state": [
        {"state_key": f"s{i}", "direct_oft_success": i in {1, 2}}
        for i in range(4)
    ]}
    generation = {
        "state_keys": [f"s{i}" for i in range(4)],
        "diversity": {
            "mean_chunk_l2": 0.2,
            "per_state": [
                {"mean_pairwise_chunk_l2": value}
                for value in (0.1, 0.2, 0.3, 0.4)
            ],
        },
    }
    result = mod.analyze(
        keys, resample, source, [fallback], generation,
        bootstrap_replicates=100, bootstrap_seed=7,
    )
    assert result["verifier_training_gate"] == "open"
    assert result["cross_policy_claim_gate"] == "open"
    assert result["portfolio"]["resample_only_vs_fallback"] == 1
    assert result["metrics"]["mixed_outcome_roots"] == 2

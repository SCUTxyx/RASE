from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from rase.collect.lerobot_libero_plus_adapter import _queued_env_action_suffix
from rase.interventions.decision_context import (
    build_decision_context,
    finalize_source_suffix_parity,
    strict_continue_suffix,
    validate_decision_context,
)


def test_decision_context_roundtrip_and_checksum():
    suffix = np.arange(35, dtype=np.float32).reshape(5, 7)
    context = build_decision_context(
        source_policy="smolvla:test",
        snapshot_env_step=15,
        action_chunk_size=10,
        action_chunk_offset=5,
        active_action_suffix=suffix,
        public_action_history=np.zeros((4, 7), dtype=np.float32),
        public_proprio_history=np.zeros((4, 8), dtype=np.float32),
        public_observation_history=[{"env_step": 15, "observations": {"agentview": "key"}}],
    )
    finalize_source_suffix_parity(context, suffix.copy())
    validate_decision_context(context)
    np.testing.assert_array_equal(
        strict_continue_suffix({"decision_context": context}), suffix
    )


def test_decision_context_rejects_boundary_snapshot_and_tampering():
    suffix = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly inside"):
        build_decision_context(
            source_policy="smolvla:test",
            snapshot_env_step=10,
            action_chunk_size=10,
            action_chunk_offset=0,
            active_action_suffix=np.zeros((10, 7), dtype=np.float32),
        )
    context = build_decision_context(
        source_policy="smolvla:test",
        snapshot_env_step=15,
        action_chunk_size=10,
        action_chunk_offset=5,
        active_action_suffix=suffix,
    )
    finalize_source_suffix_parity(context, suffix.copy())
    context["active_action_suffix"][0, 0] = 1.0
    with pytest.raises(ValueError, match="checksum"):
        validate_decision_context(context)


def test_queued_suffix_is_converted_to_env_space_without_consuming_queue():
    import torch

    queue = deque(
        [torch.full((1, 7), float(index), dtype=torch.float32) for index in range(3)]
    )
    policy = type("Policy", (), {"_queues": {"action": queue}})()
    bundle = {
        "postprocessor": lambda value: value + 1,
        "env_postprocessor": lambda transition: {"action": transition["action"] * 2},
    }
    suffix = _queued_env_action_suffix(bundle, policy)
    assert suffix.shape == (3, 7)
    np.testing.assert_array_equal(suffix[:, 0], np.asarray([2, 4, 6]))
    assert len(queue) == 3


def test_strict_continue_rejects_legacy_state():
    with pytest.raises(ValueError, match="no decision_context"):
        strict_continue_suffix({"snapshot_format": "rase.forkable_env/v1"})

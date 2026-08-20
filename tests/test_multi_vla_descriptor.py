from __future__ import annotations

import numpy as np
import pytest

from rase.risk.multi_vla_descriptor import behavior_descriptor, descriptors_by_policy


def test_behavior_descriptor_is_fixed_and_outcome_free() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (10, 2, 3, 16, 16), dtype=np.uint8)
    proprio = rng.normal(size=(10, 8)).astype(np.float32)
    action = rng.normal(size=(10, 20)).astype(np.float32)
    value = behavior_descriptor(image, proprio, action)
    assert value.shape == (80,)
    assert np.isfinite(value).all()


def test_descriptors_use_only_selected_rows_and_require_support() -> None:
    data = {
        "image": np.zeros((16, 2, 3, 8, 8), dtype=np.uint8),
        "proprio": np.zeros((16, 8), dtype=np.float32),
        "action_summary": np.zeros((16, 20), dtype=np.float32),
        "policy_id": np.asarray(["a"] * 8 + ["b"] * 8),
        # Deliberately present forbidden outcomes: the API has no path to read them.
        "source_failure": np.asarray([0, 1] * 8),
    }
    result = descriptors_by_policy(data, np.arange(16))
    assert set(result) == {"a", "b"}
    with pytest.raises(ValueError, match="only 7"):
        descriptors_by_policy(data, np.arange(7))

import io

import numpy as np
from PIL import Image

from rase.selector.features import extract_deployable_features


def _png(value):
    stream = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(stream, "PNG")
    return stream.getvalue()


def test_features_only_use_current_observation_and_proprio():
    result = extract_deployable_features(
        observations={"agentview": _png(128), "wrist": _png(64)},
        proprio=np.asarray([1.0, -2.0, 3.0], dtype=np.float32),
        t0=7,
    )
    assert result["t0"] == 7
    assert result["proprio_l1"] == 6
    assert result["image_agentview_mean"] > result["image_wrist_mean"]
    assert not {"level", "perturb_dim", "episode_outcome"} & set(result)

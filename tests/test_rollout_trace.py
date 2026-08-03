import json

import numpy as np
import pytest

from rase.collect.rollout_trace import RolloutTraceRecorder, observation_montage


def _obs(value=0):
    return {
        "pixels": {
            "image": np.full((1, 8, 6, 3), value, dtype=np.uint8),
            "image2": np.full((1, 8, 4, 3), value + 1, dtype=np.uint8),
        }
    }


def test_observation_montage_combines_agent_and_wrist():
    frame = observation_montage(_obs())
    assert frame.shape == (8, 10, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[:, :6] == 0)
    assert np.all(frame[:, 6:] == 1)


def test_trace_recorder_stride_cap_and_archive(tmp_path):
    pytest.importorskip("PIL")
    recorder = RolloutTraceRecorder(stride=2, max_frames=2, jpeg_quality=80)
    for step in range(6):
        recorder(_obs(step), phase="candidate", timestep=step)
    assert len(recorder.frames) == 2
    assert [frame.timestep for frame in recorder.frames] == [0, 2]

    manifest_path = recorder.write_frame_archive(
        tmp_path / "trace",
        metadata={"state_key": "sp1_demo", "success": True},
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == "rase-rollout-trace/v1"
    assert manifest["frames_seen"] == 6
    assert manifest["frames_saved"] == 2
    assert manifest["metadata"]["success"] is True
    assert (manifest_path.parent / "00000.jpg").is_file()


def test_trace_recorder_validation():
    with pytest.raises(ValueError, match="stride"):
        RolloutTraceRecorder(stride=0)
    with pytest.raises(ValueError, match="max_frames"):
        RolloutTraceRecorder(max_frames=0)

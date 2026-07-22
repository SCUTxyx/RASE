import json
import zipfile

import numpy as np
import pytest

from rase.envs.snapshot import EnvSnapshot, SnapshotError


def test_snapshot_roundtrip_is_pickle_free(tmp_path):
    snapshot = EnvSnapshot(
        task_fingerprint="task-sha256",
        payload={
            "sim": np.arange(12, dtype=np.float64).reshape(3, 4),
            "rng": ("MT19937", np.arange(8, dtype=np.uint32), 7, 0, 0.0),
            "nested": [True, None, {"count": np.int64(3)}],
        },
    )

    json_path, npz_path = snapshot.save(tmp_path / "state")
    restored = EnvSnapshot.load(json_path)

    assert restored.version == snapshot.version
    assert restored.task_fingerprint == snapshot.task_fingerprint
    np.testing.assert_array_equal(restored.payload["sim"], snapshot.payload["sim"])
    np.testing.assert_array_equal(restored.payload["rng"][1], snapshot.payload["rng"][1])
    assert restored.payload["rng"][0] == "MT19937"
    assert restored.payload["nested"] == [True, None, {"count": 3}]

    manifest = json.loads(json_path.read_text())
    assert manifest["format"] == "rase.forkable_env"
    with zipfile.ZipFile(npz_path) as archive:
        assert all(not name.endswith((".pkl", ".pickle")) for name in archive.namelist())


def test_object_arrays_are_rejected(tmp_path):
    snapshot = EnvSnapshot(
        task_fingerprint="task-sha256",
        payload={"unsafe": np.array([object()], dtype=object)},
    )
    with pytest.raises(SnapshotError, match="object-dtype"):
        snapshot.save(tmp_path / "unsafe")


def test_manifest_tampering_is_detected(tmp_path):
    snapshot = EnvSnapshot(
        task_fingerprint="task-sha256",
        payload={"array": np.array([1.0, 2.0])},
    )
    json_path, _ = snapshot.save(tmp_path / "state")
    manifest = json.loads(json_path.read_text())
    manifest["arrays"]["a000000"]["shape"] = [99]
    json_path.write_text(json.dumps(manifest))

    with pytest.raises(SnapshotError, match="manifest validation"):
        EnvSnapshot.load(json_path)

import json

import pytest

from rase.collect.run_manifest import (
    build_run_manifest,
    write_run_manifest,
)


def test_write_manifest_is_idempotent_and_rejects_drift(tmp_path):
    candidates = tmp_path / "cands"
    candidates.mkdir()
    (candidates / "a.npz").write_bytes(b"abc")
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "manifest.json").write_text("{}", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "env.lock.md").write_text("lock\n", encoding="utf-8")

    manifest = build_run_manifest(
        repo_root=repo,
        resolved_config={"mode": "smoke"},
        pool_root=pool,
        candidates_dir=candidates,
        policy_path=None,
        policy_hash="deadbeef",
        protocol_version="wilson-onesided-alpha-spend-v1",
    )
    run_root = tmp_path / "run"
    path = write_run_manifest(run_root, manifest)
    assert path.is_file()
    write_run_manifest(run_root, manifest)  # idempotent

    drifted = dict(manifest)
    drifted["policy_hash"] = "cafebabe"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        write_run_manifest(run_root, drifted)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["policy_hash"] == "deadbeef"

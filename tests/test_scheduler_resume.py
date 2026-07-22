from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey


def test_resume_skips_completed_tuple(tmp_path):
    keys = [
        RolloutKey("state/a", candidate=0, rollout=0),
        RolloutKey("state/a", candidate=0, rollout=1),
    ]
    first = DiskRolloutScheduler(tmp_path)
    assert first.claim(keys[0], "worker-1", now=10) is not None
    first.complete(keys[0], {"success": True}, worker="worker-1")

    resumed = DiskRolloutScheduler(tmp_path)
    assert resumed.is_complete(keys[0])
    assert list(resumed.pending(keys)) == [keys[1]]
    assert resumed.result(keys[0])["result"] == {"success": True}


def test_completion_is_idempotent(tmp_path):
    scheduler = DiskRolloutScheduler(tmp_path)
    key = RolloutKey("same", 2, 7)
    first = scheduler.complete(key, {"success": True})
    second = scheduler.complete(key, {"success": False})
    assert second == first
    assert scheduler.result(key)["result"]["success"] is True


def test_retry_limit_survives_restart(tmp_path):
    key = RolloutKey("retry", 1, 0)
    scheduler = DiskRolloutScheduler(tmp_path, max_attempts=2)
    scheduler.fail(key, "first")
    assert scheduler.can_retry(key)

    resumed = DiskRolloutScheduler(tmp_path, max_attempts=2)
    resumed.fail(key, "second")
    assert not resumed.can_retry(key)
    assert list(resumed.pending([key])) == []


def test_expired_lease_is_reclaimed(tmp_path):
    key = RolloutKey("lease", 0, 0)
    scheduler = DiskRolloutScheduler(tmp_path, lease_seconds=5)
    assert scheduler.claim(key, "old", now=10) is not None
    assert scheduler.claim(key, "new", now=14) is None
    assert scheduler.claim(key, "new", now=15) is not None


def test_same_worker_reclaims_unexpired_lease_after_crash(tmp_path):
    key = RolloutKey("crash", 0, 1)
    scheduler = DiskRolloutScheduler(tmp_path, lease_seconds=3600)
    assert scheduler.claim(key, "pilot-worker", now=100) is not None
    # Crash leaves the claim file; same worker must resume immediately.
    assert scheduler.claim(key, "pilot-worker", now=101) is not None
    assert scheduler.claim(key, "other", now=102) is None

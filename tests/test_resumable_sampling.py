from rase.collect.adaptive import PROTOCOL_SEQUENTIAL_ONESIDED_V1
from rase.collect.resumable_sampling import adaptive_sample_resumable, outcomes_from_scheduler
from rase.collect.scheduler import DiskRolloutScheduler, RolloutKey


def test_resumable_skips_completed_and_matches_fresh(tmp_path):
    scheduler = DiskRolloutScheduler(tmp_path)
    calls = []

    def rollout(index: int):
        calls.append(index)
        return index % 2 == 0

    # Pre-complete first two rollouts.
    for index in (0, 1):
        key = RolloutKey("s", 0, index)
        scheduler.complete(key, {"success": rollout(index)})
    calls.clear()

    result = adaptive_sample_resumable(
        scheduler,
        "s",
        0,
        "w1",
        rollout,
        n_first=6,
        n_total=20,
        protocol_version=PROTOCOL_SEQUENTIAL_ONESIDED_V1,
        alpha_first=0.01,
        alpha_final=0.04,
        sidedness="one-sided",
    )
    assert 0 not in calls and 1 not in calls
    assert result.trials in {6, 20}
    outcomes = outcomes_from_scheduler(scheduler, "s", 0, n_total=result.trials)
    assert all(value is not None for value in outcomes[: result.trials])


def test_resumable_all_fail_early_stop(tmp_path):
    scheduler = DiskRolloutScheduler(tmp_path)
    calls = []

    def rollout(index: int):
        calls.append(index)
        return False

    result = adaptive_sample_resumable(
        scheduler,
        "fail",
        3,
        "worker",
        rollout,
        n_first=6,
        n_total=20,
        protocol_version=PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    )
    assert result.trials == 6
    assert calls == list(range(6))
    assert result.upper < 0.5

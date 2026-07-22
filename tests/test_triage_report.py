from rase.collect.adaptive import PROTOCOL_SEQUENTIAL_ONESIDED_V1
from rase.collect.resumable_sampling import adaptive_sample_resumable
from rase.collect.scheduler import DiskRolloutScheduler
from rase.collect.triage_report import summarize_run, write_json


def test_summarize_run_rebuilds_set_c(tmp_path):
    scheduler = DiskRolloutScheduler(tmp_path / "sched")

    def always_fail(_index: int) -> bool:
        return False

    for candidate in range(8):
        adaptive_sample_resumable(
            scheduler,
            "s0",
            candidate,
            "w",
            always_fail,
            n_first=6,
            n_total=20,
            protocol_version=PROTOCOL_SEQUENTIAL_ONESIDED_V1,
        )

    summary = summarize_run(
        scheduler,
        ["s0"],
        k=8,
        n_first=6,
        n_total=20,
        protocol_version=PROTOCOL_SEQUENTIAL_ONESIDED_V1,
    )
    assert summary["label_counts"]["C"] == 1
    assert summary["total_rollouts_completed"] == 8 * 6
    out = tmp_path / "summary.json"
    write_json(out, summary)
    assert out.is_file()

import json
from pathlib import Path

import pytest

from scripts import generate_pool_candidates, rollout_pool_candidates
from scripts.preflight_runner import _gpu_capacity_failure

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_key_checksum_is_shared_and_provenance_is_recorded(tmp_path):
    keys = ["sp1_a", "sp1_b"]
    checksum = rollout_pool_candidates._state_keys_checksum(keys)
    assert generate_pool_candidates._state_keys_checksum(keys) == checksum

    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps({"state_keys": keys, "state_keys_sha256": checksum}) + "\n",
        encoding="utf-8",
    )
    loaded, provenance = rollout_pool_candidates._load_state_keys_json(path)

    assert loaded == keys
    assert provenance["source"] == str(path.resolve())
    assert provenance["state_keys_sha256"] == checksum
    assert provenance["n_states"] == 2
    assert len(provenance["artifact_sha256"]) == 64


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"state_keys": ["sp1_a", "sp1_a"]}, "duplicate"),
        ({"state_keys": ["sp1_a"], "state_keys_sha256": "wrong"}, "mismatch"),
        ({"state_keys": []}, "no state_keys"),
    ],
)
def test_frozen_key_artifact_rejects_invalid_input(tmp_path, payload, match):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        rollout_pool_candidates._load_state_keys_json(path)


def test_oft_shell_runner_uses_lock_owned_pid_and_resume_default():
    source = (ROOT / "scripts/run_oft_verify_suites.sh").read_text(encoding="utf-8")
    assert 'FRESH_RUN="${FRESH_RUN:-0}"' in source
    assert "flock -n 8" in source
    assert "current_server_pid=$!" in source
    assert "SKIP_COMPLETED" in source
    assert "pgrep" not in source
    assert 'OFT_MIN_FREE_MIB="${OFT_MIN_FREE_MIB:-20000}"' in source
    assert '--min-free-gpu-mib "$OFT_MIN_FREE_MIB"' in source
    assert 'OFT_RUNNER="${OFT_RUNNER:-verify}"' in source
    assert 'OFT_SUITE_SHORTS="${OFT_SUITE_SHORTS:-spatial,object,goal,10}"' in source
    assert "SKIP_EMPTY_SUITE" in source
    assert "rollout_oft_prefix_ablation.py" in source
    assert "generate_oft_pool_candidates.py" in source
    assert 'OFT_RUNNER" != "generate-prefix"' in source


def test_oft_gpu_capacity_gate_is_hard_by_default():
    assert _gpu_capacity_failure(11, minimum_mib=20_000, allow_busy=False)
    assert _gpu_capacity_failure(25_000, minimum_mib=20_000, allow_busy=False) is None
    assert _gpu_capacity_failure(11, minimum_mib=20_000, allow_busy=True) is None


def test_w4_summary_only_guards_all_execution_stages():
    source = (ROOT / "scripts/run_w4_adequate_pipeline.sh").read_text(
        encoding="utf-8"
    )
    summary_guard = source.index('if [[ "$SUMMARY_ONLY" == "1" ]]')
    candidate_stage = source.index("# --- 0) candidates")
    summary_stage = source.index("# --- 3) dual-oracle")
    assert summary_guard < candidate_stage < summary_stage
    assert 'log "SUMMARY_ONLY=1: skipping candidates and both oracle runners"' in source

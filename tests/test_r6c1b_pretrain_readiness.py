from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_r6c1b_label_support import load_groups  # noqa: E402
from audit_r6c1b_pretrain_readiness import auc_score, average_precision  # noqa: E402


def _write_replica(root: Path, replica: int, successes: tuple[bool, bool, bool]) -> None:
    directory = root / "suite_spatial" / "pi0fast_libero" / "train_enrichment" / "seed_1" / f"rep{replica}"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if replica == 0 else f"__rep{replica}"
    path = directory / f"state__seed1{suffix}.json"
    rows = []
    for elapsed, success in zip((0, 8, 16), successes):
        rows.append({
            "policy_id": "pi0fast_libero",
            "seed_index": 1,
            "state_key": "state",
            "task_id": "libero_spatial_000001",
            "suite": "Spatial",
            "instruction": "move the bowl",
            "group_id": "state:pi0fast_libero:seed1",
            "elapsed_source_steps": elapsed,
            "persistent_success_if_enter_now": success,
            "persistent_teacher_steps_if_enter_now": 10 + elapsed,
            "source_final_success": False,
        })
    path.write_text(json.dumps({
        "rollout_index": replica,
        "source_success": False,
        "source_steps": 490,
        "rows": rows,
        "npz": str(path.with_suffix(".npz")),
    }))


def test_replica_majority_does_not_inflate_group_count(tmp_path: Path) -> None:
    _write_replica(tmp_path, 0, (True, False, False))
    _write_replica(tmp_path, 1, (False, False, False))
    _write_replica(tmp_path, 2, (True, False, True))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"status": "frozen", "excluded": []}))

    groups = load_groups([tmp_path], exclusions)
    assert len(groups) == 1
    assert groups[0]["n_replicas"] == 3
    assert groups[0]["boundaries"][0]["success_probability"] == 2 / 3
    assert groups[0]["boundaries"][0]["majority_success"] is True
    assert groups[0]["early_rescuable"] is True


def test_auc_and_ap_handle_ties() -> None:
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    assert auc_score(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    assert auc_score(labels, np.ones(4)) == 0.5

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rase.collect.dataset_export import audit_split_support
from rase.collect.perturb_sampler import sample_perturbations
from rase.collect.pipeline import load_config

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "configs/collect_w10_object_spatial_failures.json"
BENCHMARK = ROOT / "configs/ngc_w10_object_spatial_benchmark.yaml"
REQUIREMENTS = ROOT / "configs/ngc_w10_split_requirements.json"
RUNBOOK = ROOT / "docs/runbooks/w10_object_spatial_benchmark.md"


def test_w10_collection_freeze_is_parseable() -> None:
    config = load_config(COLLECT)
    collection = config["collection"]
    assert collection == {
        "output_dir": "pool/ngc_w10_object_spatial_failures",
        "episodes": 80,
        "seed": 20260731,
        "action_chunks_per_episode": 52,
        "max_action_chunks": None,
        "snapshot_cadence_action_chunks": 2,
        "successful_snapshot_retention": 0.0,
        "dry_run": False,
    }
    assert config["sampling"] == {
        "dimension_quotas": {"camera": 50, "robot": 50},
        "suite_quotas": {"Object": 50, "Spatial": 50},
        "levels_by_dimension": {"camera": [1, 2], "robot": [1, 2]},
    }
    requests = sample_perturbations(
        collection["episodes"], collection["seed"], **config["sampling"]
    )
    assert len(requests) == 80
    assert {x.suite for x in requests} == {"Object", "Spatial"}
    assert {x.dimension for x in requests} == {"camera", "robot"}
    assert {x.level for x in requests} == {1, 2}
    assert sum(x.suite == "Object" for x in requests) == 40
    assert sum(x.suite == "Spatial" for x in requests) == 40
    assert sum(x.dimension == "camera" for x in requests) == 40
    assert sum(x.dimension == "robot" for x in requests) == 40


def test_w10_benchmark_freeze_is_parseable() -> None:
    config = yaml.safe_load(BENCHMARK.read_text(encoding="utf-8"))
    assert config["pool"] == "pool/ngc_w10_object_spatial_failures"
    assert config["mode"] == "direct-policy-controls"
    assert config["sample"] == {
        "strategy": "stratified",
        "per_cell": 2,
        "sample_seed": 20260731,
        "selection": "random",
        "strata": ["suite", "dim", "level"],
        "dims": ["camera", "robot"],
        "suites": ["Object", "Spatial"],
        "levels": [1, 2],
        "episode_outcomes": ["failure"],
        "distinct_episodes": True,
        "min_remaining_steps": 100,
    }
    assert config["protocol"]["expected_n_states"] == 16
    assert config["adapter"]["continuation_temperature"] == 0.5
    assert (
        config["candidates"]["policy_hash"]
        == "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
    )


def test_w10_split_requirements_use_supported_audit_fields() -> None:
    requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
    result = audit_split_support(
        [], {"splits": {"train": [], "val": [], "test": []}}, requirements=requirements
    )
    assert result["status"] == "NOT_READY"
    assert result["requirements"]["required_suites"] == {
        "train": ["Object", "Spatial"],
        "val": ["Object", "Spatial"],
        "test": ["Object", "Spatial"],
    }


def test_w10_runbook_orders_outcomes_dataset_and_split_gate() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    smol = text.index("python scripts/rollout_direct_smol.py")
    oft = text.index("OFT_RUNNER=prefix-ablation")
    dataset = text.index("python scripts/export_selector_action_dataset.py")
    clean_filter = text.index("python scripts/filter_selector_dataset.py", dataset)
    merge = text.index("python scripts/merge_selector_datasets.py", clean_filter)
    split = text.index("python scripts/build_selector_splits.py", merge)
    assert smol < dataset and oft < dataset < clean_filter < merge < split
    assert "MANUAL BLOCKED — identity freeze" in text
    assert "MANUAL BLOCKED — prior-cohort exclusion" in text
    assert "MANUAL BLOCKED — evaluation schedule/seed" in text
    assert "MANUAL BLOCKED — W9C clean identity and cross-source audit" in text
    assert "runs/ngc_w9c_clean_action_dataset.jsonl" in text
    assert "--manifest runs/ngc_w10_object_spatial_heldout_merge_manifest.json" in text
    assert "Never run `train_lightweight_selector.py`" in text
    assert "runs/ngc_w10_direct_oft_object_object_spatial16/summary.json" in text
    assert "runs/ngc_w10_direct_oft_spatial_object_spatial16/summary.json" in text

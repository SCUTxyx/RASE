import json

import pytest

from rase.collect.lerobot_libero_plus_adapter import (
    _CatalogTask,
    _load_catalog,
    select_catalog_task,
)
from rase.collect.perturb_sampler import PerturbationRequest, sample_perturbations


def request(**changes):
    values = {
        "index": 0,
        "suite": "Spatial",
        "dimension": "camera",
        "subdimension": "viewpoint",
        "level": 3,
        "seed": 7,
    }
    values.update(changes)
    return PerturbationRequest(**values)


def test_custom_pilot_quotas_only_emit_camera_robot():
    requests = sample_perturbations(
        20,
        17,
        dimension_quotas={"camera": 1, "robot": 1},
        suite_quotas={"Spatial": 1},
    )
    assert {item.dimension for item in requests} == {"camera", "robot"}
    assert {item.suite for item in requests} == {"Spatial"}
    assert len(requests) == 20


def test_catalog_selection_filters_suite_category_and_level_deterministically():
    catalog = {
        "libero_spatial": (
            _CatalogTask("libero_spatial", 1, "camera-l2", "Camera Viewpoints", 2),
            _CatalogTask("libero_spatial", 2, "camera-l3-a", "Camera Viewpoints", 3),
            _CatalogTask("libero_spatial", 3, "camera-l3-b", "Camera Viewpoints", 3),
            _CatalogTask("libero_spatial", 4, "robot-l3", "Robot Initial States", 3),
        )
    }
    first = select_catalog_task(catalog, request(seed=7))
    second = select_catalog_task(catalog, request(seed=7))
    assert first == second
    assert first.category == "Camera Viewpoints"
    assert first.level == 3
    assert first.task_id in {2, 3}


def test_other_subdimension_maps_to_upstream_category():
    catalog = {
        "libero_goal": (
            _CatalogTask("libero_goal", 11, "noise-l4", "Sensor Noise", 4),
        )
    }
    selected = select_catalog_task(
        catalog,
        request(
            suite="Goal",
            dimension="other",
            subdimension="noise",
            level=4,
        ),
    )
    assert selected.task_id == 11


def test_combination_fails_until_paired_protocol_is_validated():
    with pytest.raises(ValueError, match="no camera\\+robot combination category"):
        select_catalog_task(
            {},
            request(
                dimension="combination",
                subdimension="camera+robot",
            ),
        )


def test_load_catalog_skips_null_difficulty(tmp_path):
    path = tmp_path / "task_classification.json"
    path.write_text(
        json.dumps(
            {
                "libero_goal": [
                    {
                        "id": 1,
                        "name": "ok",
                        "category": "Camera Viewpoints",
                        "difficulty_level": 3,
                    },
                    {
                        "id": 2,
                        "name": "null-level",
                        "category": "Light Conditions",
                        "difficulty_level": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = _load_catalog(path)
    assert len(catalog["libero_goal"]) == 1
    assert catalog["libero_goal"][0].task_id == 1

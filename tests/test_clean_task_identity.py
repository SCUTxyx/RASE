"""Fail-closed tests: clean controls must use official names, not Plus variants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rase.backends.libero_clean import (
    N_CLEAN_TASKS,
    assert_clean_task_name,
    build_clean_suite,
    clean_task_name,
    load_clean_task_names,
)
from rase.collect.lerobot_libero_plus_adapter import select_catalog_task
from rase.collect.perturb_sampler import PerturbationRequest
from rase.eval.collapse import CollapseError


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "clean_libero_task_names.json"


def _request(**changes):
    values = {
        "index": 0,
        "suite": "Object",
        "dimension": "clean",
        "subdimension": "none",
        "level": 0,
        "seed": 0,
        "task_id": 1,
    }
    values.update(changes)
    return PerturbationRequest(**values)


def test_frozen_catalog_has_ten_official_names_per_suite():
    suites = load_clean_task_names(CATALOG)
    assert set(suites) == {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }
    for suite, names in suites.items():
        assert len(names) == N_CLEAN_TASKS
        for name in names:
            assert_clean_task_name(name)


def test_assert_clean_task_name_rejects_plus_layout_variants():
    for bad in (
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_2",
        "open_the_middle_drawer_of_the_cabinet_tb_3",
    ):
        with pytest.raises(CollapseError, match="refusing perturbed"):
            assert_clean_task_name(bad)
    # Official clean name contains "table" as a word, not a Plus suffix.
    assert_clean_task_name(
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    )


def test_select_catalog_task_clean_resolves_frozen_official_names():
    suites = load_clean_task_names(CATALOG)
    for suite_label, suite_key in (
        ("Spatial", "libero_spatial"),
        ("Object", "libero_object"),
        ("Goal", "libero_goal"),
        ("Long", "libero_10"),
    ):
        for task_id in range(1, N_CLEAN_TASKS + 1):
            selected = select_catalog_task(
                {},
                _request(suite=suite_label, task_id=task_id, seed=task_id),
            )
            assert selected.suite == suite_key
            assert selected.task_id == task_id
            assert selected.name == suites[suite_key][task_id - 1]
            assert selected.name == clean_task_name(suite_key, task_id)
            assert_clean_task_name(selected.name)


def test_plus_index_zero_through_nine_are_not_treated_as_clean_names(monkeypatch):
    """Regression: Plus suite index 0..9 are layout variants, not clean-10."""
    # Even if a Plus suite were injected, clean selection must not accept
    # variant names from catalog_task_to_suite_index(1..10).
    selected = select_catalog_task({}, _request(suite="Object", task_id=1))
    assert selected.name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    assert "table" not in selected.name


def test_build_clean_suite_n_tasks_ten_and_exact_names():
    suite = build_clean_suite(
        "libero_object", clean_root="/root/autodl-tmp/src/LIBERO"
    )
    assert suite.n_tasks == N_CLEAN_TASKS
    expected = load_clean_task_names(CATALOG)["libero_object"]
    assert tuple(suite.get_task_names()) == expected
    for name in suite.get_task_names():
        assert_clean_task_name(name)


def test_long_suite_language_strips_scene_prefix():
    suite = build_clean_suite(
        "libero_10", clean_root="/root/autodl-tmp/src/LIBERO"
    )
    languages = [task.language for task in suite.tasks]
    assert languages[0] == (
        "put both the alphabet soup and the tomato sauce in the basket"
    )
    assert all("SCENE" not in lang for lang in languages)
    assert all(not lang.startswith("LIVING ROOM") for lang in languages)
    assert all(not lang.startswith("KITCHEN") for lang in languages)


def test_collect_config_points_at_w9c_pool_and_schedule():
    cfg = json.loads(
        (ROOT / "configs" / "collect_w9c_clean_controls.json").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["protocol"]["version"] == "W9C-clean-control/v1"
    assert cfg["collection"]["output_dir"] == "pool/ngc_w9c_clean_controls"
    assert "w9b" not in cfg["collection"]["output_dir"]
    assert cfg["adapter_config"]["libero_clean_root"]

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_rase_vnext_protocol import validate


CONFIG = Path(__file__).parents[1] / "configs" / "rase_vnext_protocol_v1.json"


def test_frozen_protocol_passes_with_all_scientific_choices_resolved() -> None:
    config = json.loads(CONFIG.read_text())
    assert validate(config, allow_draft=False) == []
    assert [row["value"] for row in config["collection"]["decision_points"]] == [8, 16]


def test_future_dependent_fractional_decision_points_are_rejected() -> None:
    config = json.loads(CONFIG.read_text())
    config["collection"]["decision_points"][0] = {
        "decision_point_id": "source.fraction.0_33",
        "rule": "source_elapsed_fraction",
        "value": 0.33,
    }
    assert any("source_elapsed_step" in error for error in validate(config, allow_draft=False))


def test_frozen_protocol_cannot_have_unresolved_scientific_choices() -> None:
    config = json.loads(CONFIG.read_text())
    config["utility"]["harm_weight"] = None
    errors = validate(config, allow_draft=False)
    assert any("unresolved" in error for error in errors)


def test_candidates_cannot_inflate_operator_prior() -> None:
    config = json.loads(CONFIG.read_text())
    config["operators"].append(copy.deepcopy(config["operators"][2]))
    assert any("five semantic" in error for error in validate(config, allow_draft=True))

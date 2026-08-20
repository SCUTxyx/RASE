from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_rase_vnext_protocol import validate


CONFIG = Path(__file__).parents[1] / "configs" / "rase_vnext_protocol_v1.json"


def test_draft_protocol_passes_structural_validation_only() -> None:
    config = json.loads(CONFIG.read_text())
    assert validate(config, allow_draft=True) == []
    assert "draft_locked" in validate(config, allow_draft=False)[0]


def test_frozen_protocol_cannot_have_unresolved_scientific_choices() -> None:
    config = json.loads(CONFIG.read_text())
    config["status"] = "frozen"
    errors = validate(config, allow_draft=False)
    assert any("unresolved" in error for error in errors)


def test_candidates_cannot_inflate_operator_prior() -> None:
    config = json.loads(CONFIG.read_text())
    config["operators"].append(copy.deepcopy(config["operators"][2]))
    assert any("five semantic" in error for error in validate(config, allow_draft=True))

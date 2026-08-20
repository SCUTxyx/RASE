"""Verify LightRiskStudent has no V-JEPA dependency at import time."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_no_vjepa_import():
    """LightRiskStudent and its dependencies must not import V-JEPA."""
    allowed_modules = [
        "rase.risk.light_risk_student",
        "rase.risk.tiny_universal_state_encoder",
        "rase.risk.canonical_action",
        "rase.risk.conformal_calibrator",
        "rase.controllers.safe_handback_controller",
    ]
    modules_before = set(sys.modules)
    for mod_name in allowed_modules:
        __import__(mod_name)
    newly_imported = set(sys.modules) - modules_before
    forbidden = sorted(
        name for name in newly_imported
        if "vjepa" in name.lower() or name.startswith("rase.world_models")
    )
    assert not forbidden, f"deploy import pulled teacher modules: {forbidden}"


if __name__ == "__main__":
    test_no_vjepa_import()

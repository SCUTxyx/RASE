from __future__ import annotations

import importlib.util
from pathlib import Path


def load_analyzer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_continue_fallback_opportunity.py"
    spec = importlib.util.spec_from_file_location("g2b_analyzer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_direct_result_provenance() -> None:
    module = load_analyzer()
    arm = module._direct_arm(
        {
            "result": {
                "prefix_source": "direct",
                "prefix_steps": 0,
                "env_steps": 17,
                "stop_reason": "success",
                "success": True,
            }
        }
    )
    assert arm["success"] is True
    assert arm["env_steps"] == 17

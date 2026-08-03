import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rollout_smol_interventions.py"
    spec = importlib.util.spec_from_file_location("rollout_smol_interventions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mcnemar_exact_handles_ties_and_one_sided_disagreement():
    module = _module()
    assert module._mcnemar_exact(0, 0) == 1.0
    assert module._mcnemar_exact(1, 0) == 1.0
    assert module._mcnemar_exact(6, 0) == 0.03125
    assert module._mcnemar_exact(2, 2) == 1.0

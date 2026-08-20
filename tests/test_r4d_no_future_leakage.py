#!/usr/bin/env python3
"""Verify the R4-D training pipeline has no future leakage.

Checks that the LightRiskStudent training script never consumes labels or
features that would be unavailable at deployment time:
  - no `task_ordinal` as a model feature (comments/docstrings allowed)
  - no real `remaining_teacher_steps` as a model input (cost is predicted)
  - no V-JEPA import in risk modules
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove comments, strings, and docstrings so only executable code remains."""
    tree = ast.parse(src)
    # Remove docstring nodes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
    # Extract only attribute/name/load contexts from the tree
    code_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            code_names.append(node.attr)
        elif isinstance(node, ast.Name):
            code_names.append(node.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                code_names.append(node.value)
    return "\n".join(code_names)


def main() -> int:
    failures = []

    def check_forbidden(path: str, needles: list[str]) -> None:
        text = _strip_comments_and_docstrings(source(path))
        for needle in needles:
            if needle in text:
                failures.append(f"{path}: executable code contains {needle!r}")

    # 1. task_ordinal must not appear as an executable token
    for path in (
        "scripts/train_r4_safe_handback_wm_ridge.py",
        "scripts/train_r4d_light_risk_student.py",
        "rase/risk/light_risk_student.py",
        "rase/risk/tiny_universal_state_encoder.py",
        "rase/risk/canonical_action.py",
    ):
        check_forbidden(path, ["task_ordinal"])

    # 2. remaining_teacher_steps may appear only as a training *label* in
    #    build_batch. It must never be an input to the model forward call.
    train_src = source("scripts/train_r4d_light_risk_student.py")
    # The only allowed uses are in the label arrays inside build_batch/evaluate.
    for needle in ("remaining_teacher_steps",):
        # Ban it anywhere in the model-forward call sites: find each "out = model("
        # and ensure the argument block does not reference it.
        import re

        for m in re.finditer(r"out\s*=\s*model\(", train_src):
            call_body = train_src[m.end():m.end() + 2000].split(")")[0]
            if needle in call_body:
                failures.append(f"train_r4d_light_risk_student.py: {needle} passed into model call")

    # 3. Risk modules must not import V-JEPA / world-model / teacher packages
    for path in (
        "rase/risk/light_risk_student.py",
        "rase/risk/tiny_universal_state_encoder.py",
        "rase/risk/canonical_action.py",
        "rase/risk/vla_action_adapters.py",
        "rase/controllers/safe_handback_controller.py",
        "rase/risk/conformal_calibrator.py",
    ):
        check_forbidden(path, ["vjepa", "world_models", "Oracle", "vla_teacher"])

    # 4. The controller must not take remaining_teacher_steps as input
    check_forbidden("rase/controllers/safe_handback_controller.py", ["remaining_teacher_steps"])

    if failures:
        print("FUTURE LEAKAGE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("No future-leakage violation found.")
    return 0


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

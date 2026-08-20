from __future__ import annotations

import pytest

from scripts.collect_r6b1_dynamic_boundaries import is_invalid_action_token_error


def test_matches_pi0fast_action_grammar_assertion() -> None:
    exc = AssertionError(
        "Token sequence does not start with ['Action', ':']: ['Sub', 'boxes']"
    )
    assert is_invalid_action_token_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        AssertionError("different model invariant"),
        RuntimeError("Token sequence does not start with ['Action', ':']"),
        ValueError("decoder failed"),
    ],
)
def test_rejects_unrelated_exceptions(exc: BaseException) -> None:
    assert not is_invalid_action_token_error(exc)

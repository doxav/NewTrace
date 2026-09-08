"""Portable optimizer contract, real subprocess and exact fixture accounting."""

import os
from pathlib import Path

import pytest

from opto.features.recursive_opt.optimizer_program import (
    evaluate_program,
    parse_program,
    propose_point,
)

VALID = "def propose(history, bounds, seed):\n    return [0.0] * len(bounds)\n"


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("def :", "syntax_error"),
        ("import no_such_phase0_module", "import_error"),
        ("x = 1", "missing_propose"),
        ("def propose(a): return a", "signature_error"),
        ('def propose(history, bounds, seed): raise RuntimeError("bad")', "exception"),
        ("def propose(history, bounds, seed):\n while True: pass", "timeout"),
        ("def propose(history, bounds, seed): return [0]", "shape_error"),
        ('def propose(history, bounds, seed): return [float("nan"), 0]', "nonfinite"),
        ('def propose(history, bounds, seed): return [float("inf"), 0]', "nonfinite"),
        ("def propose(history, bounds, seed): return [6, 0]", "out_of_bounds"),
        (
            "import random\ndef propose(history, bounds, seed): return [random.random(), 0]",
            "nondeterministic",
        ),
        (VALID, "valid"),
    ],
)
def test_validation(code: str, status: str) -> None:
    """Invalid execution never becomes an objective value."""
    result = propose_point(code, [], [[-5, 5], [-5, 5]], 0, timeout_s=0.5)
    assert result.status == status
    assert result.valid is (status == "valid")
    assert (result.point is not None) is result.valid


def test_fixture_budget_and_repeatability() -> None:
    """Only objective evaluations consume the black-box budget."""
    first = evaluate_program(VALID, seed=2, budget=3)
    second = evaluate_program(VALID, seed=2, budget=3)
    assert first.valid and first.evaluations == second.evaluations
    assert first.evaluated_count == 3
    assert first.best_value == 2.125
    assert first.behavior_signature == second.behavior_signature
    failed = evaluate_program(
        "def propose(history, bounds, seed):\n return [0,0] if not history else [9,9]",
        seed=2,
        budget=3,
    )
    assert not failed.valid and failed.evaluated_count == 1
    assert failed.best_value is None
    assert len(failed.evaluations) == 1


def test_environment_and_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate receives no inherited credentials or dataset metadata."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-only")
    code = """import os
def propose(history, bounds, seed):
    assert 'OPENROUTER_API_KEY' not in os.environ
    assert 'PYTHONPATH' not in os.environ
    assert all(set(row) == {'x', 'value'} for row in history)
    assert not os.path.exists('AGENTS.md')
    return [0, 0]
"""
    assert evaluate_program(code, seed=0, budget=2).valid
    assert os.environ["OPENROUTER_API_KEY"] == "secret-test-only"


@pytest.mark.parametrize(
    "bounds", [[], [[1, 0]], [[0, float("inf")]], [[0]], [[False, 1]]]
)
def test_invalid_bounds(bounds: list) -> None:
    """Reject malformed host inputs before executing code."""
    with pytest.raises(ValueError):
        propose_point(VALID, [], bounds, 0)


def test_input_and_parser() -> None:
    """Parsing accepts plain code or one code fence without repairing programs."""
    assert parse_program("```python\n" + VALID + "```") == VALID.strip() + "\n"
    assert parse_program(VALID) == VALID
    with pytest.raises(ValueError):
        parse_program("```python\nx=1\n```\n```python\ny=2\n```")
    with pytest.raises(ValueError):
        propose_point(
            VALID, [{"x": [0, 0], "value": 1, "holdout": True}], [[-5, 5]] * 2, 0
        )
    with pytest.raises(ValueError):
        evaluate_program(VALID, seed=0, budget=0)


def test_portable_file(tmp_path: Path) -> None:
    """An optimizer.py file is the complete artifact."""
    path = tmp_path / "optimizer.py"
    path.write_text(VALID)
    assert evaluate_program(path.read_text(), seed=0, budget=1).valid

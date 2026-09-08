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


def test_worker_has_no_objective_or_installed_packages() -> None:
    """The child runtime exposes proposal validation, never the parent objective."""
    code = """import __main__
def propose(history, bounds, seed):
    assert not hasattr(__main__, 'evaluate_program')
    return [0, 0]
"""
    assert propose_point(code, [], [[-5, 5]] * 2, 0).valid
    assert (
        propose_point("import numpy\n" + VALID, [], [[-5, 5]] * 2, 0).status
        == "import_error"
    )


def test_seeded_randomness_and_mutation() -> None:
    """Local seeded random sampling is reproducible and cannot mutate parent inputs."""
    code = """import random
def propose(history, bounds, seed):
    rng = random.Random(seed + len(history))
    history.clear()
    return [rng.uniform(lo, hi) for lo, hi in bounds]
"""
    history = [{"x": [0, 0], "value": 2.125}]
    assert propose_point(code, history, [[-5, 5]] * 2, 4).valid
    assert len(history) == 1


def test_canonical_optimizer_artifact_and_typed_invalidity(tmp_path: Path) -> None:
    """The canonical module stores optimizer.py source and returns typed fixture feedback."""
    from opto.features.recursive_opt import spec as S
    from opto.features.recursive_opt.optimizer_program import optimizer_spec

    raw = optimizer_spec(VALID, seed=0, budget=2)
    raw["outputs"] = {"directory": str(tmp_path)}
    result = S.execute_plan(S.compile_plan(raw))[0]
    assert result.valid and result.portable
    assert result.evaluation.metrics["value"] == 2.125
    assert result.artifact["components"]["optimizer"] == VALID
    assert result.budget["accounted"]["evaluator_runs"] == 1
    assert result.evaluation.artifacts[0]["evaluated_count"] == 2
    raw = optimizer_spec(
        "def propose(history, bounds, seed): return [99, 0]", seed=0, budget=2
    )
    invalid = S.execute_plan(S.compile_plan(raw))[0]
    assert not invalid.valid
    assert invalid.status == "invalid"
    assert not invalid.evaluation.metrics
    assert invalid.error == "out_of_bounds"


def test_optimizer_artifact_is_trainable_through_real_trace() -> None:
    """A real trainer update reaches source text and receives deterministic feedback."""
    from typing import Any

    from opto.features.recursive_opt import spec as S
    from opto.features.recursive_opt.optimizer_program import optimizer_spec
    from opto.optimizers.optimizer import Optimizer

    improved = "def propose(history, bounds, seed):\n return [1.0, -1.0]\n"

    class Change(Optimizer):
        """Use the existing optimizer update protocol with a deterministic candidate."""

        def _step(self, *args: Any, **kwargs: Any) -> dict[Any, str]:
            """Replace the trainable optimizer source."""
            return {p: improved for p in self.parameters}

    raw = optimizer_spec(VALID, seed=0, budget=2, engine="trace")
    raw["runtime"]["test_mode"] = True
    result = S.execute_plan(S.compile_plan(raw), {"optimizer": Change})[0]
    assert result.valid and result.evaluation.metrics["value"] == 0.125
    assert result.artifact["components"]["optimizer"] == improved
    assert result.metadata["menu_evidence"]["effective_menu_size"] == 2
    assert (
        result.metadata["menu_evidence"]["basis_of_equivalence"] == "behavior_signature"
    )


def test_scalar_trace_continues_after_invalid_candidate_without_metric_imputation() -> (
    None
):
    """A typed-invalid proposal cannot crash scalar ranking or become a numeric objective."""
    from typing import Any

    from opto.features.recursive_opt import spec as S
    from opto.features.recursive_opt.optimizer_program import optimizer_spec
    from opto.optimizers.optimizer import Optimizer

    calls: list[int] = []

    class Change(Optimizer):
        """First emit invalid syntax, then a valid improving source."""

        def _step(self, *args: Any, **kwargs: Any) -> dict[Any, str]:
            """Use the real trainer's next proposal even after an invalid result."""
            calls.append(1)
            source = (
                "def :"
                if len(calls) == 1
                else "def propose(history, bounds, seed): return [1.0, -1.0]"
            )
            return {p: source for p in self.parameters}

    raw = optimizer_spec(VALID, seed=0, budget=2, engine="trace")
    raw["runtime"]["test_mode"] = True
    raw["datasets"]["validation"] = []
    raw["engine"]["config"]["iterations"] = 3
    result = S.execute_plan(S.compile_plan(raw), {"optimizer": Change})[0]
    assert result.valid and len(calls) == 2
    assert result.evaluation.metrics["value"] == 0.125
    records = result.level_results[0]["metadata"]["evaluator_records"]
    invalid = [r for r in records if not r["valid"]]
    assert invalid and all(not r["metrics"] for r in invalid)

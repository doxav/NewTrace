from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest

from examples.recursive_opt_three_way import make_solver_critic_evaluator
from opto.features.recursive_opt import CodeArtifactLevel, ComponentSpec, MemoryLite, RecursiveGuide


def _critic_seed(self: Any, critic_input: str) -> str:
    """Return a corrected answer only when the solver draft is visible."""
    return "fixed" if "solver_draft: draft:q1" in str(critic_input) else "missing"


def test_solver_critic_evaluator_invokes_critic_on_traced_path(tmp_path: Path) -> None:
    calls: List[Tuple[str, Any]] = []

    def solver_fn(question: Any) -> str:
        calls.append(("solver", question))
        return f"draft:{question}"

    def base_evaluate(candidate: Callable[..., Any], task: Any) -> Tuple[float, str]:
        calls.append(("base", task))
        answer = candidate(question="q1")
        return (1.0 if answer == "fixed" else 0.0), f"answer={answer}"

    level = CodeArtifactLevel(
        ComponentSpec(
            name="critic",
            baseline=_critic_seed,
            evaluate=make_solver_critic_evaluator(
                solver_fn=solver_fn,
                base_evaluate=base_evaluate,
            ),
        ),
        memory=MemoryLite(root=str(tmp_path)),
    )

    score, feedback = RecursiveGuide()("task", level.forward("task"), None)

    assert score == pytest.approx(1.0)
    assert "answer=fixed" in feedback
    assert calls == [("base", "task"), ("solver", "q1")]
    assert level._last_node is not None


def test_solver_critic_evaluator_falls_back_to_solver_draft() -> None:
    def solver_fn(question: Any) -> str:
        return f"draft:{question}"

    def failing_critic(_critic_input: str) -> str:
        raise RuntimeError("critic failed")

    def base_evaluate(candidate: Callable[..., Any], task: Any) -> Tuple[str, str]:
        return str(candidate(question="q2")), str(task)

    evaluator = make_solver_critic_evaluator(
        solver_fn=solver_fn,
        base_evaluate=base_evaluate,
    )

    answer, feedback = evaluator(failing_critic, "task")

    assert answer == "draft:q2"
    assert feedback == "task"

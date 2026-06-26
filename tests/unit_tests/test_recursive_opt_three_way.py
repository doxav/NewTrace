from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest

from examples.recursive_opt_three_way import make_code_arm, make_solver_critic_evaluator
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


def _policy_seed(self: Any, signal: str) -> str:
    """Tiny importable policy baseline for code-arm transfer tests."""
    return f"seed:{signal}"


def test_code_arm_transfer_phase2_optimizes_heldout_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    optimize_families: List[str] = []
    train_families: List[str] = []
    target_families: List[str] = []

    def fake_optimize(level: CodeArtifactLevel, dataset: dict, **_kwargs: Any) -> None:
        family = str(dataset["inputs"][0])
        optimize_families.append(family)
        level.forward(family)

    monkeypatch.setattr("opto.features.recursive_opt.optimize", fake_optimize)

    def train_eval(component: Callable[..., Any], family: Any) -> Tuple[float, str]:
        train_families.append(str(family))
        component("train")
        return 0.2, "source score"

    def target_eval(component: Callable[..., Any], family: Any) -> Tuple[float, str]:
        target_families.append(str(family))
        component("target")
        return 0.9, "target score"

    runner = make_code_arm(
        baseline=_policy_seed,
        evaluate=train_eval,
        task_id="source_task",
        objective="transfer source policy to target",
        warm=True,
        transfer=True,
        transfer_phase2=True,
        holdout_task_id="target_task",
        holdout_evaluate=target_eval,
    )

    result = runner(
        "recursive",
        {"_component": "policy", "_total_candidates": 4, "_num_candidates": 2, "_max_examples": 1},
        0,
        None,
        {"optimizer_llm_calls": 10, "eval_llm_calls": 10, "candidates": 10, "on_exceed": "return_best"},
        tmp_path,
    )

    assert result.error is None
    assert result.score == pytest.approx(0.9)
    assert optimize_families == ["source_task", "target_task"]
    assert train_families == ["source_task"]
    assert target_families
    assert target_families[-1] == "target_task"

"""The scripted regression runner must work whenever Trace-Bench is installed.

NOTE on the backend requirement. Example C's capability evaluation calls
``param.forward(...)`` on a Trace-Bench learner, which is a real provider call.
Without a configured LLM backend it raises ``OpenAIError`` (no api_key).

This test used to pass in exactly that situation, because
``make_multiobjective_evaluator`` swallowed *every* exception and returned a
plausible-looking ``({"accuracy": 0.0, "cost": 1.0}, ..., -0.5)`` for each
candidate. The assertions below were therefore satisfied by a run in which every
evaluation had failed and the "Pareto-best capability" was chosen among
identically-zero results. ``episodes >= 1`` held because the artifact records to
memory whether or not the evaluation succeeded.

That swallow is now removed (assessment defect D2), so the requirement is
explicit: skip without a backend rather than assert a number produced by a failure.
"""
import pytest

from opto.features.recursive_opt.runmode import have_key
from opto.features.recursive_opt.tracebench import HAVE_TB


@pytest.mark.skipif(not HAVE_TB, reason="Trace-Bench not installed (real adapter required)")
@pytest.mark.skipif(
    not have_key(),
    reason="example C evaluates through a real provider call; without a backend key the "
           "evaluation genuinely cannot run (it previously 'passed' only because the "
           "evaluator swallowed the failure into a score — see D2)",
)
def test_review_regression_returns_rows() -> None:
    from examples.recursive_opt_review_regression import run_review_regression
    rows = run_review_regression()
    assert [r["example"] for r in rows] == ["A", "C"]
    assert all(r["memory"]["episodes"] >= 1 for r in rows)


@pytest.mark.skipif(not HAVE_TB, reason="Trace-Bench not installed (real adapter required)")
def test_capability_evaluator_surfaces_backend_failure_instead_of_scoring_it() -> None:
    """Without a backend the evaluator must RAISE, not return a plausible score.

    This is the guard for D2: a provider/wiring failure must never be
    indistinguishable from a genuinely poor capability.
    """
    from opto.features.recursive_opt.tracebench import (
        ensure_eval_only_task_adapter, make_multiobjective_evaluator)

    if have_key():
        pytest.skip("a backend is configured, so the failure path is not exercised here")

    ensure_eval_only_task_adapter(require=True)
    evaluate = make_multiobjective_evaluator(
        ["internal:multiobjective_gsm8k"], {"accuracy": "max", "cost": "min"}
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate(lambda task=None: {"answer": "Answer directly.", "task": task},
                 "internal:multiobjective_gsm8k")

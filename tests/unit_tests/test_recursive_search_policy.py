"""Item 6: learnable active-search / lessons-learnt tool."""
from __future__ import annotations

import tempfile
import pytest

from opto.features.recursive_opt.capabilities import (
    run_search_policy, make_search_policy_tool, make_search_policy_evaluator,
    DEFAULT_SEARCH_POLICY,
)
from opto.features.recursive_opt.memory import MemoryLite


def _memory_with_failures():
    mem = MemoryLite(root=tempfile.mkdtemp())
    for i, fb in enumerate(["off-by-one in loop", "wrong base case handling",
                            "forgot to handle empty", "index error on edge"]):
        mem.record(level="O1", cfg={}, family="codegen", score=0.0, feedback=fb)
    return mem


def test_different_policies_yield_different_lessons() -> None:
    mem = _memory_with_failures()
    recent = run_search_policy({"k": 2, "strategy": "recent", "template": "{lessons}"},
                               mem, family="codegen")
    diverse = run_search_policy({"k": 2, "strategy": "diverse", "template": "{lessons}"},
                                mem, family="codegen")
    assert recent and diverse and recent != diverse   # the policy is causal


def test_template_is_applied() -> None:
    mem = _memory_with_failures()
    out = run_search_policy({"k": 1, "strategy": "recent", "template": "LESSON: {lessons}"},
                            mem, family="codegen")
    assert out.startswith("LESSON: ")


def test_empty_memory_returns_empty() -> None:
    mem = MemoryLite(root=tempfile.mkdtemp())
    assert run_search_policy(DEFAULT_SEARCH_POLICY, mem, family="x") == ""
    assert run_search_policy(DEFAULT_SEARCH_POLICY, None) == ""


def test_tool_is_function_callable() -> None:
    mem = _memory_with_failures()
    tool = make_search_policy_tool({"k": 2, "strategy": "recent"}, mem, family="codegen")
    assert callable(tool)
    assert tool("any feedback")          # matches other tools' (fb)->str signature


def test_evaluator_rewards_helpful_policy() -> None:
    """A policy whose lesson surfaces the relevant cue must out-score one that doesn't."""
    mem = _memory_with_failures()

    def base_eval(prior):                 # synthetic: 'base case' in prior helps
        return 0.5 + (0.4 if "base case" in prior else 0.0)

    ev = make_search_policy_evaluator(mem, base_eval, family="codegen")
    broad_lift, fb = ev({"k": 4, "strategy": "all", "template": "{lessons}"})
    narrow_lift, _ = ev({"k": 1, "strategy": "recent", "template": "{lessons}"})
    assert broad_lift > narrow_lift       # optimizing the policy is meaningful
    assert broad_lift == pytest.approx(0.4)
    assert "lift=" in fb                  # progress is reported

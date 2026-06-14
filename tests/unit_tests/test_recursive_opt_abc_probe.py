from __future__ import annotations

from typing import Any, Dict

import pytest

from examples import recursive_opt_abc_probe as probe


def test_waste_evaluator_rewards_fast_equivalent_solver() -> None:
    def fast_solver(question: str) -> str:
        expr = str(question).strip()
        if expr.endswith(" is"):
            expr = expr[:-3].strip()
        try:
            value = eval(expr, {"__builtins__": {}}, {"True": True, "False": False})
        except Exception:
            value = False
        return "True" if bool(value) else "False"

    evaluator = probe.make_waste_evaluator(repeats=1)

    def slow_solver(question: str) -> str:
        return probe.wasteful_bool_solver(None, question)

    _slow_metrics, _slow_feedback, slow_score = evaluator(slow_solver, None)
    _fast_metrics, _fast_feedback, fast_score = evaluator(fast_solver, None)

    assert fast_score > slow_score


def test_bool_tool_graph_oracle_beats_draft_route() -> None:
    pytest.importorskip("langgraph")
    from opto.features.graph import LangGraphAdapter

    examples = [
        ("not True is", "False"),
        ("not False is", "True"),
        ("True and True is", "True"),
        ("False and True is", "False"),
    ]
    adapter = LangGraphAdapter(
        graph_factory=probe.build_bool_tool_graph,
        function_targets={
            "graph_draft_agent": probe.graph_draft_agent,
            "graph_tool_agent": probe.graph_tool_agent,
            "graph_merge_agent": probe.graph_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="question",
        output_key="answer",
    )
    module = adapter.as_module()
    guide = probe.BooleanAnswerGuide()

    draft_score = probe._score_graph_module(module, examples, guide)
    adapter.graph_knobs["route_policy"]._set("tool")
    tool_score = probe._score_graph_module(module, examples, guide)

    assert draft_score < tool_score
    assert tool_score == pytest.approx(1.0)


def test_graph_tool_agent_unwraps_trace_nodes() -> None:
    from opto.trace import node

    state: Dict[str, Any] = {"question": node("not False is", name="question")}

    assert probe.graph_tool_agent(state)["tool_answer"] == "True"


def test_graph_output_keeps_route_policy_on_trace_path() -> None:
    pytest.importorskip("langgraph")
    from opto.features.graph import LangGraphAdapter

    adapter = LangGraphAdapter(
        graph_factory=probe.build_bool_tool_graph,
        function_targets={
            "graph_draft_agent": probe.graph_draft_agent,
            "graph_tool_agent": probe.graph_tool_agent,
            "graph_merge_agent": probe.graph_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="question",
        output_key="answer",
    )
    module = adapter.as_module()
    output = module.forward("not True is")
    route_param = adapter.graph_knobs["route_policy"]

    def walk(node_obj: Any, seen: set[int] | None = None) -> bool:
        seen = seen or set()
        if id(node_obj) in seen:
            return False
        seen.add(id(node_obj))
        if node_obj is route_param:
            return True
        parents = getattr(node_obj, "parents", None) or getattr(node_obj, "_inputs", None) or []
        if isinstance(parents, dict):
            parents = parents.values()
        return any(walk(parent, seen) for parent in parents)

    assert walk(output)


def test_suboptimizer_graph_oracle_beats_draft_route() -> None:
    pytest.importorskip("langgraph")
    from opto.features.graph import LangGraphAdapter

    centers = [-3.5, 1.75, 4.25, 7.0]
    adapter = LangGraphAdapter(
        graph_factory=probe.build_suboptimizer_graph,
        function_targets={
            "subopt_draft_agent": probe.subopt_draft_agent,
            "scipy_suboptimizer_agent": probe.scipy_suboptimizer_agent,
            "subopt_merge_agent": probe.subopt_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="center",
        output_key="x",
        train_graph_agents_functions=False,
    )
    module = adapter.as_module()

    draft_score = probe._score_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("scipy")
    scipy_score = probe._score_suboptimizer_module(module, centers)

    assert draft_score < scipy_score
    assert scipy_score == pytest.approx(1.0)


def test_conditional_suboptimizer_route_beats_always_tool_and_draft() -> None:
    pytest.importorskip("langgraph")
    from opto.features.graph import LangGraphAdapter

    centers = [-3.5, -0.1, 0.0, 0.2, 1.75, 4.25]
    adapter = LangGraphAdapter(
        graph_factory=probe.build_conditional_suboptimizer_graph,
        function_targets={
            "subopt_draft_agent": probe.subopt_draft_agent,
            "scipy_suboptimizer_agent": probe.scipy_suboptimizer_agent,
            "conditional_subopt_merge_agent": probe.conditional_subopt_merge_agent,
        },
        graph_knobs={"route_policy": "draft"},
        input_key="center",
        output_key="result",
        train_graph_agents_functions=False,
    )
    module = adapter.as_module()

    draft_score = probe._score_conditional_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("scipy")
    always_tool_score = probe._score_conditional_suboptimizer_module(module, centers)
    adapter.graph_knobs["route_policy"]._set("conditional")
    conditional_score = probe._score_conditional_suboptimizer_module(module, centers)

    assert draft_score < always_tool_score < conditional_score
    assert conditional_score == pytest.approx(0.875)


def test_suboptimizer_agent_unwraps_trace_nodes() -> None:
    from opto.trace import node

    state: Dict[str, Any] = {"center": node(2.5, name="center")}

    out = probe.scipy_suboptimizer_agent(state)

    assert abs(float(out["tool_x"]) - 2.5) < 1e-4

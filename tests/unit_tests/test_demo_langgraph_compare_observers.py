import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


pytest.importorskip("langgraph.graph")


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples/notebooks/demo_langgraph_instrument_and_compare_observers.py"
    spec = importlib.util.spec_from_file_location("compare_observers_demo_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_demo_module_with_env(**env_overrides):
    saved = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return _load_demo_module()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _graph_source(graph):
    if graph is None:
        pytest.skip("graphviz Python package is not installed")
    return graph.source


def _attr(key, value):
    payload = {}
    if isinstance(value, bool):
        payload["boolValue"] = value
    else:
        payload["stringValue"] = str(value)
    return {"key": key, "value": payload}


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def _fake_llm(messages=None, **_kwargs):
    user = next((m["content"] for m in (messages or []) if m.get("role") == "user"), "")
    if "Create a short plan for:" in user:
        query = user.split("Create a short plan for:", 1)[1].strip()
        return _Resp(f"Plan for {query}: define mechanism, examples, caveats.")
    if "Answer directly in the first sentence." in user:
        return _Resp(
            "CRISPR is a gene-editing technology.\n"
            "# Mechanism\n- edits DNA\n"
            "# Examples\n- medicine\n"
            "# Caveats\n- off-target effects"
        )
    return _Resp("Fallback answer")


def test_tgj_to_digraph_preserves_ids_for_otel_dict_nodes():
    demo = _load_demo_module()
    doc = {
        "version": "trace-json/1.0+otel",
        "nodes": {
            "svc:param_planner_prompt": {
                "kind": "parameter",
                "name": "planner_prompt",
                "data": "Create a short plan for: {query}",
                "trainable": True,
            },
            "svc:planner_node": {
                "kind": "msg",
                "name": "planner_node",
                "inputs": {"param_planner_prompt": "svc:param_planner_prompt"},
                "data": {"message_id": "planner_node"},
            },
            "svc:synth_node": {
                "kind": "msg",
                "name": "synth_node",
                "inputs": {"parent": "svc:planner_node"},
                "data": {"message_id": "synth_node"},
            },
        },
    }

    graph = demo.tgj_to_digraph(doc, title="otel-render")
    source = _graph_source(graph)

    assert "planner_prompt" in source
    assert "planner_node" in source
    assert "synth_node" in source
    assert "node_0 -> node_1" in source
    assert "node_1 -> node_2" in source
    assert "svc:planner_node ->" not in source
    assert "svc:synth_node ->" not in source


def test_make_otel_view_merges_multiple_scope_spans_and_docs():
    demo = _load_demo_module()
    otlp = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [_attr("service.name", "svc")]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "scope-a"},
                        "spans": [
                            {
                                "traceId": "trace-1",
                                "spanId": "span-1",
                                "name": "planner_node",
                                "startTimeUnixNano": "1",
                                "attributes": [
                                    _attr("message.id", "planner_node"),
                                    _attr("param.planner_prompt", "Create a short plan for: {query}"),
                                    _attr("param.planner_prompt.trainable", True),
                                ],
                            }
                        ],
                    },
                    {
                        "scope": {"name": "scope-b"},
                        "spans": [
                            {
                                "traceId": "trace-1",
                                "spanId": "span-2",
                                "name": "synth_node",
                                "startTimeUnixNano": "2",
                                "attributes": [
                                    _attr("message.id", "synth_node"),
                                    _attr("param.synth_prompt", "Answer: {query}\\nPlan: {plan}"),
                                    _attr("param.synth_prompt.trainable", True),
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
    }

    view = demo._make_otel_view(otlp, config="otel", origin="backend")
    summary = view["summary"]
    graph = demo.tgj_to_digraph(view["doc"], title="otel-merged")
    source = _graph_source(graph)

    assert summary["span_count"] == 2
    assert summary["span_names"] == ["planner_node", "synth_node"]
    assert summary["semantic_messages"] == ["planner_node", "synth_node"]
    assert summary["param_names"] == ["planner_prompt", "synth_prompt"]
    assert "planner_prompt" in source
    assert "planner_node" in source
    assert "synth_node" in source


def test_tgj_to_digraph_uses_safe_internal_ids_for_colon_node_ids():
    demo = _load_demo_module()
    doc = {
        "tgj": "1.0",
        "nodes": {
            "param:planner_prompt": {
                "id": "param:planner_prompt",
                "kind": "parameter",
                "name": "planner_prompt",
                "value": "Plan {query}",
            },
            "msg:planner": {
                "id": "msg:planner",
                "kind": "message",
                "name": "planner_node",
                "inputs": {"prompt": {"ref": "param:planner_prompt"}},
                "output": {"value": "plan"},
            },
            "msg:synth": {
                "id": "msg:synth",
                "kind": "message",
                "name": "synth_node",
                "inputs": {"parent": {"ref": "msg:planner"}},
                "output": {"value": "answer"},
            },
        },
    }

    graph = demo.tgj_to_digraph(doc, title="safe-ids")
    source = _graph_source(graph)

    assert "node_0" in source
    assert "node_1" in source
    assert "param:planner_prompt -> msg:planner" not in source
    assert "msg:planner -> msg:synth" not in source


def test_env_overrides_can_focus_compare_observers_demo():
    demo = _load_demo_module_with_env(
        COMPARE_OBSERVERS_ITERATIONS="5",
        COMPARE_OBSERVERS_QUERY_LIMIT="1",
        COMPARE_OBSERVERS_CASES="otel",
    )

    cases = demo.build_cases(llm=None)

    assert demo.ITERATIONS == 5
    assert demo.QUERIES == ["What is CRISPR?"]
    assert [name for name, _builder in cases] == ["otel"]


def test_schedule_mode_with_fake_llm_keeps_scores_identical_across_backends():
    demo = _load_demo_module_with_env(OPENROUTER_API_KEY="")
    demo.ITERATIONS = 2
    demo.QUERIES = [demo.QUERIES[0]]

    rows = []
    for name, builder in demo.build_cases(_fake_llm):
        if name not in {"trace", "otel", "sysmon"}:
            continue
        if name == "sysmon" and not hasattr(sys, "monitoring"):
            continue
        rows.append(demo.run_case(name, builder))

    assert len(rows) >= 2
    assert len({tuple(row["score_history"]) for row in rows}) == 1
    assert len({row["baseline_score"] for row in rows}) == 1
    assert len({row["best_score"] for row in rows}) == 1


def test_live_key_switches_demo_cases_to_real_optimizer_mode():
    demo = _load_demo_module_with_env(
        OPENROUTER_API_KEY="sk-test",
        OPENROUTER_MODEL="gemini-3-flash-preview",
    )

    _instrumented, optimizer, _prompt_getter = demo.make_otel_case(_fake_llm)
    assert optimizer is None
    kwargs = demo._comparison_optimizer_kwargs()
    assert kwargs is not None
    assert kwargs["llm"].model_name == "openrouter/google/gemini-3-flash-preview"


def test_compare_report_surfaces_topology_metrics_that_differ_by_backend():
    if not hasattr(sys, "monitoring"):
        return

    demo = _load_demo_module_with_env(OPENROUTER_API_KEY="")
    demo.ITERATIONS = 2
    demo.QUERIES = [demo.QUERIES[0]]
    rows = [
        demo.run_case(name, builder)
        for name, builder in demo.build_cases(_fake_llm)
        if name in {"otel", "sysmon"}
    ]

    assert len(rows) == 2
    assert rows[0]["score_history"] == rows[1]["score_history"]
    assert rows[0]["edge_count"] != rows[1]["edge_count"]

    buf = io.StringIO()
    with redirect_stdout(buf):
        demo.print_cli_report(rows)
    output = buf.getvalue()

    assert "edge_count" in output
    assert "node_count" in output
